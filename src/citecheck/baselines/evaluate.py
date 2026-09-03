"""Evaluation for baseline predictions.

Pure functions: given a list of prediction dicts (as written by
:mod:`citecheck.baselines.runner`) plus a ground-truth map, compute
per-class metrics, a confusion matrix, and cost / latency / search
aggregates.

Reuses :mod:`citecheck.eval.metrics` and
:mod:`citecheck.eval.ground_truth` so the baseline numbers are
directly comparable to the main detector's headline results.

Unlike :func:`citecheck.eval.runner.evaluate_predictions`, this module
**does not re-classify by score** — baselines emit a discrete label
directly, and that label is what we evaluate.  Threshold tuning only
applies to the verifier's continuous score.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from ..eval.dataset import Dataset, load_dataset
from ..eval.ground_truth import build_gt_map
from ..eval.metrics import (
    calculate_metrics,
    confusion_matrix_dict,
    format_confusion_matrix,
    format_metrics_summary,
    print_metrics,
)


__all__ = (
    "evaluate_baseline_predictions",
    "summarize_costs",
    "save_evaluation",
    "per_collection_metrics",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gt_for(
    pred: Mapping[str, Any],
    gt_map: Mapping[str, Mapping[int, str]],
) -> Optional[str]:
    """Look up the ground-truth label for *pred* in *gt_map*.

    *gt_map* is keyed by ``collection_id`` (as produced by
    :func:`citecheck.eval.ground_truth.build_gt_map`).
    """
    cnum = pred.get("citation_number")
    if cnum is None:
        return None
    cnum = int(cnum)

    cid = pred.get("collection_id")
    if cid is not None and cid in gt_map and cnum in gt_map[cid]:
        return gt_map[cid][cnum]

    return None


# ---------------------------------------------------------------------------
# Cost / latency aggregates
# ---------------------------------------------------------------------------

def summarize_costs(predictions: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate cost, tokens, latency, search calls, and parse failures."""
    if not predictions:
        return {
            "n": 0, "total_cost_usd": 0.0,
            "total_prompt_tokens": 0, "total_completion_tokens": 0,
            "mean_cost_per_citation_usd": 0.0,
            "mean_latency_s": 0.0, "p50_latency_s": 0.0, "p95_latency_s": 0.0,
            "max_latency_s": 0.0,
            "total_search_calls": 0, "mean_search_calls": 0.0,
            "parse_failures": 0, "call_errors": 0,
        }

    costs = [float(p.get("cost_usd", 0.0)) for p in predictions]
    in_tok = [int(p.get("prompt_tokens", 0)) for p in predictions]
    out_tok = [int(p.get("completion_tokens", 0)) for p in predictions]
    lat = [float(p.get("latency_s", 0.0)) for p in predictions]
    searches = [int(p.get("web_search_calls", 0)) for p in predictions]
    parse_failures = sum(1 for p in predictions if p.get("parse_failure"))
    call_errors = sum(1 for p in predictions if p.get("call_error"))

    arr_lat = np.array(lat) if lat else np.array([0.0])

    return {
        "n": len(predictions),
        "total_cost_usd": round(sum(costs), 6),
        "mean_cost_per_citation_usd": round(sum(costs) / len(predictions), 6),
        "total_prompt_tokens": int(sum(in_tok)),
        "total_completion_tokens": int(sum(out_tok)),
        "mean_latency_s": round(float(np.mean(arr_lat)), 4),
        "p50_latency_s": round(float(np.percentile(arr_lat, 50)), 4),
        "p95_latency_s": round(float(np.percentile(arr_lat, 95)), 4),
        "max_latency_s": round(float(np.max(arr_lat)), 4),
        "total_search_calls": int(sum(searches)),
        "mean_search_calls": round(sum(searches) / len(predictions), 3),
        "parse_failures": int(parse_failures),
        "call_errors": int(call_errors),
    }


# ---------------------------------------------------------------------------
# Per-collection breakdown
# ---------------------------------------------------------------------------

def per_collection_metrics(
    predictions: Sequence[Mapping[str, Any]],
    gt_map: Mapping[str, Mapping[int, str]],
) -> List[Dict[str, Any]]:
    """Per-collection precision/recall/F1, sorted by collection id."""
    by_collection: Dict[str, Dict[str, List[str]]] = {}
    for p in predictions:
        gt = _gt_for(p, gt_map)
        if gt is None:
            continue
        key = str(p.get("collection_id") or p.get("collection_number") or "?")
        bucket = by_collection.setdefault(key, {"y_true": [], "y_pred": []})
        bucket["y_true"].append(gt)
        bucket["y_pred"].append(str(p.get("predicted_label", "major_hallucination")))

    out: List[Dict[str, Any]] = []
    for key in sorted(by_collection):
        m = calculate_metrics(by_collection[key]["y_true"], by_collection[key]["y_pred"])
        m["collection_id"] = key
        out.append(m)
    return out


# ---------------------------------------------------------------------------
# Top-level evaluation
# ---------------------------------------------------------------------------

def evaluate_baseline_predictions(
    predictions: Sequence[Mapping[str, Any]],
    gt: Any,
    *,
    tag: str = "",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Compute metrics + cost summary for a list of baseline predictions.

    Parameters
    ----------
    predictions
        List of dicts as produced by
        :func:`citecheck.baselines.runner.run_baseline_config`.  Each
        item must carry ``predicted_label`` plus enough identifiers
        (``collection_id`` / ``collection_number`` /
        ``citation_number``) to look up its ground truth.
    gt
        Either a :class:`~citecheck.eval.dataset.Dataset` (in which
        case a ground-truth map is built on the fly), a path to
        ``corruption_metadata.json``, or a pre-built map of the form
        ``{collection_id: {citation_number: class_label}}`` (as
        produced by :func:`citecheck.eval.ground_truth.build_gt_map`).
    tag
        Optional run identifier shown in the printed report.
    verbose
        If ``True``, print metrics + cost summary to stdout.

    Returns
    -------
    dict
        Payload with ``tag``, ``global_metrics``, ``cost_summary``,
        ``confusion_matrix``, ``y_true``, ``y_pred``, and
        ``per_collection_metrics`` keys.  Each prediction dict is also
        annotated in-place with its resolved ``ground_truth_label``
        when one was found.
    """
    if isinstance(gt, Dataset):
        gt_map = build_gt_map(gt)
    elif isinstance(gt, (str, Path)):
        gt_map = build_gt_map(load_dataset(Path(gt)))
    elif isinstance(gt, Mapping):
        gt_map = gt  # already-built {collection_id: {citation_number: label}}
    else:
        raise TypeError(
            f"gt must be a Dataset, path, or mapping; got {type(gt).__name__}"
        )

    y_true: List[str] = []
    y_pred: List[str] = []

    annotated: List[Mapping[str, Any]] = []
    for p in predictions:
        resolved = _gt_for(p, gt_map)
        if resolved is None:
            annotated.append(p)
            continue
        y_true.append(resolved)
        y_pred.append(str(p.get("predicted_label", "major_hallucination")))
        if isinstance(p, dict):
            p["ground_truth_label"] = resolved
        annotated.append(p)

    global_metrics = calculate_metrics(y_true, y_pred) if y_true else {}
    cost_summary = summarize_costs(predictions)
    cm_dict = confusion_matrix_dict(y_true, y_pred) if y_true else {}
    cm_text = format_confusion_matrix(y_true, y_pred) if y_true else ""

    per_coll = per_collection_metrics(predictions, gt_map)

    if verbose:
        header = f"BASELINE METRICS ({tag})" if tag else "BASELINE METRICS"
        print(f"\n{'=' * 60}\n{header}\n{'=' * 60}")
        if global_metrics:
            print_metrics(global_metrics, f"\nGlobal {header}")
        if cm_text:
            print(cm_text)
        print("\nCost / Latency:")
        for k, v in cost_summary.items():
            print(f"  {k:>30s}: {v}")

    return {
        "tag": tag,
        "global_metrics": global_metrics,
        "cost_summary": cost_summary,
        "confusion_matrix": cm_dict,
        "y_true": y_true,
        "y_pred": y_pred,
        "per_collection_metrics": per_coll,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_evaluation(
    predictions: Sequence[Mapping[str, Any]],
    evaluation: Mapping[str, Any],
    save_dir: Path,
    *,
    tag: str = "",
) -> None:
    """Write evaluation artefacts under *save_dir*.

    Files written:

    - ``evaluation_results.json`` — the full :func:`evaluate_baseline_predictions` payload.
    - ``cost_summary.json``       — just the cost / latency block.
    - ``metrics_summary.txt``     — human-readable metrics + confusion matrix.
    - ``predictions.json``        — the per-citation predictions, with ground truth filled in.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    (save_dir / "evaluation_results.json").write_text(
        json.dumps(evaluation, indent=2, default=str), encoding="utf-8",
    )
    (save_dir / "cost_summary.json").write_text(
        json.dumps(evaluation.get("cost_summary", {}), indent=2),
        encoding="utf-8",
    )
    (save_dir / "predictions.json").write_text(
        json.dumps(list(predictions), indent=2, default=str),
        encoding="utf-8",
    )

    lines: List[str] = []
    if tag:
        lines.append(f"BASELINE: {tag}\n")
    gm = evaluation.get("global_metrics") or {}
    if gm:
        lines.append(format_metrics_summary(gm))
        lines.append("")
    cs = evaluation.get("cost_summary") or {}
    if cs:
        lines.append("COST / LATENCY")
        for k, v in cs.items():
            lines.append(f"  {k}: {v}")
    cm = evaluation.get("confusion_matrix") or {}
    if cm:
        y_true_repeats: List[str] = []
        y_pred_repeats: List[str] = []
        for gt_cls, row in cm.items():
            for pr_cls, n in row.items():
                y_true_repeats.extend([gt_cls] * int(n))
                y_pred_repeats.extend([pr_cls] * int(n))
        lines.append("")
        lines.append(format_confusion_matrix(y_true_repeats, y_pred_repeats))

    (save_dir / "metrics_summary.txt").write_text(
        "\n".join(lines), encoding="utf-8",
    )
