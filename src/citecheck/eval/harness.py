"""Dev/test evaluation harness with threshold tuning + optional few-shot.

This is the top-level orchestrator that the ``citecheck-evaluate`` CLI calls
into.  Given a :class:`~citecheck.eval.dataset.Dataset`
and a save directory, it:

1. Builds (or loads) a deterministic dev/test split with
   :func:`citecheck.eval.splits.make_splits`.
2. Optionally builds a few-shot exemplar block to append to the
   verifier's system prompt.
3. Runs detection on the dev split (with caching to JSON).
4. Runs detection on the test split (with caching to JSON).
5. Tunes ``(minor, exact)`` thresholds via grid-search on dev (minus
   the exemplars in few-shot mode -- never grade citations that have
   already been shown to the model).
6. Evaluates dev and test at both default (paper) and tuned
   thresholds.
7. Writes everything under *save_dir*::

       splits.json
       dev_predictions[_fewshot].json
       test_predictions[_fewshot].json
       threshold_analysis.json
       dev_evaluation.json
       test_evaluation.json
       metrics_summary.txt
       fewshot_block.txt   (few-shot mode only)

The headline numbers are the **TEST / tuned** block in
``metrics_summary.txt``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .. import config
from .dataset import Dataset, load_dataset
from .ground_truth import build_gt_map
from .metrics import format_metrics_summary
from .runner import evaluate_predictions, run_detection
from .splits import (
    Exemplar,
    Splits,
    load_splits,
    make_fewshot_exemplars,
    make_splits,
    save_splits,
)
from .threshold_search import optimize_thresholds


__all__ = (
    "build_fewshot_block",
    "evaluate_dataset",
)


# ---------------------------------------------------------------------------
# Few-shot block (appended to the verifier's system prompt)
# ---------------------------------------------------------------------------

_FEWSHOT_HEADER = (
    "REFERENCE EXAMPLES (illustrate the rubric using citation text alone).\n"
    "These show, for each class, what a typical citation looks like and the\n"
    "intended label/score. The matched-source block in the user message may\n"
    "or may not be present at inference time; weight it together with these\n"
    "rubric examples when making your final decision.\n"
)


def _score_for_exemplar_label(label: str) -> int:
    """Pick a representative integer score for the few-shot examples."""
    return {
        "exact_match": 10,
        "minor_hallucination": 6,
        "major_hallucination": 1,
    }.get(label, 1)


def _render_exemplar(idx: int, ex: Exemplar) -> str:
    text = " ".join((ex.citation_text or "").split())
    reasoning = " ".join((ex.reasoning or "").split())
    return (
        f"Example {idx}:\n"
        f"  Citation: {text}\n"
        f"  Label: {ex.label}\n"
        f"  Score: {_score_for_exemplar_label(ex.label)}/10\n"
        f"  Reasoning: {reasoning}\n"
    )


def build_fewshot_block(exemplars: Sequence[Exemplar]) -> str:
    """Render *exemplars* into a single block ready to append to the
    verifier's system prompt."""
    if not exemplars:
        return ""
    blocks = [_render_exemplar(i + 1, ex) for i, ex in enumerate(exemplars)]
    return _FEWSHOT_HEADER + "\n" + "\n".join(blocks)


# ---------------------------------------------------------------------------
# JSON safety + I/O helpers
# ---------------------------------------------------------------------------

def _json_safe(obj: Any) -> Any:
    """Coerce numpy scalars / arrays so ``json.dump`` is happy."""
    try:
        import numpy as np
    except ImportError:
        np = None  # numpy unavailable -- fall through to str()
    if np is not None:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    return str(obj)


def _strip_predictions(predictions: Sequence[Dict]) -> List[Dict]:
    """Drop ``matched_source`` (large nested API payload) per prediction."""
    out: List[Dict] = []
    for p in predictions:
        sp = {k: v for k, v in p.items() if k != "matched_source"}
        sp["matched_source_summary"] = (
            p.get("matched_source") if isinstance(p.get("matched_source"), dict) else None
        )
        out.append(sp)
    return out


def _save_predictions(predictions: Sequence[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "total": len(predictions),
                "predictions": _strip_predictions(predictions),
            },
            indent=2, default=_json_safe,
        ),
        encoding="utf-8",
    )


def _load_predictions(path: Path) -> List[Dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data.get("predictions", [])
    return list(data)


# ---------------------------------------------------------------------------
# Splits resolution
# ---------------------------------------------------------------------------

def _resolve_splits(
    dataset: Dataset,
    *,
    seed: int,
    splits_path: Optional[Path],
    save_dir: Path,
    need_exemplars: bool,
) -> tuple[Splits, List[Exemplar]]:
    """Load a pre-computed splits file or compute fresh, then mirror it
    into ``save_dir/splits.json`` so each run is self-contained."""
    if splits_path is not None and splits_path.exists():
        print(f"Loading splits from {splits_path}")
        splits, exemplars = load_splits(splits_path)
        if need_exemplars and not exemplars:
            print(
                f"[fewshot] {splits_path.name} had no exemplars -- recomputing "
                f"deterministically (seed={splits.seed})"
            )
            exemplars = make_fewshot_exemplars(
                dataset, splits, seed=splits.seed,
            )
    else:
        print(f"Computing splits (seed={seed})")
        splits = make_splits(dataset, seed=seed)
        exemplars = (
            make_fewshot_exemplars(dataset, splits, seed=seed)
            if need_exemplars else []
        )

    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "splits.json"
    if (
        splits_path is not None
        and splits_path.resolve() != out_path.resolve()
    ):
        shutil.copy2(splits_path, out_path)
    else:
        save_splits(splits, exemplars, out_path)
    return splits, exemplars


# ---------------------------------------------------------------------------
# Detection (with caching)
# ---------------------------------------------------------------------------

def _run_or_load_detection(
    dataset: Dataset,
    collection_numbers: Sequence[int],
    cache_path: Path,
    *,
    label: str,
    verbose: bool,
    detailed: bool,
    gt_map: Optional[Dict[str, Dict[int, str]]],
    detector_kwargs: Dict[str, Any],
    use_cache: bool,
) -> List[Dict]:
    if use_cache and cache_path.exists():
        if verbose:
            print(f"\n[{label}] Loading cached predictions from {cache_path}")
        return _load_predictions(cache_path)

    collections = [
        c for c in dataset.collections
        if c.collection_number in set(collection_numbers)
    ]
    if verbose:
        ids = [c.collection_id for c in collections]
        print(
            f"\n[{label}] Running detector on {len(collections)} collection(s):\n"
            f"  {ids}"
        )

    preds = run_detection(
        collections,
        verbose=verbose,
        detailed=detailed,
        gt_map=gt_map if detailed else None,
        **detector_kwargs,
    )

    _save_predictions(preds, cache_path)
    if verbose:
        print(f"[{label}] Saved {len(preds)} predictions -> {cache_path}")
    return preds


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------

def _write_summary(
    save_dir: Path,
    *,
    splits: Splits,
    default_thresholds: Dict[str, float],
    tuned_thresholds: Dict[str, float],
    dev_default: Dict,
    dev_tuned: Dict,
    test_default: Dict,
    test_tuned: Dict,
    optim_results: Dict,
    fewshot: bool,
    n_exemplars: int,
    n_dev_dropped: int,
) -> Path:
    path = save_dir / "metrics_summary.txt"
    lines: List[str] = []

    lines.append(
        "CITECHECK EVALUATION (dev/test split + threshold tuning)"
    )
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Mode             : {'few-shot' if fewshot else 'zero-shot'}")
    if fewshot:
        lines.append(
            f"Exemplars used   : {n_exemplars} "
            f"(removed from dev tuning set: {n_dev_dropped})"
        )
    lines.append(f"Split seed       : {splits.seed}")
    lines.append(
        f"Dev collections  : {len(splits.dev_collections)} "
        f"-> {list(splits.dev_collections)}"
    )
    lines.append(
        f"Test collections : {len(splits.test_collections)} "
        f"-> {list(splits.test_collections)}"
    )
    lines.append("")
    lines.append("DEFAULT THRESHOLDS")
    lines.append(
        f"  exact >= {default_thresholds['exact']}, "
        f"minor >= {default_thresholds['minor']}"
    )
    lines.append("")
    lines.append("TUNED THRESHOLDS (grid search on DEV)")
    lines.append(
        f"  exact >= {tuned_thresholds['exact']}, "
        f"minor >= {tuned_thresholds['minor']}"
    )
    gs = optim_results.get("grid_search", {})
    lines.append(
        f"  weighted F1 on dev (tuned): {gs.get('weighted_f1', 'n/a')}"
    )
    lines.append("")

    def _section(title: str, ev: Dict) -> None:
        lines.append("-" * 70)
        lines.append(title)
        lines.append("-" * 70)
        lines.append(format_metrics_summary(ev["global_metrics"]))
        lines.append("")

    _section("DEV  /  default thresholds", dev_default)
    _section("DEV  /  tuned thresholds  ", dev_tuned)
    _section("TEST /  default thresholds", test_default)
    _section(
        "TEST /  tuned thresholds  (FINAL -- quote this in the paper)",
        test_tuned,
    )

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def evaluate_dataset(
    dataset: Optional[Dataset] = None,
    *,
    save_dir: Path,
    dataset_path: Optional[Path] = None,
    seed: int = config.DEFAULT_SEED,
    splits_path: Optional[Path] = None,
    default_exact_threshold: float = config.EXACT_THRESHOLD,
    default_minor_threshold: float = config.MINOR_THRESHOLD,
    dev_predictions_path: Optional[Path] = None,
    test_predictions_path: Optional[Path] = None,
    use_cache: bool = True,
    verbose: bool = True,
    detailed: bool = False,
    fewshot: bool = False,
    # Detector kwargs forwarded to detect_citations
    try_arxiv: bool = config.TRY_ARXIV,
    try_web_search: bool = config.TRY_WEB_SEARCH,
    web_search_model: str = config.WEB_SEARCH_MODEL,
    accept_best_web_search: bool = config.ACCEPT_BEST_WEB_SEARCH,
    api_timeout: float = config.API_TIMEOUT,
    inter_api_delay: float = config.INTER_API_DELAY,
    llm_provider: str = config.LLM_PROVIDER,
    llm_model: Optional[str] = None,
    llm_temperature: Optional[float] = None,
    review_enabled: bool = config.REVIEW_ENABLED,
    review_model: Optional[str] = config.REVIEW_MODEL,
    review_sim_threshold: float = config.REVIEW_SIM_THRESHOLD,
    llm_parse_enabled: bool = config.LLM_PARSE_ENABLED,
    llm_parse_model: Optional[str] = config.LLM_PARSE_MODEL,
    monitor: bool = False,
) -> Dict[str, Any]:
    """Run the full dev/test evaluation pipeline.

    Parameters
    ----------
    dataset
        Pre-loaded :class:`Dataset`.  If ``None``, *dataset_path* is
        used (or :data:`config.DEFAULT_DATASET_PATH` if both are
        ``None``).
    save_dir
        Where every artifact is written.  Created if missing.
    dataset_path
        Optional explicit path to ``corruption_metadata.json``.  Only
        consulted when *dataset* is ``None``.
    seed, splits_path
        Forwarded to :func:`_resolve_splits`.
    default_exact_threshold, default_minor_threshold
        Thresholds for the "before tuning" comparison.  Defaults are
        the paper's headline values.
    dev_predictions_path, test_predictions_path
        Optional paths to cached prediction JSON files (skip
        detection, no API cost).  Required when re-tuning thresholds
        offline.
    use_cache
        If ``True`` and a default cache file already exists in
        *save_dir*, reuse it instead of re-running detection.
    fewshot
        When True, the exemplars from ``splits.json`` are appended to
        the verifier's system prompt and removed from the dev set
        before threshold tuning.

    Returns
    -------
    dict
        Mirror of the persisted artifacts (splits, predictions,
        thresholds, evaluation payloads, threshold analysis).
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if dataset is None:
        dataset = load_dataset(dataset_path)

    # ── Splits + exemplars ──────────────────────────────────────────────
    splits, exemplars = _resolve_splits(
        dataset, seed=seed, splits_path=splits_path, save_dir=save_dir,
        need_exemplars=fewshot,
    )
    dev_nums = list(splits.dev_collections)
    test_nums = list(splits.test_collections)

    if verbose:
        print(f"\nDev  collections ({len(dev_nums)}): {dev_nums}")
        print(f"Test collections ({len(test_nums)}): {test_nums}")
        print(f"Mode: {'few-shot' if fewshot else 'zero-shot'}")
        if fewshot:
            print(
                f"Exemplars: {len(exemplars)} "
                "(will be appended to the verifier prompt)"
            )

    # ── Few-shot block ──────────────────────────────────────────────────
    fewshot_block: Optional[str] = None
    exemplar_keys: set[tuple[str, int]] = set()
    if fewshot:
        if not exemplars:
            raise RuntimeError(
                "fewshot=True but no exemplars are available. Either pass a "
                "splits_path containing exemplars, or rebuild splits in-place "
                "(remove --splits_path)."
            )
        fewshot_block = build_fewshot_block(exemplars)
        exemplar_keys = {
            (e.collection_id, int(e.citation_number)) for e in exemplars
        }
        (save_dir / "fewshot_block.txt").write_text(fewshot_block, encoding="utf-8")
        if verbose:
            print(
                f"Wrote few-shot block -> {save_dir / 'fewshot_block.txt'} "
                f"({len(fewshot_block):,} chars, {len(exemplars)} examples)"
            )

    # ── Ground truth ────────────────────────────────────────────────────
    gt_map = build_gt_map(dataset)

    # ── Detector kwargs (shared across dev + test) ──────────────────────
    detector_kwargs: Dict[str, Any] = dict(
        exact_threshold=default_exact_threshold,
        minor_threshold=default_minor_threshold,
        try_arxiv=try_arxiv,
        try_web_search=try_web_search,
        web_search_model=web_search_model,
        accept_best_web_search=accept_best_web_search,
        api_timeout=api_timeout,
        inter_api_delay=inter_api_delay,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_temperature=llm_temperature,
        review_enabled=review_enabled,
        review_model=review_model,
        review_sim_threshold=review_sim_threshold,
        llm_parse_enabled=llm_parse_enabled,
        llm_parse_model=llm_parse_model,
        fewshot_block=fewshot_block,
        monitor=monitor,
    )

    mode_suffix = "_fewshot" if fewshot else ""

    # ── Detection (dev) ─────────────────────────────────────────────────
    dev_cache = (
        Path(dev_predictions_path)
        if dev_predictions_path is not None
        else save_dir / f"dev_predictions{mode_suffix}.json"
    )
    dev_preds = _run_or_load_detection(
        dataset, dev_nums, dev_cache,
        label="DEV", verbose=verbose, detailed=detailed,
        gt_map=gt_map, detector_kwargs=detector_kwargs,
        use_cache=use_cache,
    )

    # ── Detection (test) ────────────────────────────────────────────────
    test_cache = (
        Path(test_predictions_path)
        if test_predictions_path is not None
        else save_dir / f"test_predictions{mode_suffix}.json"
    )
    test_preds = _run_or_load_detection(
        dataset, test_nums, test_cache,
        label="TEST", verbose=verbose, detailed=detailed,
        gt_map=gt_map, detector_kwargs=detector_kwargs,
        use_cache=use_cache,
    )

    # ── Threshold tuning on DEV only ────────────────────────────────────
    dev_preds_for_tuning = dev_preds
    n_dropped = 0
    if fewshot and exemplar_keys:
        dev_preds_for_tuning = [
            p for p in dev_preds
            if (p.get("collection_id", ""), int(p.get("citation_number", -1)))
            not in exemplar_keys
        ]
        n_dropped = len(dev_preds) - len(dev_preds_for_tuning)
        if verbose:
            print(
                f"\n[fewshot] Dropped {n_dropped} exemplar citation(s) from "
                f"dev tuning set ({len(dev_preds)} -> {len(dev_preds_for_tuning)})"
            )

    if verbose:
        print("\n" + "=" * 70)
        print(
            "THRESHOLD TUNING (grid search on DEV"
            f"{' minus exemplars' if fewshot else ''})"
        )
        print("=" * 70)

    optim_results = optimize_thresholds(
        dev_preds_for_tuning, gt_map=gt_map, verbose=verbose,
    )
    tuned_exact = float(optim_results["recommended"]["exact_threshold"])
    tuned_minor = float(optim_results["recommended"]["minor_threshold"])

    # ── Evaluations ─────────────────────────────────────────────────────
    dev_default = evaluate_predictions(
        dev_preds_for_tuning,
        exact_threshold=default_exact_threshold,
        minor_threshold=default_minor_threshold,
        gt_map=gt_map, verbose=verbose, tag="DEV / default",
    )
    dev_tuned = evaluate_predictions(
        dev_preds_for_tuning,
        exact_threshold=tuned_exact, minor_threshold=tuned_minor,
        gt_map=gt_map, verbose=verbose, tag="DEV / tuned",
    )
    test_default = evaluate_predictions(
        test_preds,
        exact_threshold=default_exact_threshold,
        minor_threshold=default_minor_threshold,
        gt_map=gt_map, verbose=verbose, tag="TEST / default",
    )
    test_tuned = evaluate_predictions(
        test_preds,
        exact_threshold=tuned_exact, minor_threshold=tuned_minor,
        gt_map=gt_map, verbose=verbose, tag="TEST / tuned (FINAL)",
    )

    # ── Persist ─────────────────────────────────────────────────────────
    (save_dir / "threshold_analysis.json").write_text(
        json.dumps(optim_results, indent=2, default=_json_safe),
        encoding="utf-8",
    )

    threshold_payload = {
        "default": {
            "exact": default_exact_threshold,
            "minor": default_minor_threshold,
        },
        "tuned": {"exact": tuned_exact, "minor": tuned_minor},
    }

    (save_dir / "dev_evaluation.json").write_text(
        json.dumps({
            "default": dev_default, "tuned": dev_tuned,
            "thresholds": threshold_payload,
        }, indent=2, default=_json_safe),
        encoding="utf-8",
    )

    (save_dir / "test_evaluation.json").write_text(
        json.dumps({
            "default": test_default, "tuned": test_tuned,
            "thresholds": threshold_payload,
        }, indent=2, default=_json_safe),
        encoding="utf-8",
    )

    summary_path = _write_summary(
        save_dir,
        splits=splits,
        default_thresholds={
            "exact": default_exact_threshold,
            "minor": default_minor_threshold,
        },
        tuned_thresholds={"exact": tuned_exact, "minor": tuned_minor},
        dev_default=dev_default, dev_tuned=dev_tuned,
        test_default=test_default, test_tuned=test_tuned,
        optim_results=optim_results,
        fewshot=fewshot, n_exemplars=len(exemplars), n_dev_dropped=n_dropped,
    )

    if verbose:
        print(
            f"\n{'=' * 70}\nFINAL TEST METRICS "
            f"(tuned thresholds: exact>={tuned_exact}, minor>={tuned_minor})\n"
            f"{'=' * 70}"
        )
        gm = test_tuned["global_metrics"]
        print(f"  Accuracy:                   {gm.get('accuracy', 0):.4f}")
        for cls in ("exact_match", "minor_hallucination", "major_hallucination"):
            cm = gm.get(cls, {})
            print(
                f"  {cls:>22s} F1:  {cm.get('f1', 0):.4f}  "
                f"(P={cm.get('precision', 0):.3f}  R={cm.get('recall', 0):.3f})"
            )
        print(f"\nWrote summary -> {summary_path}")

    return {
        "mode": "fewshot" if fewshot else "zeroshot",
        "n_exemplars": len(exemplars),
        "n_dev_dropped": n_dropped,
        "splits": splits,
        "dev_predictions": dev_preds,
        "test_predictions": test_preds,
        "thresholds": threshold_payload,
        "dev_default": dev_default,
        "dev_tuned": dev_tuned,
        "test_default": test_default,
        "test_tuned": test_tuned,
        "threshold_analysis": optim_results,
    }
