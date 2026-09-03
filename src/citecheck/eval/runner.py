"""Dataset-driven detection runner + evaluator.

Two responsibilities:

- :func:`run_detection` -- iterate over collections, parse each
  citation's ``corrupted_citation`` text, hand the resulting
  :class:`~citecheck.core.citation.Citation` objects to
  :func:`~citecheck.core.detector.detect_citations`, and
  attach ``collection_id`` / ``collection_number`` to every
  prediction.
- :func:`evaluate_predictions` -- re-classify cached predictions using
  a chosen ``(exact, minor)`` threshold pair, join with ground truth,
  and return per-class metrics, per-collection metrics, and per-class
  score distributions.

Note on caching
---------------
:func:`run_detection` returns predictions but doesn't write them.
Caching, splits, and threshold tuning live in
:mod:`citecheck.eval.harness` -- this module focuses on
"given a list of collections, produce predictions / metrics".
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .. import config
from ..core.citation import Citation, parse_citation
from ..core.classify import classify_score
from ..core.detector import detect_citations
from .dataset import Collection, Dataset
from .ground_truth import CLASS_NAMES, resolve_gt_map
from .metrics import (
    calculate_metrics,
    print_confusion_matrix,
    print_metrics,
)


__all__ = (
    "run_detection",
    "evaluate_predictions",
)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _print_citation_detail(
    pred: Dict,
    gt_label: Optional[str],
    *,
    review_enabled: bool,
    review_sim_threshold: float,
) -> None:
    """Print verbose info for a single citation prediction."""
    cnum = pred["citation_number"]
    cite_text = pred.get("citation_text") or ""
    if len(cite_text) > 120:
        cite_text = cite_text[:117] + "..."

    matched_title = pred.get("matched_title") or "(no match)"
    if len(matched_title) > 100:
        matched_title = matched_title[:97] + "..."
    matched_authors = pred.get("matched_authors") or ""
    matched_year = pred.get("matched_year") or ""
    api = pred.get("match_source_api", "?")
    sim = pred.get("match_title_similarity", 0.0)

    label = pred["predicted_label"]
    score = pred["llm_score"]
    reasoning = pred.get("llm_reasoning", "") or ""

    gt_str = f"  GT: {gt_label}" if gt_label else ""
    correct = "" if not gt_label else (" OK" if label == gt_label else " MISS")

    print(f"\n  [{cnum}] {cite_text}")
    print(f"      Match ({api}, sim={sim:.1f}): {matched_title}")
    if matched_authors or matched_year:
        print(f"      Match meta: {matched_authors}, {matched_year}")
    print(f"      LLM: {label} (score={score:.1f}){gt_str}{correct}")
    print(f"      Reasoning: {reasoning}")

    if pred.get("reviewer_triggered"):
        orig_label = pred.get("original_llm_label", "?")
        orig_score = pred.get("original_llm_score", 0.0) or 0.0
        rev_label = pred.get("reviewer_label", "?")
        rev_reasoning = pred.get("reviewer_reasoning", "") or ""
        changed = "CHANGED" if rev_label != orig_label else "AGREED"
        print(
            f"      >>> REVIEWER [{changed}]: "
            f"{orig_label} (score={orig_score:.1f}) -> {rev_label}"
        )
        print(f"      >>> Reviewer reasoning: {rev_reasoning}")
    elif review_enabled:
        if api not in ("arXiv", "WebSearch"):
            print(f"      >>> Reviewer skipped: match source is {api} (not arXiv/WebSearch)")
        elif sim >= review_sim_threshold:
            print(
                f"      >>> Reviewer skipped: title similarity "
                f"{sim:.1f}% >= threshold {review_sim_threshold:.1f}%"
            )
        else:
            print("      >>> Reviewer skipped: no match found")


def _collection_citations(collection: Collection) -> List[Citation]:
    """Parse every ``corrupted_citation`` in *collection* into a :class:`Citation`."""
    return [
        parse_citation(c.citation_number, c.corrupted_citation)
        for c in collection.citations
    ]


def run_detection(
    collections: Iterable[Collection],
    *,
    verbose: bool = True,
    detailed: bool = False,
    gt_map: Optional[Dict[str, Dict[int, str]]] = None,
    review_sim_threshold: float = config.REVIEW_SIM_THRESHOLD,
    exact_threshold: float = config.EXACT_THRESHOLD,
    minor_threshold: float = config.MINOR_THRESHOLD,
    **detector_kwargs,
) -> List[Dict]:
    """Run the detector on every citation in *collections*.

    Each prediction is annotated with ``collection_id`` and
    ``collection_number`` so downstream evaluation can join with
    ground truth.

    Parameters
    ----------
    collections
        Iterable of :class:`Collection` (e.g.
        ``dataset.collections`` or one split's collections).
    verbose
        Print per-collection progress and the
        :func:`~citecheck.core.detector.detect_citations`
        progress bar.
    detailed
        After each collection, print per-citation details (citation
        text, matched source, LLM verdict, reviewer status, ground
        truth comparison).  Requires *gt_map* for the GT comparison
        column to appear.
    gt_map
        ``{collection_id: {citation_number: class_label}}`` mapping
        (see :func:`citecheck.eval.ground_truth.build_gt_map`).
        Only used when *detailed*.
    review_sim_threshold, exact_threshold, minor_threshold
        Forwarded to the detector and used when re-classifying scores
        for the per-collection accuracy printout.
    detector_kwargs
        Forwarded to
        :func:`citecheck.core.detector.detect_citations`.
    """
    # The detector itself uses thresholds + reviewer sim threshold,
    # forward them through.
    detector_kwargs.setdefault("exact_threshold", exact_threshold)
    detector_kwargs.setdefault("minor_threshold", minor_threshold)
    detector_kwargs.setdefault("review_sim_threshold", review_sim_threshold)

    review_enabled = detector_kwargs.get("review_enabled", config.REVIEW_ENABLED)

    all_predictions: List[Dict] = []
    collections = list(collections)

    for collection in collections:
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"COLLECTION {collection.collection_number} -- {collection.collection_id}")
            print(f"{'=' * 60}")

        citations = _collection_citations(collection)

        preds = detect_citations(
            citations,
            verbose=verbose,
            **detector_kwargs,
        )

        for pred in preds:
            pred["collection_id"] = collection.collection_id
            pred["collection_number"] = collection.collection_number

        # ── Per-citation detail printout ────────────────────────────
        if detailed:
            collection_gt = gt_map.get(collection.collection_id, {}) if gt_map else {}
            correct, total_with_gt = 0, 0

            for pred in preds:
                cnum = pred["citation_number"]
                gt_label = collection_gt.get(cnum)
                _print_citation_detail(
                    pred, gt_label,
                    review_enabled=review_enabled,
                    review_sim_threshold=review_sim_threshold,
                )
                if gt_label is not None:
                    reclassified = classify_score(
                        pred["llm_score"], exact_threshold, minor_threshold,
                    )
                    total_with_gt += 1
                    if reclassified == gt_label:
                        correct += 1

            if total_with_gt > 0:
                acc = correct / total_with_gt
                print(
                    f"\n  >> Collection {collection.collection_number} accuracy: "
                    f"{correct}/{total_with_gt} = {acc:.1%}"
                )
            print()

        all_predictions.extend(preds)

    return all_predictions


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_predictions(
    predictions: Sequence[Dict],
    *,
    exact_threshold: float,
    minor_threshold: float,
    gt_map: Optional[Dict[str, Dict[int, str]]] = None,
    dataset: Optional[Dataset] = None,
    verbose: bool = True,
    tag: str = "",
) -> Dict:
    """Re-classify *predictions* under the given thresholds and score them.

    Either *gt_map* or *dataset* must be provided -- the latter just
    builds a gt_map via
    :func:`~citecheck.eval.ground_truth.build_gt_map`.

    Returns a dict with:

    - ``global_metrics`` -- :func:`calculate_metrics` on the full
      flattened list.
    - ``per_collection_metrics`` -- one entry per collection, each
      with its own metrics + ``collection_id`` field.
    - ``score_distributions`` -- ``{class_label: [llm_scores]}``,
      grouped by the **ground-truth** label.
    - ``thresholds`` -- the ``(exact, minor)`` pair used.
    - ``y_true`` / ``y_pred`` -- flat label lists for downstream
      use (e.g. saving a separate confusion matrix).
    """
    truth = resolve_gt_map(gt_map, dataset)

    y_true: List[str] = []
    y_pred: List[str] = []
    per_collection: Dict[str, Dict[str, list]] = {}
    score_by_gt: Dict[str, List[float]] = {c: [] for c in CLASS_NAMES}

    for pred in predictions:
        cid = pred.get("collection_id", "")
        cnum = pred["citation_number"]
        gt_label = truth.get(cid, {}).get(cnum)
        if gt_label is None:
            continue

        reclassified = classify_score(
            pred["llm_score"], exact_threshold, minor_threshold,
        )

        y_true.append(gt_label)
        y_pred.append(reclassified)
        pred["ground_truth_label"] = gt_label
        pred["reclassified_label"] = reclassified

        score_by_gt[gt_label].append(pred["llm_score"])

        per_collection.setdefault(cid, {"y_true": [], "y_pred": []})
        per_collection[cid]["y_true"].append(gt_label)
        per_collection[cid]["y_pred"].append(reclassified)

    global_metrics = calculate_metrics(y_true, y_pred) if y_true else {}

    per_collection_metrics: List[Dict] = []
    for cid in sorted(per_collection):
        m = calculate_metrics(
            per_collection[cid]["y_true"], per_collection[cid]["y_pred"],
        )
        m["collection_id"] = cid
        per_collection_metrics.append(m)

    if verbose:
        _print_eval_block(
            tag=tag,
            exact_threshold=exact_threshold,
            minor_threshold=minor_threshold,
            score_by_gt=score_by_gt,
            global_metrics=global_metrics,
            y_true=y_true,
            y_pred=y_pred,
            per_collection_metrics=per_collection_metrics,
        )

    return {
        "global_metrics": global_metrics,
        "per_collection_metrics": per_collection_metrics,
        "score_distributions": dict(score_by_gt),
        "thresholds": {"exact": exact_threshold, "minor": minor_threshold},
        "y_true": y_true,
        "y_pred": y_pred,
    }


# ---------------------------------------------------------------------------
# Verbose output for evaluate_predictions
# ---------------------------------------------------------------------------

def _print_eval_block(
    *,
    tag: str,
    exact_threshold: float,
    minor_threshold: float,
    score_by_gt: Dict[str, List[float]],
    global_metrics: Dict,
    y_true: List[str],
    y_pred: List[str],
    per_collection_metrics: List[Dict],
) -> None:
    header = f"METRICS ({tag})" if tag else "METRICS"
    print(f"\n{'=' * 60}")
    print(f"{header} -- exact>={exact_threshold}, minor>={minor_threshold}")
    print(f"{'=' * 60}")

    print("\nLLM score distribution by ground-truth label:")
    print("-" * 60)
    for cls in CLASS_NAMES:
        scores = score_by_gt.get(cls, [])
        if not scores:
            continue
        n = len(scores)
        mean = sum(scores) / n
        sorted_s = sorted(scores)
        median = (
            sorted_s[n // 2]
            if n % 2 == 1
            else (sorted_s[n // 2 - 1] + sorted_s[n // 2]) / 2
        )
        # Population std-dev to match numpy's default in the legacy code.
        var = sum((s - mean) ** 2 for s in scores) / n
        std = var ** 0.5
        print(
            f"  {cls:>25s}: n={n:3d}  mean={mean:.2f}  std={std:.2f}  "
            f"min={sorted_s[0]:.1f}  median={median:.1f}  max={sorted_s[-1]:.1f}"
        )

    print_metrics(global_metrics, f"\nGlobal {header}")

    if y_true:
        print_confusion_matrix(y_true, y_pred)

    if len(per_collection_metrics) > 1:
        print(
            f"\n  {'Collection':<25} {'Acc':>6} "
            f"{'ExF1':>6} {'MiF1':>6} {'MaF1':>6}"
        )
        print("  " + "-" * 51)
        for m in per_collection_metrics:
            ef1 = m.get("exact_match", {}).get("f1", 0)
            mif1 = m.get("minor_hallucination", {}).get("f1", 0)
            maf1 = m.get("major_hallucination", {}).get("f1", 0)
            print(
                f"  {m['collection_id']:<25} {m['accuracy']:>6.3f} "
                f"{ef1:>6.3f} {mif1:>6.3f} {maf1:>6.3f}"
            )
