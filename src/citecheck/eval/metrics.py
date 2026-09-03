"""Per-class precision / recall / F1, accuracy, and confusion-matrix display.

Kept dependency-light on purpose: just the standard library, so eval
results can be loaded and reformatted without pulling in
scikit-learn.  The threshold-tuning code in
:mod:`citecheck.eval.threshold_search` does pull in
scikit-learn, but only that module.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence

from .ground_truth import CLASS_NAMES


__all__ = (
    "calculate_metrics",
    "format_metrics_summary",
    "print_metrics",
    "format_confusion_matrix",
    "print_confusion_matrix",
    "confusion_matrix_dict",
    "weighted_f1",
)


_CLASS_SHORT: Dict[str, str] = {
    "exact_match": "exact",
    "minor_hallucination": "minor",
    "major_hallucination": "major",
}


# ---------------------------------------------------------------------------
# Per-class metrics
# ---------------------------------------------------------------------------

def calculate_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
) -> Dict:
    """Per-class precision / recall / F1 plus overall accuracy.

    The returned dict has one entry per class in :data:`CLASS_NAMES`
    (each with ``precision``, ``recall``, ``f1``, ``support``,
    ``true_positives``, ``false_positives``, ``false_negatives``) plus
    top-level ``accuracy`` and ``total`` keys.
    """
    metrics: Dict = {}

    for cls in CLASS_NAMES:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )

        metrics[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(1 for t in y_true if t == cls),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
        }

    metrics["accuracy"] = (
        sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
        if y_true else 0.0
    )
    metrics["total"] = len(y_true)
    return metrics


def weighted_f1(metrics: Dict) -> float:
    """Support-weighted F1 averaged over the three classes."""
    total = max(int(metrics.get("total", 0)), 1)
    return sum(
        metrics.get(c, {}).get("f1", 0.0) * metrics.get(c, {}).get("support", 0)
        for c in CLASS_NAMES
    ) / total


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------

def _format_class(cls: str, m: Dict) -> List[str]:
    return [
        f"{cls.upper().replace('_', ' ')}:",
        f"  Precision: {m['precision']:.3f}  "
        f"(TP={m['true_positives']}, FP={m['false_positives']})",
        f"  Recall:    {m['recall']:.3f}  "
        f"(TP={m['true_positives']}, FN={m['false_negatives']})",
        f"  F1 Score:  {m['f1']:.3f}",
        f"  Support:   {m['support']} citations",
        "",
    ]


def format_metrics_summary(metrics: Dict) -> str:
    """Return the metrics block as a string ready to write to a summary file."""
    lines: List[str] = ["=" * 60, "EVALUATION METRICS SUMMARY", "=" * 60, ""]
    for cls in CLASS_NAMES:
        if cls in metrics:
            lines.extend(_format_class(cls, metrics[cls]))
    lines.append(
        f"Overall Accuracy: {metrics.get('accuracy', 0.0):.3f} "
        f"({metrics.get('total', 0)} citations)"
    )
    lines.append("=" * 60)
    return "\n".join(lines)


def print_metrics(metrics: Dict, title: str) -> None:
    """Print *metrics* under *title* using the same layout as :func:`format_metrics_summary`."""
    print(title)
    print("-" * 60)
    for cls in CLASS_NAMES:
        if cls in metrics:
            for line in _format_class(cls, metrics[cls]):
                print(line)
    print(
        f"Overall Accuracy: {metrics.get('accuracy', 0.0):.3f} "
        f"({metrics.get('total', 0)} citations)"
    )
    print()


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def confusion_matrix_dict(
    y_true: Sequence[str],
    y_pred: Sequence[str],
) -> Dict[str, Dict[str, int]]:
    """Return the 3x3 confusion matrix as a nested dict.

    Outer keys are the ground-truth classes, inner keys are the
    predicted classes — useful for JSON persistence and downstream
    aggregation. Unknown labels are silently skipped.
    """
    matrix: Dict[str, Dict[str, int]] = {
        gt: {p: 0 for p in CLASS_NAMES} for gt in CLASS_NAMES
    }
    for gt, pr in zip(y_true, y_pred):
        if gt in matrix and pr in matrix[gt]:
            matrix[gt][pr] += 1
    return matrix


def format_confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
) -> str:
    """Return a 3x3 confusion matrix (rows=true, cols=predicted) as a string."""
    matrix: Dict[str, Counter] = {gt: Counter() for gt in CLASS_NAMES}
    for gt, pr in zip(y_true, y_pred):
        if gt in matrix:
            matrix[gt][pr] += 1

    col_w = max(len(s) for s in _CLASS_SHORT.values()) + 2
    row_w = col_w

    header = " " * (row_w + 6) + "".join(
        f"{_CLASS_SHORT[c]:>{col_w}}" for c in CLASS_NAMES
    )
    sep = "-" * len(header)

    lines = ["Confusion Matrix (rows=true, cols=predicted):", sep, header, sep]
    for gt_cls in CLASS_NAMES:
        counts = "".join(
            f"{matrix[gt_cls][pr_cls]:>{col_w}}" for pr_cls in CLASS_NAMES
        )
        lines.append(f"  {_CLASS_SHORT[gt_cls]:>{row_w}}  |{counts}")
    lines.append(sep)
    return "\n".join(lines)


def print_confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
) -> None:
    """Print a 3x3 confusion matrix (rows=true, cols=predicted)."""
    print()
    print(format_confusion_matrix(y_true, y_pred))
