"""Evaluation layer: dataset loader, ground truth, metrics, runner.

The threshold-tuning and split / harness modules are layered on top of
the four primitives exposed here.
"""

from __future__ import annotations

from .dataset import CitationRecord, Collection, Dataset, load_dataset
from .ground_truth import (
    CLASS_NAMES,
    RAW_TO_CLASS,
    build_flat_gt_map,
    build_gt_map,
    resolve_gt_map,
    to_class_label,
)
from .metrics import (
    calculate_metrics,
    confusion_matrix_dict,
    format_confusion_matrix,
    format_metrics_summary,
    print_confusion_matrix,
    print_metrics,
    weighted_f1,
)
from .harness import build_fewshot_block, evaluate_dataset
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
    # dataset
    "CitationRecord",
    "Collection",
    "Dataset",
    "load_dataset",
    # ground truth
    "CLASS_NAMES",
    "RAW_TO_CLASS",
    "build_gt_map",
    "build_flat_gt_map",
    "resolve_gt_map",
    "to_class_label",
    # metrics
    "calculate_metrics",
    "weighted_f1",
    "format_metrics_summary",
    "print_metrics",
    "format_confusion_matrix",
    "print_confusion_matrix",
    "confusion_matrix_dict",
    # runner
    "run_detection",
    "evaluate_predictions",
    # splits
    "Exemplar",
    "Splits",
    "make_splits",
    "make_fewshot_exemplars",
    "save_splits",
    "load_splits",
    # threshold search
    "optimize_thresholds",
    # harness
    "build_fewshot_block",
    "evaluate_dataset",
)
