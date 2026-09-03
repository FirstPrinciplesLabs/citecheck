"""Threshold tuning for the verifier's 0-10 score.

Three complementary methods:

1. **Grid search** -- exhaustively evaluates every
   ``(minor, exact)`` pair on a 0-to-10 grid (default step 0.25) and
   returns the pair that maximises support-weighted F1.  This is the
   primary recommendation: it gives a pair of *interpretable*
   thresholds you can paste into a config file.
2. **Decision tree (depth=2)** on ``llm_score`` alone -- learns
   the two split points sklearn would put on a single feature.  Acts
   as a sanity check that the grid-search optimum sits where the
   tree wants to split.
3. **Random forest** on ``[llm_score, match_title_similarity,
   match_confidence]`` -- shows the ceiling accuracy when you let a
   model use the cascade's confidence scores too, and reports
   feature importances.  Reported with cross-validated weighted F1
   (folds = ``min(5, n_classes_present)``).

The recommended thresholds always come from method 1 -- the others
are reported alongside as diagnostics and never used to set defaults.

This is the *only* module in the package that depends on
``scikit-learn`` and ``numpy``; both are listed under the ``dev``
extra in ``pyproject.toml`` so casual users can run the detector
without installing them.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from ..core.classify import classify_score
from .dataset import Dataset
from .ground_truth import resolve_gt_map
from .metrics import (
    calculate_metrics,
    print_confusion_matrix,
    print_metrics,
    weighted_f1,
)


__all__ = ("optimize_thresholds",)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _extract_supervised(
    predictions: Sequence[Dict],
    gt_map: Dict[str, Dict[int, str]],
) -> Tuple[List[float], List[List[float]], List[str]]:
    """Pull ``(llm_score, [features], gt_label)`` triples for every
    prediction that has ground truth attached."""
    scores: List[float] = []
    features_multi: List[List[float]] = []
    labels: List[str] = []

    for pred in predictions:
        cid = pred.get("collection_id", "")
        cnum = pred["citation_number"]
        gt_label = gt_map.get(cid, {}).get(cnum)
        if gt_label is None:
            continue

        scores.append(pred["llm_score"])
        labels.append(gt_label)
        features_multi.append([
            pred["llm_score"],
            pred.get("match_title_similarity", 0.0) or 0.0,
            pred.get("match_confidence", 0.0) or 0.0,
        ])

    return scores, features_multi, labels


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def _grid_search(
    scores: Sequence[float],
    labels: Sequence[str],
    *,
    grid_min: float,
    grid_max: float,
    grid_step: float,
) -> Tuple[Tuple[float, float], float, Dict]:
    """Exhaustive search over ``(minor, exact)`` thresholds.

    Returns ``((best_minor, best_exact), best_weighted_f1, best_metrics)``.
    The grid uses an inclusive range ``[grid_min, grid_max]`` with a
    ``grid_step`` increment.
    """
    best_wf1 = -1.0
    best_pair: Tuple[float, float] = (1.5, 5.0)
    best_metrics: Dict = {}

    n_steps = int(round((grid_max - grid_min) / grid_step)) + 1
    candidates = [grid_min + i * grid_step for i in range(n_steps)]

    for minor_t in candidates:
        for exact_t in candidates:
            if exact_t < minor_t:
                continue
            preds = [classify_score(s, exact_t, minor_t) for s in scores]
            m = calculate_metrics(labels, preds)
            wf1 = weighted_f1(m)
            if wf1 > best_wf1:
                best_wf1 = wf1
                best_pair = (minor_t, exact_t)
                best_metrics = m

    return best_pair, best_wf1, best_metrics


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def optimize_thresholds(
    predictions: Sequence[Dict],
    *,
    gt_map: Optional[Dict[str, Dict[int, str]]] = None,
    dataset: Optional[Dataset] = None,
    grid_min: float = 0.0,
    grid_max: float = 10.0,
    grid_step: float = 0.25,
    rf_n_estimators: int = 200,
    rf_max_depth: int = 5,
    rf_random_state: int = 42,
    verbose: bool = True,
) -> Dict:
    """Tune ``(minor, exact)`` thresholds on the supplied *predictions*.

    Either *gt_map* or *dataset* must be provided.  See module
    docstring for what each method computes.

    Parameters
    ----------
    predictions
        Predictions from
        :func:`citecheck.eval.runner.run_detection`.  Only
        records whose ``(collection_id, citation_number)`` is present
        in the ground truth contribute.
    grid_min, grid_max, grid_step
        Inclusive 1-D grid over which both thresholds are searched.
    rf_n_estimators, rf_max_depth, rf_random_state
        Random-forest hyperparameters.

    Returns
    -------
    dict
        ::

            {
              "grid_search":   { "minor_threshold", "exact_threshold",
                                 "weighted_f1", "metrics" },
              "decision_tree": { "thresholds", "metrics" },
              "random_forest": { "cv_weighted_f1_mean",
                                 "cv_weighted_f1_std",
                                 "feature_importances", "metrics" },
              "recommended":   { "minor_threshold", "exact_threshold" },
            }

        ``recommended`` always equals the grid-search result.
    """
    truth = resolve_gt_map(gt_map, dataset)

    # Local imports keep numpy / sklearn out of the import path of
    # the runtime detector.
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.tree import DecisionTreeClassifier

    scores, features_multi, labels = _extract_supervised(predictions, truth)
    if not scores:
        raise ValueError(
            "No supervised predictions found -- nothing to tune. "
            "Did you pass the right gt_map / dataset for these predictions?"
        )

    # ── 1. Grid search ────────────────────────────────────────────────
    (grid_minor, grid_exact), best_wf1, grid_metrics = _grid_search(
        scores, labels,
        grid_min=grid_min, grid_max=grid_max, grid_step=grid_step,
    )
    grid_preds = [classify_score(s, grid_exact, grid_minor) for s in scores]

    # ── 2. Decision tree (depth=2 on llm_score alone) ─────────────────
    X_1d = np.array(scores).reshape(-1, 1)
    y = np.array(labels)
    dt = DecisionTreeClassifier(max_depth=2, random_state=rf_random_state)
    dt.fit(X_1d, y)
    dt_thresholds = sorted(set(
        float(t) for t in dt.tree_.threshold if t != -2.0
    ))
    dt_preds = dt.predict(X_1d).tolist()
    dt_metrics = calculate_metrics(y.tolist(), dt_preds)

    # ── 3. Random forest (multi-feature) ──────────────────────────────
    feature_names = ("llm_score", "match_title_similarity", "match_confidence")
    X_multi = np.array(features_multi)
    rf = RandomForestClassifier(
        n_estimators=rf_n_estimators,
        max_depth=rf_max_depth,
        random_state=rf_random_state,
        class_weight="balanced",
    )
    n_folds = min(5, len(np.unique(y)))
    cv_scores = cross_val_score(
        rf, X_multi, y, cv=n_folds, scoring="f1_weighted",
    )
    rf.fit(X_multi, y)
    rf_preds = rf.predict(X_multi).tolist()
    rf_metrics = calculate_metrics(y.tolist(), rf_preds)
    importances = dict(zip(feature_names, rf.feature_importances_.tolist()))

    if verbose:
        _print_optimisation_block(
            grid_minor=grid_minor, grid_exact=grid_exact,
            best_wf1=best_wf1, grid_metrics=grid_metrics, grid_preds=grid_preds,
            grid_labels=labels,
            dt_thresholds=dt_thresholds, dt_metrics=dt_metrics,
            dt_labels=y.tolist(), dt_preds=dt_preds,
            rf_cv=cv_scores, rf_metrics=rf_metrics,
            rf_labels=y.tolist(), rf_preds=rf_preds,
            importances=importances,
            rf_n_estimators=rf_n_estimators,
            rf_max_depth=rf_max_depth,
            n_features=len(feature_names),
        )

    return {
        "grid_search": {
            "minor_threshold": round(grid_minor, 2),
            "exact_threshold": round(grid_exact, 2),
            "weighted_f1": round(best_wf1, 4),
            "metrics": grid_metrics,
        },
        "decision_tree": {
            "thresholds": [round(t, 2) for t in dt_thresholds],
            "metrics": dt_metrics,
        },
        "random_forest": {
            "cv_weighted_f1_mean": round(float(cv_scores.mean()), 4),
            "cv_weighted_f1_std": round(float(cv_scores.std()), 4),
            "feature_importances": {
                k: round(v, 4) for k, v in importances.items()
            },
            "metrics": rf_metrics,
        },
        "recommended": {
            "minor_threshold": round(grid_minor, 2),
            "exact_threshold": round(grid_exact, 2),
        },
    }


# ---------------------------------------------------------------------------
# Verbose printout
# ---------------------------------------------------------------------------

def _print_optimisation_block(
    *,
    grid_minor: float,
    grid_exact: float,
    best_wf1: float,
    grid_metrics: Dict,
    grid_preds: List[str],
    grid_labels: Sequence[str],
    dt_thresholds: List[float],
    dt_metrics: Dict,
    dt_labels: List[str],
    dt_preds: List[str],
    rf_cv,            # numpy.ndarray
    rf_metrics: Dict,
    rf_labels: List[str],
    rf_preds: List[str],
    importances: Dict[str, float],
    rf_n_estimators: int,
    rf_max_depth: int,
    n_features: int,
) -> None:
    bar = "=" * 60
    print(f"\n{bar}\nTHRESHOLD OPTIMISATION\n{bar}")

    print(f"\n1) Grid search (best weighted-F1 = {best_wf1:.4f}):")
    print(f"   minor_threshold = {grid_minor:.2f}")
    print(f"   exact_threshold = {grid_exact:.2f}")
    print_metrics(grid_metrics, "   Grid-search metrics")
    print_confusion_matrix(grid_labels, grid_preds)

    print(f"\n2) Decision tree (depth=2) on llm_score:")
    for i, t in enumerate(dt_thresholds, start=1):
        print(f"   split {i}: {t:.2f}")
    print_metrics(dt_metrics, "   Decision-tree metrics (resubstitution)")
    print_confusion_matrix(dt_labels, dt_preds)

    print(
        f"\n3) Random forest ({rf_n_estimators} trees, "
        f"depth={rf_max_depth}, {n_features} features):"
    )
    print(
        f"   CV weighted-F1: {rf_cv.mean():.4f} (+/- {rf_cv.std():.4f})"
    )
    print("   Feature importances:")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"     {feat:>25s}: {imp:.4f}")
    print_metrics(rf_metrics, "   Random-forest metrics (resubstitution)")
    print_confusion_matrix(rf_labels, rf_preds)
