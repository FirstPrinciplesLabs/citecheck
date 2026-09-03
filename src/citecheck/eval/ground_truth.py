"""Ground-truth label mapping and lookup tables.

The dataset uses three raw labels (``valid``, ``hallucinated_minor``,
``hallucinated_major``) which map onto the detector's three
classification labels (``exact_match``, ``minor_hallucination``,
``major_hallucination``).  This module makes that mapping explicit and
provides a flat lookup table indexed by ``(collection_id,
citation_number)`` that's convenient for joining predictions with the
truth.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping

from .dataset import Collection, Dataset


__all__ = (
    "CLASS_NAMES",
    "RAW_TO_CLASS",
    "to_class_label",
    "build_gt_map",
    "build_flat_gt_map",
    "resolve_gt_map",
)


CLASS_NAMES: tuple[str, ...] = (
    "exact_match",
    "minor_hallucination",
    "major_hallucination",
)


# Schema-level mapping: dataset's raw labels -> detector's labels.
RAW_TO_CLASS: Mapping[str, str] = {
    "valid": "exact_match",
    "hallucinated_minor": "minor_hallucination",
    "hallucinated_major": "major_hallucination",
}


def to_class_label(raw_label: str) -> str:
    """Map a raw dataset label to the detector's label vocabulary.

    Unknown labels fall through to ``"exact_match"`` to preserve
    behaviour from the legacy code; in practice every label in the
    bundled dataset is one of the three known values.
    """
    return RAW_TO_CLASS.get(raw_label, "exact_match")


def _per_collection(collection: Collection) -> Dict[int, str]:
    """``{citation_number: class_label}`` for one collection."""
    return {
        c.citation_number: to_class_label(c.ground_truth_label)
        for c in collection.citations
    }


def build_gt_map(
    source: Dataset | Iterable[Collection],
) -> Dict[str, Dict[int, str]]:
    """Return ``{collection_id: {citation_number: class_label}}``.

    Accepts either a full :class:`Dataset` or any iterable of
    :class:`Collection` (e.g. a split's collections).
    """
    collections: Iterable[Collection] = (
        source.collections if isinstance(source, Dataset) else source
    )
    return {c.collection_id: _per_collection(c) for c in collections}


def build_flat_gt_map(
    source: Dataset | Iterable[Collection],
) -> Dict[tuple[str, int], str]:
    """Flat variant of :func:`build_gt_map` keyed on ``(collection_id, citation_number)``."""
    collections: Iterable[Collection] = (
        source.collections if isinstance(source, Dataset) else source
    )
    flat: Dict[tuple[str, int], str] = {}
    for col in collections:
        for c in col.citations:
            flat[(col.collection_id, c.citation_number)] = to_class_label(
                c.ground_truth_label,
            )
    return flat


def resolve_gt_map(
    gt_map: Dict[str, Dict[int, str]] | None,
    dataset: Dataset | None,
) -> Dict[str, Dict[int, str]]:
    """Return *gt_map* if non-None, else build one from *dataset*.

    Convenience used by callers (eval runner, threshold tuner) that
    accept either source -- raises :class:`ValueError` when both are
    ``None``.
    """
    if gt_map is not None:
        return gt_map
    if dataset is not None:
        return build_gt_map(dataset)
    raise ValueError("Provide either gt_map or dataset")
