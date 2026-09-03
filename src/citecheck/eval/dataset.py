"""Typed loader for ``dataset/corruption_metadata.json``.

The package's entire ground truth lives in a single JSON file -- see
``dataset/SCHEMA.md`` for the on-disk layout.  This module provides
three frozen dataclasses (:class:`CitationRecord`, :class:`Collection`,
:class:`Dataset`) plus :func:`load_dataset` that deserialise that JSON
into a structure the rest of the eval pipeline can consume without
having to know about JSON keys.

Conventions used by callers
---------------------------

- The detector's *input* is :attr:`CitationRecord.corrupted_citation` --
  this is what users would actually paste into a report.  The
  :attr:`~CitationRecord.original_citation` is reserved for analysis
  (e.g. measuring how far the corruption strayed from the source).
- :attr:`CitationRecord.ground_truth_label` is the raw dataset label
  (``"valid"`` / ``"hallucinated_minor"`` / ``"hallucinated_major"``).
  Use :mod:`citecheck.eval.ground_truth` to map it onto
  the detector's label vocabulary.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

from .. import config


__all__ = (
    "CitationRecord",
    "Collection",
    "Dataset",
    "load_dataset",
)


@dataclass(frozen=True)
class CitationRecord:
    """One labelled citation from ``corruption_metadata.json``."""

    citation_number: int
    original_citation: str
    corrupted_citation: str
    ground_truth_label: str  # "valid" | "hallucinated_minor" | "hallucinated_major"
    error_type: str          # "none" | "hallucination"
    error_severity: Optional[str] = None  # "minor" | "major" | None (when error_type="none")
    changes: Optional[str] = None


@dataclass(frozen=True)
class Collection:
    """A single subtopic-worth of citations, identified by *collection_id*."""

    collection_id: str  # e.g. "1-astrophysics"
    topic: str          # e.g. "astrophysics"
    subtopic: str       # H1 title of the underlying report
    total_citations: int
    corruption_summary: Mapping[str, int]
    citations: Tuple[CitationRecord, ...]

    @property
    def collection_number(self) -> int:
        """Numeric prefix of the collection id (``"1-astrophysics" -> 1``)."""
        head = self.collection_id.split("-", 1)[0]
        return int(head) if head.isdigit() else -1


@dataclass(frozen=True)
class Dataset:
    """Top-level container with all collections and the generation config."""

    collections: Tuple[Collection, ...]
    configuration: Mapping[str, object] = field(default_factory=dict)
    generation_timestamp: Optional[str] = None

    # ── Convenience lookups ──────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.collections)

    def __iter__(self):
        return iter(self.collections)

    @property
    def total_citations(self) -> int:
        return sum(c.total_citations for c in self.collections)

    @property
    def collection_ids(self) -> Tuple[str, ...]:
        return tuple(c.collection_id for c in self.collections)

    @property
    def collection_numbers(self) -> Tuple[int, ...]:
        return tuple(c.collection_number for c in self.collections)

    def by_id(self, collection_id: str) -> Collection:
        for c in self.collections:
            if c.collection_id == collection_id:
                return c
        raise KeyError(f"Unknown collection_id: {collection_id!r}")

    def by_number(self, n: int) -> Collection:
        for c in self.collections:
            if c.collection_number == n:
                return c
        raise KeyError(f"No collection with numeric prefix {n}")

    def select(
        self,
        *,
        ids: Optional[List[str]] = None,
        numbers: Optional[List[int]] = None,
    ) -> Tuple[Collection, ...]:
        """Return collections filtered by *ids* and / or *numbers* (logical AND)."""
        out = self.collections
        if ids is not None:
            ids_set = set(ids)
            out = tuple(c for c in out if c.collection_id in ids_set)
        if numbers is not None:
            n_set = set(numbers)
            out = tuple(c for c in out if c.collection_number in n_set)
        return out


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _coerce_citation(raw: Dict) -> CitationRecord:
    """Build a :class:`CitationRecord` from one entry in ``citations``."""
    return CitationRecord(
        citation_number=int(raw["citation_number"]),
        original_citation=str(raw.get("original_citation", "")),
        corrupted_citation=str(raw.get("corrupted_citation", "")),
        ground_truth_label=str(raw["ground_truth_label"]),
        error_type=str(raw.get("error_type", "none")),
        error_severity=raw.get("error_severity"),
        changes=raw.get("changes"),
    )


def _coerce_collection(raw: Dict) -> Collection:
    """Build a :class:`Collection` from one entry in ``collections``."""
    summary = dict(raw.get("corruption_summary") or {})
    citations = tuple(_coerce_citation(c) for c in raw.get("citations", []))
    return Collection(
        collection_id=str(raw["collection_id"]),
        topic=str(raw.get("topic", "")),
        subtopic=str(raw.get("subtopic", "")),
        total_citations=int(raw.get("total_citations", len(citations))),
        corruption_summary=summary,
        citations=citations,
    )


def load_dataset(path: Optional[Path] = None) -> Dataset:
    """Load and parse ``corruption_metadata.json``.

    Parameters
    ----------
    path
        Path to the JSON file.  Defaults to
        :data:`citecheck.config.DEFAULT_DATASET_PATH` (the
        bundled dataset shipped with the package).
    """
    if path is None:
        path = config.DEFAULT_DATASET_PATH
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))

    if "collections" not in raw:
        raise ValueError(
            f"{path} does not look like a citecheck dataset "
            "(missing top-level 'collections' key)"
        )

    return Dataset(
        collections=tuple(_coerce_collection(c) for c in raw["collections"]),
        configuration=dict(raw.get("configuration") or {}),
        generation_timestamp=raw.get("generation_timestamp"),
    )
