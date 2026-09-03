"""Deterministic dev/test split + few-shot exemplar selection.

Given a :class:`~citecheck.eval.dataset.Dataset`, this module:

1. Picks **one collection per topic** as the dev set (the rest become
   test) using a seeded RNG.  See :func:`make_splits`.
2. For each dev collection, picks **one citation per class**
   (``exact_match`` / ``minor_hallucination`` / ``major_hallucination``)
   to form a 27-shot exemplar pool, again with a seeded RNG.  If the
   dev collection happens to lack one of the three classes, the
   selector falls back to other collections from the **same topic**.
   See :func:`make_fewshot_exemplars`.

Both the split and the exemplar pool are JSON-serialisable so the same
selection can be reloaded for downstream evaluation runs (the
detector's eval harness and the comparison baselines both consume the
same ``splits.json`` file).
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .dataset import CitationRecord, Collection, Dataset
from .ground_truth import CLASS_NAMES, to_class_label


__all__ = (
    "Exemplar",
    "Splits",
    "make_splits",
    "make_fewshot_exemplars",
    "save_splits",
    "load_splits",
)


@dataclass
class Exemplar:
    """A single labelled example shown to the LLM (verifier or baseline).

    Mirrors the fields needed to render a citation in a few-shot
    block: ``citation_text`` is what appears in the report (the
    *corrupted* citation), ``label`` is the class to teach, and
    ``reasoning`` is a short justification derived from the dataset's
    ``changes`` field.
    """

    collection_id: str
    collection_number: int
    topic: str
    citation_number: int
    citation_text: str          # corrupted citation as it appears in the report
    label: str                  # one of CLASS_NAMES
    reasoning: str              # short justification for the label
    original_citation: Optional[str] = None  # for debugging / context


@dataclass
class Splits:
    """Output of :func:`make_splits`."""

    seed: int
    dataset_path: str
    topic_to_dev_collection: Mapping[str, int]
    dev_collections: Sequence[int]   # numeric prefixes
    test_collections: Sequence[int]  # numeric prefixes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _by_topic(collections: Iterable[Collection]) -> Dict[str, List[int]]:
    """``{topic: sorted([collection_number, ...])}``."""
    out: Dict[str, List[int]] = {}
    for c in collections:
        out.setdefault(c.topic, []).append(c.collection_number)
    return {topic: sorted(nums) for topic, nums in out.items()}


def _candidates_for_class(
    collection: Collection, class_label: str,
) -> List[CitationRecord]:
    """All citations in *collection* whose class label is *class_label*."""
    return [
        c for c in collection.citations
        if to_class_label(c.ground_truth_label) == class_label
    ]


def _changes_to_reasoning(citation: CitationRecord, label: str) -> str:
    """Build a short reasoning string for an exemplar.

    For hallucinated citations we use the dataset's ``changes`` field
    (it describes exactly what was perturbed).  For valid citations we
    generate a generic justification.
    """
    if label == "exact_match":
        return (
            "The citation's authors, year, title, and identifiers all "
            "correctly identify the cited paper with no perturbations."
        )
    changes = (citation.changes or "").strip()
    if not changes:
        return f"The citation has been classified as {label}."
    return changes


def _build_exemplar(
    collection: Collection, citation: CitationRecord, label: str,
) -> Exemplar:
    text = citation.corrupted_citation or citation.original_citation or ""
    return Exemplar(
        collection_id=collection.collection_id,
        collection_number=collection.collection_number,
        topic=collection.topic,
        citation_number=citation.citation_number,
        citation_text=text,
        label=label,
        reasoning=_changes_to_reasoning(citation, label),
        original_citation=citation.original_citation,
    )


# ---------------------------------------------------------------------------
# Dev/test split
# ---------------------------------------------------------------------------

def make_splits(dataset: Dataset, *, seed: int = 42) -> Splits:
    """Pick one dev collection per topic; the rest become test.

    Deterministic given the same seed: topics are iterated in sorted
    order, and within each topic the candidate collection numbers are
    sorted before sampling.

    Parameters
    ----------
    dataset
        Loaded :class:`Dataset`.
    seed
        RNG seed.  Defaults to
        :data:`citecheck.config.DEFAULT_SEED`-style 42 to
        match the paper's split.
    """
    if not dataset.collections:
        raise RuntimeError("Dataset is empty -- nothing to split")

    by_topic = _by_topic(dataset.collections)

    rng = random.Random(seed)
    topic_to_dev: Dict[str, int] = {}
    for topic in sorted(by_topic):
        topic_to_dev[topic] = rng.choice(by_topic[topic])

    dev_set = set(topic_to_dev.values())
    all_nums = sorted(c.collection_number for c in dataset.collections)
    dev = sorted(dev_set)
    test = sorted(n for n in all_nums if n not in dev_set)

    dataset_path = (
        str(dataset.configuration.get("_source_path", ""))
        or "<in-memory dataset>"
    )

    return Splits(
        seed=seed,
        dataset_path=dataset_path,
        topic_to_dev_collection=topic_to_dev,
        dev_collections=dev,
        test_collections=test,
    )


# ---------------------------------------------------------------------------
# Few-shot exemplar selection
# ---------------------------------------------------------------------------

def make_fewshot_exemplars(
    dataset: Dataset,
    splits: Splits,
    *,
    seed: int = 42,
) -> List[Exemplar]:
    """For every (dev collection x class) cell, sample one exemplar.

    If the dev collection itself lacks the target class, fall back to
    other collections from the **same topic** (sorted, then
    RNG-shuffled) so every (topic x class) cell has an exemplar.

    Returns
    -------
    list[Exemplar]
        Up to ``len(dev_collections) * 3`` exemplars (27 for the
        default seed-42 split: 9 topics x 3 classes), shuffled
        deterministically.
    """
    by_topic_collections: Dict[str, List[Collection]] = {}
    for c in dataset.collections:
        by_topic_collections.setdefault(c.topic, []).append(c)

    by_number: Dict[int, Collection] = {
        c.collection_number: c for c in dataset.collections
    }

    rng = random.Random(seed)
    exemplars: List[Exemplar] = []

    for topic in sorted(splits.topic_to_dev_collection):
        dev_num = splits.topic_to_dev_collection[topic]
        dev = by_number.get(dev_num)
        if dev is None:
            print(f"[warn] no collection #{dev_num} for topic {topic!r}", flush=True)
            continue

        for cls in CLASS_NAMES:
            candidates = _candidates_for_class(dev, cls)
            chosen_collection = dev

            if not candidates:
                fallbacks = sorted(
                    (c for c in by_topic_collections.get(topic, [])
                     if c.collection_number != dev_num),
                    key=lambda c: c.collection_number,
                )
                rng.shuffle(fallbacks)
                for fb in fallbacks:
                    fc = _candidates_for_class(fb, cls)
                    if fc:
                        candidates = fc
                        chosen_collection = fb
                        print(
                            f"[info] dev collection {dev_num} ({topic}) had no "
                            f"'{cls}' citations; falling back to collection "
                            f"{fb.collection_number}",
                            flush=True,
                        )
                        break

            if not candidates:
                print(
                    f"[warn] no '{cls}' exemplar available for topic {topic!r} "
                    f"(dev collection {dev_num}); skipping",
                    flush=True,
                )
                continue

            citation = rng.choice(candidates)
            exemplars.append(_build_exemplar(chosen_collection, citation, cls))

    rng.shuffle(exemplars)
    return exemplars


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _splits_to_payload(
    splits: Splits, exemplars: Sequence[Exemplar],
) -> Dict[str, Any]:
    return {
        "seed": splits.seed,
        "dataset_path": splits.dataset_path,
        "topic_to_dev_collection": dict(splits.topic_to_dev_collection),
        "dev_collections": list(splits.dev_collections),
        "test_collections": list(splits.test_collections),
        "exemplars": [asdict(e) for e in exemplars],
    }


def save_splits(
    splits: Splits,
    exemplars: Sequence[Exemplar],
    out_path: Path,
) -> Path:
    """Write *splits* + *exemplars* to a single JSON file.

    Returns the resolved output path for convenience.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(_splits_to_payload(splits, exemplars), indent=2),
        encoding="utf-8",
    )
    return out_path


def load_splits(path: Path) -> Tuple[Splits, List[Exemplar]]:
    """Inverse of :func:`save_splits`.

    Tolerates legacy ``report``-named keys (``dev_reports``,
    ``test_reports``, ``topic_to_dev_report``, ``report_id``,
    ``report_number``) emitted by the previous package layout, so old
    splits files keep working.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    seed = int(raw.get("seed", 42))
    dataset_path = str(raw.get("dataset_path") or raw.get("reports_dir", ""))

    topic_to_dev_raw: Dict[str, Any] = (
        raw.get("topic_to_dev_collection")
        or raw.get("topic_to_dev_report")
        or {}
    )
    topic_to_dev = {k: int(v) for k, v in topic_to_dev_raw.items()}

    dev = list(map(int, raw.get("dev_collections") or raw.get("dev_reports") or []))
    test = list(map(int, raw.get("test_collections") or raw.get("test_reports") or []))

    splits = Splits(
        seed=seed,
        dataset_path=dataset_path,
        topic_to_dev_collection=topic_to_dev,
        dev_collections=dev,
        test_collections=test,
    )

    exemplars: List[Exemplar] = []
    for ex_raw in raw.get("exemplars") or []:
        ex_raw = dict(ex_raw)
        ex_raw.setdefault(
            "collection_id", ex_raw.pop("report_id", ""),
        )
        ex_raw.setdefault(
            "collection_number", int(ex_raw.pop("report_number", -1)),
        )
        ex_raw.pop("report_id", None)
        ex_raw.pop("report_number", None)
        exemplars.append(Exemplar(
            collection_id=str(ex_raw["collection_id"]),
            collection_number=int(ex_raw["collection_number"]),
            topic=str(ex_raw.get("topic", "")),
            citation_number=int(ex_raw["citation_number"]),
            citation_text=str(ex_raw.get("citation_text", "")),
            label=str(ex_raw["label"]),
            reasoning=str(ex_raw.get("reasoning", "")),
            original_citation=ex_raw.get("original_citation"),
        ))

    return splits, exemplars
