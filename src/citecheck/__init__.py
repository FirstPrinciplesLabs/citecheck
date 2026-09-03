"""CiteCheck: retrieval-grounded detection of LLM citation hallucinations.

The detector retrieves candidate publications from CrossRef, Semantic
Scholar, OpenAlex, arXiv, and (optionally) the open web, then asks a
structured LLM verifier whether the candidate matches the citation's
metadata.  Verifier scores are mapped to ``exact_match`` /
``minor_hallucination`` / ``major_hallucination`` via two tunable
thresholds.

Top-level convenience imports
-----------------------------
The most common entry points are re-exported here so simple scripts
only need a single ``import citecheck`` line::

    import citecheck

    # programmatic detection on a list of citation strings
    from citecheck.core import parse_citation
    citations = [parse_citation(i, t) for i, t in enumerate(my_strings, start=1)]
    predictions = citecheck.detect_citations(citations)

    # full evaluation harness with threshold tuning
    citecheck.evaluate_dataset(save_dir="results/claude")

For finer-grained access (cascade primitives, prompt templates,
classification helpers, etc.), import from the corresponding subpackage
(:mod:`citecheck.core`, :mod:`citecheck.eval`,
:mod:`citecheck.baselines`, :mod:`citecheck.llm`).
"""

from __future__ import annotations

from . import config
from .core.classify import classify_score, score_for_label
from .core.detector import detect_citations, detect_one
from .eval.dataset import load_dataset
from .eval.harness import evaluate_dataset


__version__ = "0.1.0"

__all__ = (
    "__version__",
    "config",
    # core
    "detect_one",
    "detect_citations",
    "classify_score",
    "score_for_label",
    # eval
    "load_dataset",
    "evaluate_dataset",
)
