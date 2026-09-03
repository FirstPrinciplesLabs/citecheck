"""Score-to-label classification.

The verifier LLM emits a 0-10 score; this module turns it into one of
the three discrete labels using a two-threshold scheme:

    score >= exact_threshold       -> exact_match
    score >= minor_threshold       -> minor_hallucination
    score <  minor_threshold       -> major_hallucination

The reverse direction (label -> representative score) is also provided
so downstream stages can synthesise a numeric score for label-only
outputs (e.g. the reviewer LLM, which returns a label without a score).
"""

from __future__ import annotations

from .. import config


__all__ = ("classify_score", "score_for_label")


def classify_score(
    score: float,
    exact_threshold: float = config.EXACT_THRESHOLD,
    minor_threshold: float = config.MINOR_THRESHOLD,
) -> str:
    """Return the discrete label for *score* under the two-threshold rule."""
    if score >= exact_threshold:
        return "exact_match"
    if score >= minor_threshold:
        return "minor_hallucination"
    return "major_hallucination"


def score_for_label(
    label: str,
    exact_threshold: float = config.EXACT_THRESHOLD,
    minor_threshold: float = config.MINOR_THRESHOLD,
) -> float:
    """Return a representative score that lands in *label*'s region.

    Used by the reviewer pass to synthesise a numeric score consistent
    with the user's chosen thresholds whenever the reviewer LLM
    overrides the verifier's label.
    """
    if label == "exact_match":
        return (exact_threshold + 10.0) / 2.0
    if label == "minor_hallucination":
        return (minor_threshold + exact_threshold) / 2.0
    return minor_threshold / 2.0
