"""Core: citation parsing, similarity, external-API cascade, classification."""

from __future__ import annotations

from .citation import Citation, extract_arxiv_id, parse_citation
from .classify import classify_score, score_for_label
from .detector import detect_citations, detect_one
from .similarity import (
    clean_query,
    levenshtein,
    title_similarity,
    word_overlap_similarity,
)
from .verifier import (
    MatchResult,
    NOT_FOUND,
    VerificationResult,
    find_closest_reference,
)

__all__ = (
    # citation
    "Citation",
    "extract_arxiv_id",
    "parse_citation",
    # classify
    "classify_score",
    "score_for_label",
    # similarity
    "clean_query",
    "levenshtein",
    "title_similarity",
    "word_overlap_similarity",
    # verifier (cascade)
    "MatchResult",
    "NOT_FOUND",
    "VerificationResult",
    "find_closest_reference",
    # detector (orchestrator)
    "detect_one",
    "detect_citations",
)
