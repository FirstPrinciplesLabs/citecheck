"""LLM layer: prompt templates, multi-provider client, verifier + reviewer."""

from __future__ import annotations

from .client import SUPPORTED_PROVIDERS, build_structured_llm
from .verification import (
    HallucinationClassification,
    ParsedCitation,
    ReviewerClassification,
    llm_parse_citation,
    review_classification,
    verify_with_llm,
)

__all__ = (
    # client
    "SUPPORTED_PROVIDERS",
    "build_structured_llm",
    # schemas
    "HallucinationClassification",
    "ReviewerClassification",
    "ParsedCitation",
    # verifier / reviewer / parser
    "verify_with_llm",
    "review_classification",
    "llm_parse_citation",
)
