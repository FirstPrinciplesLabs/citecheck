"""LLM verification, reviewer, and citation-parser fallback.

Three structured-output LLM calls live here:

- :func:`verify_with_llm` -- primary verifier.  Given a citation and
  the cascade's best match, returns
  :class:`HallucinationClassification` (score, label, confidence,
  reasoning, key_differences).
- :func:`review_classification` -- second-pass reviewer that audits
  suspicious identifier-based matches; returns
  :class:`ReviewerClassification` (label, reasoning).
- :func:`llm_parse_citation` -- fallback citation parser used when the
  regex-based parser in :mod:`citecheck.core.citation`
  can't extract the title; mutates the input :class:`Citation` in
  place.

All three use the same multi-provider client factory from
:mod:`.client`.
"""

from __future__ import annotations

from typing import Dict, Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from .. import config
from ..core.citation import Citation
from .client import build_structured_llm
from .prompts import (
    MATCH_SYSTEM_PROMPT,
    MATCH_USER_PROMPT,
    NO_MATCH_SYSTEM_PROMPT,
    NO_MATCH_USER_PROMPT,
    REVIEWER_SYSTEM_PROMPT,
    REVIEWER_USER_PROMPT,
)


__all__ = (
    "HallucinationClassification",
    "ReviewerClassification",
    "ParsedCitation",
    "verify_with_llm",
    "review_classification",
    "llm_parse_citation",
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class HallucinationClassification(BaseModel):
    """Verifier output."""

    score: float = Field(
        description="Accuracy score from 0-10, where 10 is perfect match and 0 is complete hallucination",
    )
    classification: str = Field(
        description="One of: 'exact_match', 'minor_hallucination', 'major_hallucination'",
    )
    confidence: str = Field(
        description="Confidence level: 'high', 'medium', or 'low'",
    )
    reasoning: str = Field(
        description="Brief explanation of the score and classification decision",
    )
    key_differences: Optional[str] = Field(
        default=None,
        description="Key differences found between the citation and the source (if any)",
    )


class ReviewerClassification(BaseModel):
    """Reviewer output (no score -- the caller synthesises one)."""

    classification: str = Field(
        description="One of: 'exact_match', 'minor_hallucination', 'major_hallucination'",
    )
    reasoning: str = Field(
        description="Brief explanation of why you agree or disagree with the first-pass verifier",
    )


class ParsedCitation(BaseModel):
    """Citation parser fallback output."""

    authors: Optional[str] = Field(
        default=None,
        description="Author names (e.g. 'M. Micic et al.')",
    )
    year: Optional[int] = Field(default=None, description="Publication year")
    title: Optional[str] = Field(default=None, description="Paper title")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_LABELS = frozenset({
    "exact_match", "minor_hallucination", "major_hallucination",
})


def _override_label_from_score(
    result: HallucinationClassification,
    *,
    exact_threshold: float,
    minor_threshold: float,
) -> None:
    """Rewrite ``result.classification`` from ``result.score`` so they're consistent."""
    if result.score >= exact_threshold:
        result.classification = "exact_match"
    elif result.score >= minor_threshold:
        result.classification = "minor_hallucination"
    else:
        result.classification = "major_hallucination"


def _format_matched_source(matched_source: Dict) -> str:
    """Render a matched-source dict into a human-readable block.

    Two layouts depending on ``_source_type``:

    - ``"web"`` -- web-search hits without structured authors.
    - anything else -- the canonical arxiv-style layout used for
      CrossRef / Semantic Scholar / OpenAlex / arXiv results too.
    """
    source_type = matched_source.get("_source_type", "arxiv")

    if source_type == "web":
        snippet = matched_source.get("snippet", "N/A") or "N/A"
        return (
            "Web Source:\n"
            f"- Title: {matched_source.get('title', 'Unknown')}\n"
            f"- URL: {matched_source.get('url', 'Unknown')}\n"
            f"- Snippet: {snippet[:200]}"
        )

    authors_list = matched_source.get("authors", []) or []
    if len(authors_list) > 3:
        authors_display = ", ".join(authors_list[:3]) + " et al."
    else:
        authors_display = ", ".join(authors_list) if authors_list else "Unknown"

    published = matched_source.get("published", "Unknown") or "Unknown"
    return (
        "ArXiv Paper:\n"
        f"- Authors: {authors_display}\n"
        f"- Year: {published[:4]}\n"
        f"- Title: {matched_source.get('title', 'Unknown')}\n"
        f"- ArXiv ID: {matched_source.get('id', 'Unknown')}"
    )


def _context_line(similarity_score: float) -> str:
    """Return the 'strong/good/weak match' annotation for the user prompt."""
    if similarity_score >= 0.95:
        return "The programmatic matching found a very strong match (score >= 0.95)."
    if similarity_score >= 0.75:
        return "The programmatic matching found a good match (score 0.75-0.95)."
    return "The programmatic matching found a weak match (score < 0.75)."


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

def verify_with_llm(
    citation: Citation,
    matched_source: Optional[Dict],
    similarity_score: float,
    match_method: str,
    *,
    provider: str = config.LLM_PROVIDER,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    fewshot_block: Optional[str] = None,
    exact_threshold: float = config.EXACT_THRESHOLD,
    minor_threshold: float = config.MINOR_THRESHOLD,
) -> HallucinationClassification:
    """Run the primary verifier on a single citation.

    Parameters
    ----------
    citation
        Parsed citation from the report.
    matched_source
        The cascade's best match formatted as a dict, or ``None`` /
        empty when the cascade returned NotFound.
    similarity_score
        Title-similarity score normalised to 0-1.
    match_method
        How the source was found: ``"arxiv"``, ``"url"``, ``"title"``,
        or ``"none"``.
    provider, model, temperature
        Forwarded to :func:`citecheck.llm.client.build_structured_llm`.
    fewshot_block
        Optional pre-formatted block of labelled examples appended to
        the system prompt on every call (used by the few-shot eval mode).
    exact_threshold, minor_threshold
        Thresholds for the post-hoc label override that ensures
        ``score`` and ``classification`` are consistent regardless of
        what the LLM put in ``classification`` itself.
    """
    llm = build_structured_llm(
        HallucinationClassification,
        provider=provider, model=model, temperature=temperature,
    )

    no_match_branch = not matched_source or similarity_score == 0.0

    sys_prompt = NO_MATCH_SYSTEM_PROMPT if no_match_branch else MATCH_SYSTEM_PROMPT
    user_prompt = NO_MATCH_USER_PROMPT if no_match_branch else MATCH_USER_PROMPT
    if fewshot_block:
        sys_prompt = f"{sys_prompt}\n\n{fewshot_block}"

    template = ChatPromptTemplate.from_messages([
        ("system", sys_prompt),
        ("human", user_prompt),
    ])
    chain = template | llm

    invoke_args: Dict[str, object] = {
        "citation_authors": citation.authors or "Unknown",
        "citation_year": citation.year or "Unknown",
        "citation_title": citation.title or "No title",
        "citation_arxiv_id": citation.arxiv_id or "None",
        "citation_url": citation.url or "None",
    }

    if not no_match_branch:
        assert matched_source is not None  # narrow for type-checkers
        invoke_args.update(
            match_method=match_method.upper(),
            similarity_score=similarity_score,
            matched_info=_format_matched_source(matched_source),
            context=_context_line(similarity_score),
        )

    result: HallucinationClassification = chain.invoke(invoke_args)
    _override_label_from_score(
        result,
        exact_threshold=exact_threshold,
        minor_threshold=minor_threshold,
    )
    return result


# ---------------------------------------------------------------------------
# Reviewer
# ---------------------------------------------------------------------------

def review_classification(
    citation: Citation,
    matched_title: str,
    matched_authors: str,
    matched_year: str,
    match_source: str,
    title_similarity: float,
    verifier_label: str,
    verifier_score: float,
    verifier_reasoning: str,
    *,
    provider: str = config.LLM_PROVIDER,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> ReviewerClassification:
    """Run the second-pass reviewer.

    Parameters
    ----------
    citation
        The original citation from the report.
    matched_title, matched_authors, matched_year
        Metadata of the source paper found by the verifier.
    match_source
        How the source was found (``"arXiv"``, ``"WebSearch"``, ...).
    title_similarity
        Title similarity percentage (0-100).
    verifier_label, verifier_score, verifier_reasoning
        The first-pass verifier's verdict that the reviewer is auditing.
    """
    llm = build_structured_llm(
        ReviewerClassification,
        provider=provider, model=model, temperature=temperature,
    )

    template = ChatPromptTemplate.from_messages([
        ("system", REVIEWER_SYSTEM_PROMPT),
        ("human", REVIEWER_USER_PROMPT),
    ])
    chain = template | llm

    result: ReviewerClassification = chain.invoke({
        "citation_raw_text": citation.raw_text or "",
        "citation_authors": citation.authors or "Unknown",
        "citation_year": citation.year or "Unknown",
        "citation_title": citation.title or "No title",
        "source_authors": matched_authors or "Unknown",
        "source_year": matched_year or "Unknown",
        "source_title": matched_title or "Unknown",
        "match_source": match_source,
        "title_similarity": title_similarity,
        "verifier_label": verifier_label,
        "verifier_score": verifier_score,
        "verifier_reasoning": verifier_reasoning,
    })

    if result.classification not in _VALID_LABELS:
        result.classification = "major_hallucination"
    return result


# ---------------------------------------------------------------------------
# Citation-parser fallback
# ---------------------------------------------------------------------------

_PARSE_CITATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a citation parser. Extract the authors, publication year, "
        "and paper title from the raw citation text below. The citation may "
        "be in various formats. Return only the extracted fields.",
    ),
    ("human", "Raw citation text:\n{raw_text}"),
])


def llm_parse_citation(
    citation: Citation,
    *,
    provider: str = config.LLM_PROVIDER,
    model: Optional[str] = None,
    temperature: Optional[float] = 0.0,
) -> Citation:
    """Use an LLM to extract authors / year / title from *citation*.

    Only intended as a fallback when the regex parser couldn't find the
    title.  Mutates *citation* in place, only filling fields that are
    currently ``None`` -- regex-extracted values always win.
    """
    llm = build_structured_llm(
        ParsedCitation,
        provider=provider, model=model, temperature=temperature,
    )

    chain = _PARSE_CITATION_PROMPT | llm
    parsed: ParsedCitation = chain.invoke({"raw_text": citation.raw_text})

    if parsed.authors and citation.authors is None:
        citation.authors = parsed.authors
    if parsed.year and citation.year is None:
        citation.year = parsed.year
    if parsed.title and citation.title is None:
        citation.title = parsed.title

    return citation
