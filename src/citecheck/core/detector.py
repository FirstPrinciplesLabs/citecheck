"""End-to-end detector: cascade + verifier + reviewer + parser fallback.

Two public entry points:

- :func:`detect_one` -- async, runs the full pipeline on a single
  :class:`~citecheck.core.citation.Citation`.
- :func:`detect_citations` -- sync, iterates :func:`detect_one` over a
  list of citations with an optional progress bar and per-citation
  resource monitoring.

All defaults are pulled from :mod:`citecheck.config`, so
calling these functions with no options reproduces the paper's
headline configuration.
"""

from __future__ import annotations

import asyncio
import time
from typing import Dict, List, Optional

from tqdm import tqdm

from .. import config
from ..llm.verification import (
    llm_parse_citation,
    review_classification,
    verify_with_llm,
)
from ..monitoring import monitor_context
from .citation import Citation
from .classify import score_for_label
from .verifier import MatchResult, VerificationResult, find_closest_reference


__all__ = ("detect_one", "detect_citations")


# ---------------------------------------------------------------------------
# Helpers: MatchResult -> matched_source dict / match_method string
# ---------------------------------------------------------------------------

_MATCH_METHODS: Dict[str, str] = {
    "CrossRef": "title",
    "SemanticScholar": "title",
    "OpenAlex": "title",
    "arXiv": "arxiv",
    "WebSearch": "url",
    "NotFound": "none",
}

_IDENTIFIER_BASED_SOURCES = frozenset({"arXiv", "WebSearch"})


def _match_to_source_dict(match: MatchResult) -> Optional[Dict]:
    """Render a :class:`MatchResult` into the dict shape ``verify_with_llm`` expects."""
    if not match or not match.found:
        return None

    if match.source == "WebSearch" and not match.authors:
        return {
            "_source_type": "web",
            "title": match.title or "",
            "url": match.url or "",
            "snippet": (match.raw_response or {}).get("match_explanation", ""),
        }

    if isinstance(match.authors, list):
        authors_list = match.authors
    elif match.authors:
        authors_list = [a.strip() for a in match.authors.split(",")]
    else:
        authors_list = []

    return {
        "_source_type": "arxiv",
        "title": match.title or "",
        "authors": authors_list,
        "published": match.year or "",
        "id": match.doi or match.url or "",
    }


def _match_method(source: str) -> str:
    return _MATCH_METHODS.get(source, "none")


def _has_arxiv_signal(citation: Citation) -> bool:
    """Return True if *citation* mentions arXiv anywhere -- triggers arxiv-first cascade ordering."""
    if citation.arxiv_id:
        return True
    if citation.url and "arxiv" in citation.url.lower():
        return True
    if citation.raw_text and "arxiv" in citation.raw_text.lower():
        return True
    return False


# ---------------------------------------------------------------------------
# Single-citation pipeline
# ---------------------------------------------------------------------------

async def detect_one(
    citation: Citation,
    *,
    # Cascade parameters
    min_title_similarity: float = config.MIN_TITLE_SIMILARITY,
    fallback_confidence_threshold: float = config.FALLBACK_CONFIDENCE_THRESHOLD,
    try_arxiv: bool = config.TRY_ARXIV,
    try_web_search: bool = config.TRY_WEB_SEARCH,
    web_search_model: str = config.WEB_SEARCH_MODEL,
    web_search_candidates: int = config.WEB_SEARCH_CANDIDATES,
    accept_best_web_search: bool = config.ACCEPT_BEST_WEB_SEARCH,
    api_timeout: float = config.API_TIMEOUT,
    inter_api_delay: float = config.INTER_API_DELAY,
    # Verifier LLM
    llm_provider: str = config.LLM_PROVIDER,
    llm_model: Optional[str] = None,
    llm_temperature: Optional[float] = None,
    fewshot_block: Optional[str] = None,
    # Reviewer LLM
    review_enabled: bool = config.REVIEW_ENABLED,
    review_model: Optional[str] = config.REVIEW_MODEL,
    review_sim_threshold: float = config.REVIEW_SIM_THRESHOLD,
    # Classification thresholds (for reviewer-override score synthesis)
    exact_threshold: float = config.EXACT_THRESHOLD,
    minor_threshold: float = config.MINOR_THRESHOLD,
) -> Dict:
    """Find the closest reference for *citation* and classify it.

    Returns a prediction dict in the shape consumed by the eval harness
    and the CLI.  When the reviewer overrides the verifier, the
    reviewer's label wins and a representative score is synthesised
    from :func:`score_for_label`.
    """

    arxiv_first = _has_arxiv_signal(citation)

    _t_api = time.perf_counter()
    vr: VerificationResult = await find_closest_reference(
        citation,
        min_title_similarity=min_title_similarity,
        fallback_confidence_threshold=fallback_confidence_threshold,
        try_arxiv=try_arxiv,
        arxiv_first=arxiv_first,
        try_web_search=try_web_search,
        web_search_model=web_search_model,
        web_search_candidates=web_search_candidates,
        accept_best_web_search=accept_best_web_search,
        api_timeout=api_timeout,
        inter_api_delay=inter_api_delay,
    )
    api_elapsed = time.perf_counter() - _t_api

    chosen = vr.chosen
    matched_source = _match_to_source_dict(chosen)
    similarity_score = chosen.title_similarity / 100.0 if chosen.found else 0.0
    method = _match_method(chosen.source)

    _t_llm = time.perf_counter()
    classification = verify_with_llm(
        citation=citation,
        matched_source=matched_source,
        similarity_score=similarity_score,
        match_method=method,
        provider=llm_provider,
        model=llm_model,
        temperature=llm_temperature,
        fewshot_block=fewshot_block,
        exact_threshold=exact_threshold,
        minor_threshold=minor_threshold,
    )
    llm_elapsed = time.perf_counter() - _t_llm

    pred: Dict = {
        "citation_number": citation.number,
        "citation_text": citation.raw_text,
        "citation_authors": citation.authors,
        "citation_year": citation.year,
        "citation_title": citation.title,
        "citation_url": citation.url,
        "citation_arxiv_id": citation.arxiv_id,
        "predicted_label": classification.classification,
        "llm_score": classification.score,
        "llm_confidence": classification.confidence,
        "llm_reasoning": classification.reasoning,
        "llm_key_differences": classification.key_differences,
        "match_method": method,
        "match_source_api": chosen.source,
        "match_score": similarity_score,
        "match_title_similarity": chosen.title_similarity,
        "match_confidence": chosen.confidence,
        "matched_source": matched_source,
        "matched_title": chosen.title,
        "matched_authors": chosen.authors,
        "matched_year": chosen.year,
        "matched_url": chosen.url,
        "reviewer_triggered": False,
        "verification_result": {
            "crossref_queried": vr.crossref is not None,
            "semantic_scholar_queried": vr.semantic_scholar is not None,
            "openalex_queried": vr.openalex is not None,
            "arxiv_queried": vr.arxiv is not None,
            "web_search_queried": vr.web_search is not None,
        },
    }

    reviewer_elapsed = 0.0

    # ── Reviewer pass ─────────────────────────────────────────────────
    if (
        review_enabled
        and chosen.found
        and chosen.source in _IDENTIFIER_BASED_SOURCES
        and chosen.title_similarity < review_sim_threshold
    ):
        matched_authors_str = (
            ", ".join(chosen.authors)
            if isinstance(chosen.authors, list)
            else (chosen.authors or "")
        )

        _t_rev = time.perf_counter()
        review = review_classification(
            citation=citation,
            matched_title=chosen.title or "",
            matched_authors=matched_authors_str,
            matched_year=chosen.year or "",
            match_source=chosen.source,
            title_similarity=chosen.title_similarity,
            verifier_label=classification.classification,
            verifier_score=classification.score,
            verifier_reasoning=classification.reasoning or "",
            provider=llm_provider,
            model=review_model,
            temperature=llm_temperature,
        )
        reviewer_elapsed = time.perf_counter() - _t_rev

        new_score = score_for_label(
            review.classification,
            exact_threshold=exact_threshold,
            minor_threshold=minor_threshold,
        )

        pred.update(
            reviewer_triggered=True,
            reviewer_label=review.classification,
            reviewer_reasoning=review.reasoning,
            original_llm_score=classification.score,
            original_llm_label=classification.classification,
            llm_score=new_score,
            predicted_label=review.classification,
        )

    if pred["predicted_label"] == "minor_hallucination" and chosen.found:
        authors = chosen.authors
        if isinstance(authors, list):
            authors = ", ".join(authors)
        pred["suggested_corrected_version"] = {
            "title": chosen.title,
            "authors": authors,
            "year": chosen.year,
            "url": chosen.url,
            "doi": chosen.doi,
            "source_api": chosen.source,
        }

    pred["perf_timing"] = {
        "api_search_seconds": round(api_elapsed, 4),
        "llm_inference_seconds": round(llm_elapsed, 4),
        "reviewer_seconds": round(reviewer_elapsed, 4),
        "api_timings": dict(vr.api_timings),
    }

    return pred


# ---------------------------------------------------------------------------
# List-level pipeline
# ---------------------------------------------------------------------------

def detect_citations(
    citations: List[Citation],
    *,
    verbose: bool = True,
    monitor: bool = False,
    llm_parse_enabled: bool = config.LLM_PARSE_ENABLED,
    llm_parse_model: Optional[str] = config.LLM_PARSE_MODEL,
    **detect_one_kwargs,
) -> List[Dict]:
    """Run :func:`detect_one` on every citation and collect predictions.

    When *llm_parse_enabled*, citations missing a title are passed
    through the LLM citation-parser fallback before running the
    cascade.  When *monitor*, each citation's wall-clock / CPU / memory
    usage is recorded under ``pred["perf"]``.
    """
    if llm_parse_enabled:
        provider = detect_one_kwargs.get("llm_provider", config.LLM_PROVIDER)
        for cite in citations:
            if cite.title is None:
                llm_parse_citation(
                    cite, provider=provider, model=llm_parse_model,
                )

    iterator = (
        tqdm(citations, desc="Verifying citations", unit="cite")
        if verbose
        else citations
    )

    predictions: List[Dict] = []
    for cite in iterator:
        with monitor_context(monitor) as cite_mon:
            pred = asyncio.run(detect_one(cite, **detect_one_kwargs))

        timing = pred.pop("perf_timing", {})
        if monitor and cite_mon is not None:
            perf = cite_mon.metrics
            perf["api_search_seconds"] = timing.get("api_search_seconds", 0)
            perf["llm_inference_seconds"] = timing.get("llm_inference_seconds", 0)
            perf["reviewer_seconds"] = timing.get("reviewer_seconds", 0)
            perf["api_timings"] = timing.get("api_timings", {})
            pred["perf"] = perf

        predictions.append(pred)

    return predictions
