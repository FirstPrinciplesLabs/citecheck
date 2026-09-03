"""External-API cascade: find the closest real reference for a citation.

Five sources are queried in cascading fallback order:

    1. CrossRef          (primary -- largest DOI registry)
    2. Semantic Scholar  (fallback -- strong on CS / bio / med)
    3. OpenAlex          (fallback -- broad open-access catalogue)
    4. arXiv             (optional -- direct ID lookup)
    5. Web Search        (optional -- LLM-powered last resort)

The first three stages perform title-based searches.  The 4th stage
(``try_arxiv=True``) performs a direct ID lookup when the citation
contains an arXiv ID.  The 5th stage (``try_web_search=True``) uses
OpenAI's ``web_search_preview`` tool to look up the citation on the
open web.

When ``arxiv_first=True``, arXiv is promoted to stage 0 and the other
stages shift accordingly.

The cascade returns a :class:`VerificationResult` carrying the chosen
match plus every API's individual best result, so downstream code can
inspect why a particular match was selected.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import httpx

from .citation import Citation
from .similarity import clean_query, title_similarity, word_overlap_similarity
from .web_search_prompts import WEB_SEARCH_SYSTEM, WEB_SEARCH_USER


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    """One candidate match for a citation, returned by a single API stage.

    Attributes:
        found:             Whether any sufficiently similar reference was found.
        title:             Title of the best-matching reference.
        authors:           Comma-separated author string of the match.
        year:              Publication year of the match.
        journal:           Journal / venue name.
        doi:               DOI string (without URL prefix).
        url:               Best available URL for the matched work.
        source:            Which API provided the match
                           ("CrossRef", "SemanticScholar", "OpenAlex",
                           "arXiv", "WebSearch", "NotFound").
        title_similarity:  0-100 score between the query title and the found title.
        confidence:        0-100 overall confidence the match is correct.
        raw_response:      The raw JSON returned by the winning API
                           (kept for debugging / downstream analysis).
    """
    found: bool = False
    title: Optional[str] = None
    authors: Optional[str] = None
    year: Optional[str] = None
    journal: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    source: str = "NotFound"
    title_similarity: float = 0.0
    confidence: float = 0.0
    raw_response: Optional[dict] = field(default=None, repr=False)


NOT_FOUND = MatchResult(found=False, source="NotFound")


@dataclass
class VerificationResult:
    """Full output of the cascade for a single citation.

    Attributes:
        chosen:           The best :class:`MatchResult` selected by the cascade.
                          ``chosen.source`` tells you which API provided it.
        crossref / semantic_scholar / openalex / arxiv / web_search:
                          Each API's best result.  Three possible states:
                            - ``None``                   -> API was not queried
                            - ``MatchResult(found=False)`` -> queried but nothing found
                            - ``MatchResult(found=True)``  -> queried and got a result
        web_search_all:   All candidates returned by the web search LLM,
                          sorted by title similarity (highest first).
        api_timings:      Per-stage wall-clock seconds.
    """
    chosen: MatchResult = field(default_factory=lambda: NOT_FOUND)
    crossref: Optional[MatchResult] = None
    semantic_scholar: Optional[MatchResult] = None
    openalex: Optional[MatchResult] = None
    arxiv: Optional[MatchResult] = None
    web_search: Optional[MatchResult] = None
    web_search_all: Optional[List[MatchResult]] = None
    api_timings: Dict[str, float] = field(default_factory=dict)

    @property
    def found(self) -> bool:
        return self.chosen.found

    @property
    def source(self) -> str:
        return self.chosen.source


# ---------------------------------------------------------------------------
# CrossRef
# ---------------------------------------------------------------------------

async def _query_crossref(
    query: str, *, max_results: int = 5, timeout: float = 15.0,
) -> list[dict]:
    """Search CrossRef for works matching *query*."""
    url = "https://api.crossref.org/works"
    params = {"query": query, "rows": max_results}
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.json().get("message", {}).get("items", [])


def _best_crossref(
    items: list[dict], query_title: str, expected_year: Optional[str] = None,
) -> Optional[dict]:
    """Pick the highest-scoring CrossRef item.

    Scoring: ``title_similarity * 3``, plus +40 if the year matches
    exactly, +10 if it's off by 1 (common for preprint -> published).
    """
    if not items:
        return None
    best, best_score = None, -1.0
    for item in items:
        t = (item.get("title") or [""])[0]
        score = title_similarity(query_title, t) * 3
        yr = str((item.get("published") or {}).get("date-parts", [[None]])[0][0] or "")
        if expected_year and yr:
            if yr == expected_year:
                score += 40
            elif yr.isdigit() and expected_year.isdigit() and abs(int(yr) - int(expected_year)) == 1:
                score += 10
        if score > best_score:
            best_score = score
            best = item
    return best


def _crossref_to_match(item: dict, query_title: str) -> MatchResult:
    """Normalise a raw CrossRef item into :class:`MatchResult`."""
    result_title = (item.get("title") or [""])[0]
    authors_list = item.get("author") or []
    authors_str = ", ".join(a.get("family", "") for a in authors_list)
    year = str((item.get("published") or {}).get("date-parts", [[None]])[0][0] or "")
    journal = (item.get("container-title") or [""])[0]
    sim = title_similarity(query_title, result_title)
    return MatchResult(
        found=True,
        title=result_title, authors=authors_str, year=year, journal=journal,
        doi=item.get("DOI"), url=item.get("URL"),
        source="CrossRef",
        title_similarity=sim,
        confidence=sim,
        raw_response=item,
    )


# ---------------------------------------------------------------------------
# Semantic Scholar
# ---------------------------------------------------------------------------

async def _query_semantic_scholar(
    title: str, *, max_results: int = 5, timeout: float = 15.0,
) -> list[dict]:
    """Search Semantic Scholar for papers matching *title*."""
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    fields = "paperId,title,authors,year,venue,externalIds,url"
    params = {"query": title, "limit": max_results, "fields": fields}
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(url, params=params, headers={"Accept": "application/json"})
        if resp.status_code == 429:
            return []
        resp.raise_for_status()
        return resp.json().get("data", [])


def _best_semantic_scholar(
    papers: list[dict], query_title: str, expected_year: Optional[str] = None,
) -> Optional[dict]:
    if not papers:
        return None
    best, best_score = None, -1.0
    for p in papers:
        score = word_overlap_similarity(query_title, p.get("title", ""))
        yr = p.get("year")
        if expected_year and yr:
            if str(yr) == expected_year:
                score += 20
            elif abs(yr - int(expected_year)) == 1:
                score += 5
        if score > best_score:
            best_score = score
            best = p
    return best


def _ss_to_match(paper: dict, query_title: str) -> MatchResult:
    """Normalise a raw Semantic Scholar paper into :class:`MatchResult`."""
    result_title = paper.get("title", "")
    authors = [a.get("name", "") for a in paper.get("authors", [])]
    year = str(paper.get("year") or "")
    venue = paper.get("venue", "")
    doi = (paper.get("externalIds") or {}).get("DOI")
    url = paper.get("url") or f"https://www.semanticscholar.org/paper/{paper.get('paperId', '')}"
    sim = title_similarity(query_title, result_title)
    return MatchResult(
        found=True,
        title=result_title, authors=", ".join(authors), year=year, journal=venue,
        doi=doi, url=url,
        source="SemanticScholar",
        title_similarity=sim,
        confidence=min(95.0, sim + 10),
        raw_response=paper,
    )


# ---------------------------------------------------------------------------
# OpenAlex
# ---------------------------------------------------------------------------

async def _query_openalex(
    title: str, *, max_results: int = 5, timeout: float = 15.0,
) -> list[dict]:
    """Search OpenAlex for works matching *title*."""
    url = "https://api.openalex.org/works"
    params = {"filter": f"title.search:{title}", "per_page": max_results}
    mailto = os.getenv("OPENALEX_MAILTO", "contact@example.com")
    headers = {
        "Accept": "application/json",
        "User-Agent": f"citecheck/0.1 (mailto:{mailto})",
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(url, params=params, headers=headers)
        resp.raise_for_status()
        return resp.json().get("results", [])


def _best_openalex(
    works: list[dict], query_title: str, expected_year: Optional[str] = None,
) -> Optional[dict]:
    if not works:
        return None
    best, best_score = None, -1.0
    for w in works:
        score = word_overlap_similarity(query_title, w.get("title", ""))
        yr = w.get("publication_year")
        if expected_year and yr:
            if str(yr) == expected_year:
                score += 20
            elif abs(yr - int(expected_year)) == 1:
                score += 5
        if score > best_score:
            best_score = score
            best = w
    return best


def _oa_to_match(work: dict, query_title: str) -> MatchResult:
    """Normalise a raw OpenAlex work into :class:`MatchResult`."""
    result_title = work.get("title", "")
    authors = [
        a.get("author", {}).get("display_name", "")
        for a in work.get("authorships", [])
    ]
    year = str(work.get("publication_year") or "")
    journal = ""
    if work.get("primary_location"):
        journal = (work["primary_location"].get("source") or {}).get("display_name", "")
    doi = work.get("doi")
    url = doi or work.get("id", "")
    sim = title_similarity(query_title, result_title)
    return MatchResult(
        found=True,
        title=result_title, authors=", ".join(authors), year=year, journal=journal,
        doi=doi.replace("https://doi.org/", "") if doi else None, url=url,
        source="OpenAlex",
        title_similarity=sim,
        confidence=min(90.0, sim + 5),
        raw_response=work,
    )


# ---------------------------------------------------------------------------
# arXiv (direct ID lookup)
# ---------------------------------------------------------------------------

async def _query_arxiv(
    arxiv_id: str, *, timeout: float = 30.0,
) -> Optional[dict]:
    """Look up a single paper by arXiv ID via the official Atom API.

    Unlike the title-search APIs above this is a *direct* ID lookup,
    not a search.  Returns ``None`` on error or unknown ID.
    """
    url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}&max_results=1"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(
                url, headers={"User-Agent": "citecheck/0.1"},
            )
            resp.raise_for_status()
            xml_text = resp.text

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(xml_text)
        entry = root.find("atom:entry", ns)
        if entry is None:
            return None

        entry_title = (entry.findtext("atom:title", "", ns) or "").strip()
        if entry_title.lower() == "error":
            return None

        summary = re.sub(r"\s+", " ", (entry.findtext("atom:summary", "", ns) or "").strip())
        if not summary:
            return None

        author_els = entry.findall("atom:author/atom:name", ns)
        authors = [el.text.strip() for el in author_els if el.text]

        published = (entry.findtext("atom:published", "", ns) or "")[:10]
        year = published[:4] if len(published) >= 4 else ""

        return {
            "id": arxiv_id,
            "title": re.sub(r"\s+", " ", entry_title),
            "authors": authors,
            "summary": summary,
            "year": year,
            "published": published,
        }
    except Exception:
        return None


def _arxiv_to_match(paper: dict, query_title: str) -> MatchResult:
    """Normalise an arXiv paper dict into :class:`MatchResult`.

    Confidence is set to ``max(title_similarity, 90)`` -- direct ID
    matches carry inherent confidence even if the title was paraphrased.
    """
    result_title = paper.get("title", "")
    authors = paper.get("authors", [])
    year = paper.get("year", "")
    sim = title_similarity(query_title, result_title)
    return MatchResult(
        found=True,
        title=result_title, authors=", ".join(authors), year=year, journal="arXiv",
        doi=None, url=f"https://arxiv.org/abs/{paper.get('id', '')}",
        source="arXiv",
        title_similarity=sim,
        confidence=max(sim, 90.0),
        raw_response=paper,
    )


# ---------------------------------------------------------------------------
# Web search (OpenAI Responses + web_search_preview)
# ---------------------------------------------------------------------------

def _parse_json_response(text: str) -> Optional[dict]:
    """Parse a JSON object from LLM output, with a regex fallback."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


async def _query_web_search(
    citation: Citation,
    *,
    model: str = "gpt-5.4",
    max_candidates: int = 3,
) -> List[dict]:
    """Use an LLM with web search to find publications matching *citation*.

    Calls OpenAI's Responses API with the ``web_search_preview`` tool.
    Returns a list of structured candidate dicts with keys ``title``,
    ``authors``, ``year``, ``journal``, ``doi``, ``url``, and
    ``match_explanation``, or an empty list if the call fails or finds
    nothing.

    The retrieval stage always uses OpenAI internally regardless of
    which provider is configured for the verifier LLM.
    """
    try:
        from openai import OpenAI
    except ImportError:
        return []

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("CSG_OPENAI_API_KEY")
    if not api_key:
        return []

    user_msg = WEB_SEARCH_USER.format(
        title=citation.title or "(unknown)",
        authors=citation.authors or "(unknown)",
        year=str(citation.year) if citation.year else "(unknown)",
        url=citation.url or "(no URL available)",
        raw_text=citation.raw_text,
        max_candidates=max_candidates,
    )
    try:
        client = OpenAI(api_key=api_key)
        response = await asyncio.to_thread(
            client.responses.create,
            model=model,
            instructions=WEB_SEARCH_SYSTEM,
            input=user_msg,
            tools=[{"type": "web_search_preview"}],
        )
        parsed = _parse_json_response(response.output_text)
        if parsed and "candidates" in parsed:
            return parsed["candidates"][:max_candidates]
    except Exception as exc:
        print(f"[WebSearch] LLM call failed: {exc}", flush=True)
    return []


def _web_candidates_to_matches(
    candidates: List[dict], query_title: str,
) -> List[MatchResult]:
    """Turn LLM-returned web-search candidates into :class:`MatchResult` objects.

    Each candidate dict is expected to have keys ``title``, ``authors``,
    ``year``, ``journal``, ``doi``, ``url`` (any may be ``None``).
    Results are sorted by title similarity, highest first.
    """
    results: List[MatchResult] = []
    for c in candidates:
        c_title = c.get("title") or ""
        sim = title_similarity(query_title, c_title) if c_title else 0.0
        results.append(MatchResult(
            found=True,
            title=c_title or None,
            authors=c.get("authors"),
            year=str(c["year"]) if c.get("year") else None,
            journal=c.get("journal"),
            doi=c.get("doi"),
            url=c.get("url"),
            source="WebSearch",
            title_similarity=sim,
            confidence=sim,
            raw_response=c,
        ))
    results.sort(key=lambda m: m.title_similarity, reverse=True)
    return results


# ---------------------------------------------------------------------------
# Cascade
# ---------------------------------------------------------------------------

async def find_closest_reference(
    citation: Citation,
    *,
    min_title_similarity: float = 55.0,
    fallback_confidence_threshold: float = 70.0,
    max_results_per_api: int = 5,
    api_timeout: float = 15.0,
    inter_api_delay: float = 0.5,
    try_arxiv: bool = False,
    arxiv_first: bool = False,
    try_web_search: bool = False,
    web_search_model: str = "gpt-5.4",
    web_search_candidates: int = 3,
    accept_best_web_search: bool = False,
    debug: bool = False,
) -> VerificationResult:
    """Run the 5-stage cascade and return a :class:`VerificationResult`.

    Acceptance rule per stage:
      * title similarity must be >= ``min_title_similarity``
      * for CrossRef (stage 1) confidence must additionally be
        >= ``fallback_confidence_threshold``
      * for fallback stages, the candidate's title similarity must
        strictly beat any earlier accepted CrossRef match

    If a stage accepts, the cascade returns immediately with that match
    as ``chosen``.  If no stage accepts, the cascade returns the best
    *partial* match (CrossRef if it cleared the title bar, or the best
    web-search candidate when ``accept_best_web_search`` is True), or
    ``NOT_FOUND``.

    Parameters
    ----------
    citation
        Parsed citation.  ``raw_text`` must be set; ``title``, ``year``,
        and ``authors`` improve matching when available.
    min_title_similarity
        Minimum 0-100 title similarity required to accept a match.
    fallback_confidence_threshold
        Minimum 0-100 CrossRef confidence required to skip the fallback
        stages.
    max_results_per_api
        Number of candidate results to request from each API.
    api_timeout
        HTTP request timeout per API call (seconds).
    inter_api_delay
        Sleep between successive stages (seconds), to respect rate limits.
    try_arxiv
        Enable the arXiv direct-ID-lookup fallback when the citation has
        an arXiv ID.
    arxiv_first
        Promote the arXiv lookup to stage 0 (before CrossRef).
    try_web_search
        Enable LLM-powered web search as a last resort.
    web_search_model
        OpenAI model used for the web-search stage (this stage always
        uses OpenAI regardless of the verifier LLM provider).
    web_search_candidates
        Number of candidates the web-search LLM should return.
    accept_best_web_search
        If True and the cascade would otherwise return ``NotFound``,
        accept the best web-search candidate regardless of its title
        similarity score.
    debug
        Drop a ``breakpoint()`` after each stage for inspection.
    """
    query_title = clean_query(citation.title or "")
    query_text = clean_query(citation.raw_text or "")
    search_query = query_title if query_title else query_text
    expected_year = str(citation.year) if citation.year else None

    vr = VerificationResult()

    if not search_query:
        return vr

    # ── 0. arXiv first (optional) ────────────────────────────────────────
    if arxiv_first and citation.arxiv_id:
        _t0 = time.perf_counter()
        _accept = False
        try:
            arxiv_paper = await _query_arxiv(
                citation.arxiv_id, timeout=api_timeout,
            )
            if arxiv_paper:
                arxiv_match = _arxiv_to_match(arxiv_paper, search_query)
                vr.arxiv = arxiv_match
                if arxiv_match.title_similarity >= min_title_similarity:
                    vr.chosen = arxiv_match
                    _accept = True
            else:
                vr.arxiv = NOT_FOUND
        except Exception:
            vr.arxiv = NOT_FOUND
        vr.api_timings["arxiv"] = round(time.perf_counter() - _t0, 4)

        if debug:
            breakpoint()

        if _accept:
            return vr
        await asyncio.sleep(inter_api_delay)

    # ── 1. CrossRef ──────────────────────────────────────────────────────
    cr_match: Optional[MatchResult] = None
    _t0 = time.perf_counter()
    _accept = False
    try:
        cr_items = await _query_crossref(
            search_query, max_results=max_results_per_api, timeout=api_timeout,
        )
        best_cr = _best_crossref(cr_items, search_query, expected_year)
        if best_cr:
            cr_match = _crossref_to_match(best_cr, search_query)
            vr.crossref = cr_match
            if (
                cr_match.title_similarity >= min_title_similarity
                and cr_match.confidence >= fallback_confidence_threshold
            ):
                vr.chosen = cr_match
                _accept = True
        else:
            vr.crossref = NOT_FOUND
    except Exception:
        vr.crossref = NOT_FOUND
    vr.api_timings["crossref"] = round(time.perf_counter() - _t0, 4)

    if debug:
        breakpoint()

    if _accept:
        return vr
    await asyncio.sleep(inter_api_delay)

    # ── 2. Semantic Scholar ──────────────────────────────────────────────
    _t0 = time.perf_counter()
    _accept = False
    try:
        ss_papers = await _query_semantic_scholar(
            search_query, max_results=max_results_per_api, timeout=api_timeout,
        )
        best_ss = _best_semantic_scholar(ss_papers, search_query, expected_year)
        if best_ss:
            ss_match = _ss_to_match(best_ss, search_query)
            vr.semantic_scholar = ss_match
            if ss_match.title_similarity >= min_title_similarity:
                cr_sim = (
                    cr_match.title_similarity
                    if cr_match and cr_match.title_similarity >= min_title_similarity
                    else -1
                )
                if ss_match.title_similarity > cr_sim:
                    vr.chosen = ss_match
                    _accept = True
        else:
            vr.semantic_scholar = NOT_FOUND
    except Exception:
        vr.semantic_scholar = NOT_FOUND
    vr.api_timings["semantic_scholar"] = round(time.perf_counter() - _t0, 4)

    if debug:
        breakpoint()

    if _accept:
        return vr
    await asyncio.sleep(inter_api_delay)

    # ── 3. OpenAlex ──────────────────────────────────────────────────────
    _t0 = time.perf_counter()
    _accept = False
    try:
        oa_works = await _query_openalex(
            search_query, max_results=max_results_per_api, timeout=api_timeout,
        )
        best_oa = _best_openalex(oa_works, search_query, expected_year)
        if best_oa:
            oa_match = _oa_to_match(best_oa, search_query)
            vr.openalex = oa_match
            if oa_match.title_similarity >= min_title_similarity:
                cr_sim = (
                    cr_match.title_similarity
                    if cr_match and cr_match.title_similarity >= min_title_similarity
                    else -1
                )
                if oa_match.title_similarity > cr_sim:
                    vr.chosen = oa_match
                    _accept = True
        else:
            vr.openalex = NOT_FOUND
    except Exception:
        vr.openalex = NOT_FOUND
    vr.api_timings["openalex"] = round(time.perf_counter() - _t0, 4)

    if debug:
        breakpoint()

    if _accept:
        return vr

    # ── 4. arXiv direct ID lookup ────────────────────────────────────────
    if try_arxiv and not arxiv_first and citation.arxiv_id:
        await asyncio.sleep(inter_api_delay)
        _t0 = time.perf_counter()
        _accept = False
        try:
            arxiv_paper = await _query_arxiv(
                citation.arxiv_id, timeout=api_timeout,
            )
            if arxiv_paper:
                arxiv_match = _arxiv_to_match(arxiv_paper, search_query)
                vr.arxiv = arxiv_match
                if arxiv_match.title_similarity >= min_title_similarity:
                    cr_sim = (
                        cr_match.title_similarity
                        if cr_match and cr_match.title_similarity >= min_title_similarity
                        else -1
                    )
                    if arxiv_match.title_similarity > cr_sim:
                        vr.chosen = arxiv_match
                        _accept = True
            else:
                vr.arxiv = NOT_FOUND
        except Exception:
            vr.arxiv = NOT_FOUND
        vr.api_timings["arxiv"] = round(time.perf_counter() - _t0, 4)

        if debug:
            breakpoint()

        if _accept:
            return vr

    # ── 5. Web search (LLM, last resort) ─────────────────────────────────
    if try_web_search:
        await asyncio.sleep(inter_api_delay)
        _t0 = time.perf_counter()
        _accept = False
        try:
            ws_candidates = await _query_web_search(
                citation,
                model=web_search_model,
                max_candidates=web_search_candidates,
            )
            if ws_candidates:
                ws_matches = _web_candidates_to_matches(ws_candidates, search_query)
                vr.web_search_all = ws_matches
                best_ws = ws_matches[0] if ws_matches else None
                if best_ws:
                    vr.web_search = best_ws
                    if best_ws.title_similarity >= min_title_similarity:
                        cr_sim = (
                            cr_match.title_similarity
                            if cr_match and cr_match.title_similarity >= min_title_similarity
                            else -1
                        )
                        if best_ws.title_similarity > cr_sim:
                            vr.chosen = best_ws
                            _accept = True
            else:
                vr.web_search = NOT_FOUND
                vr.web_search_all = []
        except Exception:
            vr.web_search = NOT_FOUND
            vr.web_search_all = []
        vr.api_timings["web_search"] = round(time.perf_counter() - _t0, 4)

        if debug:
            breakpoint()

        if _accept:
            return vr

    # ── 6. Best-partial fallback ─────────────────────────────────────────
    if cr_match and cr_match.title_similarity >= min_title_similarity:
        vr.chosen = cr_match
    elif accept_best_web_search and vr.web_search and vr.web_search.found:
        vr.chosen = vr.web_search

    return vr
