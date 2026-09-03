"""LLM clients for the baseline harness.

This module wraps the three supported providers (``openai`` /
``anthropic`` / ``google``) behind a single dispatch function,
:func:`call_baseline_llm`, that returns rich per-call metrics
(:class:`BaselineCallResult` — token counts, latency, cost, search
calls, errors).

Why a fresh module instead of reusing
:mod:`citecheck.llm.client`?
-----------------------------------
The detector's :func:`~citecheck.llm.client.build_structured_llm`
factory wraps a chat model in LangChain's structured-output adapter
and deliberately throws away usage metadata.  For the baseline
harness we need the opposite shape: rich per-call metrics so we can
produce the cost-vs-accuracy Pareto plots that go in the paper.

Web search
----------
Each provider exposes web search through a different argument shape.
The dispatch keeps both code paths in one module so the call sites
(and the runner) only need to flip a flag::

    no_search = call_baseline_llm(..., web_search=False)
    with_search = call_baseline_llm(..., web_search=True, web_search_max_uses=5)

When ``web_search=True``:

- **OpenAI**: ``tools=[{"type": "web_search_preview"}]`` on the
  Responses API.  Search calls appear as ``web_search_call`` items in
  ``response.output``.  ``max_uses`` is **not** accepted by the
  Responses API and is silently ignored — cap usage in the prompt if
  needed.
- **Anthropic**: ``tools=[{"type": "web_search_20250305", "name":
  "web_search", "max_uses": N}]`` on the Messages API.  Search calls
  appear as ``server_tool_use`` blocks in ``response.content``.
- **Gemini**: ``tools=[Tool(google_search=GoogleSearch())]`` via the
  ``google-genai`` SDK.  Search queries are surfaced via
  ``response.candidates[0].grounding_metadata.web_search_queries``.

API keys
--------
``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY`` / ``GEMINI_API_KEY`` must
be set; ``CSG_OPENAI_API_KEY`` and ``GOOGLE_API_KEY`` are accepted as
alternate names for OpenAI and Gemini respectively.  ``python-dotenv`` is
used to load a local ``.env`` file when present.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


__all__ = (
    "BaselineCallResult",
    "SUPPORTED_PROVIDERS",
    "PRICING_USD_PER_MTOK",
    "SEARCH_PRICE_USD",
    "call_baseline_llm",
    "compute_token_cost",
)


SUPPORTED_PROVIDERS: Tuple[str, str, str] = ("openai", "anthropic", "google")


# ---------------------------------------------------------------------------
# Optional .env loading
# ---------------------------------------------------------------------------

try:                                         # pragma: no cover — env var convenience
    from dotenv import load_dotenv

    load_dotenv()
except Exception:                            # pragma: no cover
    pass


# ---------------------------------------------------------------------------
# Pricing tables
# ---------------------------------------------------------------------------
# Best-effort placeholders.  Override at the call site (or via a
# YAML config) if you have an exact contract price.  Lookup is
# longest-prefix against the model identifier so e.g. ``gpt-5.4-mini``
# falls back to ``gpt-5.4`` when there is no more specific entry.

PRICING_USD_PER_MTOK: Dict[str, Dict[str, float]] = {
    "gpt-5.4":          {"in": 10.00, "out": 30.00},
    "gpt-5.3":          {"in":  7.50, "out": 22.50},
    "gpt-5.2":          {"in":  5.00, "out": 15.00},
    "gpt-5":            {"in":  5.00, "out": 15.00},
    "gpt-4o-mini":      {"in":  0.15, "out":  0.60},
    "gpt-4o":           {"in":  2.50, "out": 10.00},
    "claude-sonnet-4":  {"in":  3.00, "out": 15.00},
    "claude-opus-4":    {"in": 15.00, "out": 75.00},
    "gemini-2.5-pro":   {"in":  1.25, "out": 10.00},
    "gemini-2.5-flash": {"in":  0.30, "out":  2.50},
    "_default":         {"in":  1.00, "out":  3.00},
}

SEARCH_PRICE_USD: Dict[str, float] = {
    "openai":    0.025,    # web_search_preview ≈ $25 / 1k calls
    "anthropic": 0.010,    # web_search ≈ $10 / 1k calls
    "google":    0.000,    # Search Grounding free under daily cap
}


def _price_for_model(model: str) -> Dict[str, float]:
    """Longest-prefix match against :data:`PRICING_USD_PER_MTOK`."""
    best_key: Optional[str] = None
    for key in PRICING_USD_PER_MTOK:
        if key == "_default":
            continue
        if model.startswith(key) and (best_key is None or len(key) > len(best_key)):
            best_key = key
    return PRICING_USD_PER_MTOK[best_key or "_default"]


def compute_token_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """Token-only USD cost for *prompt_tokens* / *completion_tokens* under *model*."""
    p = _price_for_model(model)
    return (
        prompt_tokens * p["in"] / 1_000_000.0
        + completion_tokens * p["out"] / 1_000_000.0
    )


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class BaselineCallResult:
    """Everything the runner needs to log for one LLM call."""

    text: str                     # raw assistant output
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_s: float
    cost_usd: float
    provider: str
    model: str
    attempts: int = 1             # how many tries before success
    error: Optional[str] = None   # populated only if all retries failed
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Per-provider helpers
# ---------------------------------------------------------------------------

def _supports_reasoning(model: str) -> bool:
    return any(model.startswith(p) for p in (
        "gpt-5.2", "gpt-5.3", "gpt-5.4", "gpt-5.5",
        "gpt-5.6", "gpt-5.7", "gpt-5.8", "gpt-5.9", "gpt-6",
    ))


def _call_openai(
    model: str,
    system: str,
    user: str,
    *,
    temperature: Optional[float],
    reasoning_effort: Optional[str],
    max_output_tokens: Optional[int],
    web_search: bool,
    openai_tool_type: str,
) -> Tuple[str, int, int, int]:
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("CSG_OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY (or CSG_OPENAI_API_KEY) not set")
    client = OpenAI(api_key=api_key)

    kwargs: Dict[str, Any] = {
        "model": model,
        "input": user,
        "instructions": system,
    }
    if max_output_tokens is not None:
        kwargs["max_output_tokens"] = max_output_tokens
    if web_search:
        kwargs["tools"] = [{"type": openai_tool_type}]

    use_reasoning = bool(reasoning_effort) and _supports_reasoning(model)
    if use_reasoning:
        kwargs["reasoning"] = {"effort": reasoning_effort}
    elif temperature is not None:
        kwargs["temperature"] = temperature

    resp = client.responses.create(**kwargs)

    text = resp.output_text or ""

    search_calls = 0
    if web_search:
        for item in getattr(resp, "output", []) or []:
            if "web_search" in getattr(item, "type", ""):
                search_calls += 1

    usage = getattr(resp, "usage", None)
    in_tok = int(getattr(usage, "input_tokens", 0) or 0)
    out_tok = int(getattr(usage, "output_tokens", 0) or 0)
    return text, in_tok, out_tok, search_calls


def _call_anthropic(
    model: str,
    system: str,
    user: str,
    *,
    temperature: Optional[float],
    max_output_tokens: Optional[int],
    web_search: bool,
    web_search_max_uses: Optional[int],
    anthropic_tool_type: str,
) -> Tuple[str, int, int, int]:
    from anthropic import Anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    client = Anthropic(api_key=api_key)

    kwargs: Dict[str, Any] = {
        "model": model,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "max_tokens": max_output_tokens or 1024,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if web_search:
        tool_obj: Dict[str, Any] = {
            "type": anthropic_tool_type,
            "name": "web_search",
        }
        if web_search_max_uses is not None:
            tool_obj["max_uses"] = web_search_max_uses
        kwargs["tools"] = [tool_obj]

    resp = client.messages.create(**kwargs)

    text_parts: list[str] = []
    search_calls = 0
    for block in resp.content or []:
        btype = getattr(block, "type", "")
        if btype == "text":
            btext = getattr(block, "text", None)
            if isinstance(btext, str):
                text_parts.append(btext)
        elif btype in ("server_tool_use", "tool_use"):
            name = getattr(block, "name", "")
            if name == "web_search" or "web_search" in btype:
                search_calls += 1
        elif "web_search" in btype:
            search_calls += 1
    text = "".join(text_parts)

    usage = getattr(resp, "usage", None)
    in_tok = int(getattr(usage, "input_tokens", 0) or 0)
    out_tok = int(getattr(usage, "output_tokens", 0) or 0)
    return text, in_tok, out_tok, search_calls


def _call_google(
    model: str,
    system: str,
    user: str,
    *,
    temperature: Optional[float],
    max_output_tokens: Optional[int],
    web_search: bool,
) -> Tuple[str, int, int, int]:
    from google import genai
    from google.genai.types import GenerateContentConfig

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) not set")
    client = genai.Client(api_key=api_key)

    config_kwargs: Dict[str, Any] = {"system_instruction": system}
    if temperature is not None:
        config_kwargs["temperature"] = temperature
    if max_output_tokens is not None:
        config_kwargs["max_output_tokens"] = max_output_tokens

    if web_search:
        from google.genai.types import GoogleSearch, Tool

        config_kwargs["tools"] = [Tool(google_search=GoogleSearch())]

    resp = client.models.generate_content(
        model=model,
        contents=user,
        config=GenerateContentConfig(**config_kwargs),
    )

    text = resp.text or ""

    search_calls = 0
    if web_search:
        for cand in getattr(resp, "candidates", None) or []:
            gm = getattr(cand, "grounding_metadata", None)
            if gm is None:
                continue
            queries = getattr(gm, "web_search_queries", None) or []
            search_calls += len(queries)

    usage = getattr(resp, "usage_metadata", None)
    in_tok = int(getattr(usage, "prompt_token_count", 0) or 0)
    out_tok = int(getattr(usage, "candidates_token_count", 0) or 0)
    return text, in_tok, out_tok, search_calls


# ---------------------------------------------------------------------------
# Retry-aware dispatch
# ---------------------------------------------------------------------------

_TRANSIENT_HINTS = (
    "timeout", "rate", "overload", "503", "502", "500", "429",
    "connection", "temporar", "unavailable",
)


def _is_transient(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(h in msg for h in _TRANSIENT_HINTS)


def call_baseline_llm(
    *,
    provider: str,
    model: str,
    system: str,
    user: str,
    web_search: bool = False,
    temperature: Optional[float] = 0.0,
    reasoning_effort: Optional[str] = None,
    max_output_tokens: Optional[int] = 1024,
    web_search_max_uses: Optional[int] = 5,
    max_retries: int = 3,
    base_backoff_s: float = 2.0,
    openai_tool_type: str = "web_search_preview",
    anthropic_tool_type: str = "web_search_20250305",
) -> BaselineCallResult:
    """Make a single classification call and return rich metrics.

    Parameters
    ----------
    provider
        ``"openai"`` | ``"anthropic"`` | ``"google"``.
    model
        Provider-specific model identifier (e.g. ``"gpt-5.4"``,
        ``"claude-sonnet-4-6"``, ``"gemini-2.5-flash"``).
    system, user
        Prompt content (system prompt + user message).
    web_search
        Enable the provider's native web-search tool.  When ``True``
        the per-search fee from :data:`SEARCH_PRICE_USD` is added to
        ``cost_usd`` and ``extra['web_search_calls']`` is populated.
    temperature
        Sampling temperature.  Set to ``None`` to omit (some providers
        don't accept it for certain modes).  For OpenAI reasoning
        models with ``reasoning_effort != None`` it is automatically
        dropped.
    reasoning_effort
        ``"none" | "low" | "medium" | "high"`` — only honoured for
        OpenAI gpt-5.2+ models.
    max_output_tokens
        Cap on completion length.  Defaults to 1024.
    web_search_max_uses
        Cap on tool-call count.  Honoured by Anthropic; ignored by
        OpenAI (Responses API doesn't accept ``max_uses``) and Gemini.
    max_retries
        Number of attempts on transient errors.  Default 3.
    base_backoff_s
        Base seconds for exponential backoff
        (``base * 2**(attempt-1)``).
    """
    provider = provider.lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported provider {provider!r}; choose from {SUPPORTED_PROVIDERS}"
        )

    last_error: Optional[str] = None
    attempt = 0
    t_start = time.perf_counter()

    while attempt < max_retries:
        attempt += 1
        try:
            if provider == "openai":
                text, in_tok, out_tok, n_search = _call_openai(
                    model, system, user,
                    temperature=temperature,
                    reasoning_effort=reasoning_effort,
                    max_output_tokens=max_output_tokens,
                    web_search=web_search,
                    openai_tool_type=openai_tool_type,
                )
            elif provider == "anthropic":
                text, in_tok, out_tok, n_search = _call_anthropic(
                    model, system, user,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    web_search=web_search,
                    web_search_max_uses=web_search_max_uses,
                    anthropic_tool_type=anthropic_tool_type,
                )
            else:
                text, in_tok, out_tok, n_search = _call_google(
                    model, system, user,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    web_search=web_search,
                )

            latency = time.perf_counter() - t_start
            token_cost = compute_token_cost(model, in_tok, out_tok)
            search_cost = (
                SEARCH_PRICE_USD.get(provider, 0.0) * n_search
                if web_search else 0.0
            )
            total_cost = token_cost + search_cost

            extra: Dict[str, Any] = {
                "web_search_enabled": web_search,
                "web_search_calls": int(n_search),
            }
            if web_search:
                extra["token_cost_usd"] = round(token_cost, 6)
                extra["search_cost_usd"] = round(search_cost, 6)

            return BaselineCallResult(
                text=text,
                prompt_tokens=in_tok,
                completion_tokens=out_tok,
                total_tokens=in_tok + out_tok,
                latency_s=round(latency, 4),
                cost_usd=round(total_cost, 6),
                provider=provider,
                model=model,
                attempts=attempt,
                extra=extra,
            )
        except Exception as exc:                     # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt >= max_retries or not _is_transient(exc):
                break
            time.sleep(base_backoff_s * (2 ** (attempt - 1)))

    latency = time.perf_counter() - t_start
    return BaselineCallResult(
        text="",
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        latency_s=round(latency, 4),
        cost_usd=0.0,
        provider=provider,
        model=model,
        attempts=attempt,
        error=last_error,
        extra={"web_search_enabled": web_search, "web_search_calls": 0},
    )
