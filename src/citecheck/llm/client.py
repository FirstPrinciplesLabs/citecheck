"""Multi-provider LangChain client factory.

A single :func:`build_structured_llm` produces a LangChain chat model
with structured (Pydantic) output for whichever provider you pick.
The three supported providers are ``openai``, ``anthropic``, and
``google``; each maps to the corresponding ``langchain-*`` package.

API-key fallback
----------------
At import time, if ``OPENAI_API_KEY`` isn't set but
``CSG_OPENAI_API_KEY`` is, the latter is aliased to the former.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Type

from dotenv import load_dotenv
from pydantic import BaseModel

from .. import config


# Load .env once at import time so callers don't have to.
load_dotenv()

if not os.getenv("OPENAI_API_KEY") and os.getenv("CSG_OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("CSG_OPENAI_API_KEY", "")


__all__ = ("build_structured_llm", "SUPPORTED_PROVIDERS")


SUPPORTED_PROVIDERS: tuple[str, ...] = ("openai", "anthropic", "google")


def build_structured_llm(
    schema: Type[BaseModel],
    *,
    provider: str = config.LLM_PROVIDER,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> Any:
    """Construct a LangChain chat model bound to *schema* for structured output.

    Parameters
    ----------
    schema
        A Pydantic model class.  The chain's ``.invoke()`` returns an
        instance of this class.
    provider
        One of ``"openai"``, ``"anthropic"``, ``"google"``.
    model
        Model name (e.g. ``"claude-sonnet-4-6"``, ``"gpt-5.4"``,
        ``"gemini-2.5-flash"``).  Defaults to
        :data:`config.LLM_MODEL`.
    temperature
        Sampling temperature.  Defaults to :data:`config.LLM_TEMPERATURE`.

    Returns
    -------
    A chat model wrapped with ``with_structured_output(schema)``.
    """
    _model = model or config.LLM_MODEL
    _temp = temperature if temperature is not None else config.LLM_TEMPERATURE

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=_model, temperature=_temp,
        ).with_structured_output(schema)

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=_model, temperature=_temp,
        ).with_structured_output(schema)

    if provider != "openai":
        raise ValueError(
            f"Unknown provider {provider!r}; expected one of {SUPPORTED_PROVIDERS}"
        )

    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=_model, temperature=_temp,
    ).with_structured_output(schema)
