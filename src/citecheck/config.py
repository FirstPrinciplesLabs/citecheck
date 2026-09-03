"""Package-wide defaults.

Every constant here matches the paper's headline run so a bare
``citecheck-evaluate`` reproduces the numbers reported in the paper.

Override at the call site (CLI flag, YAML config, or function kwarg) to
explore alternative configurations.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Classification thresholds (applied to the verifier's 0-10 score)
# ---------------------------------------------------------------------------
# Visual:
#
#   0 ------- 1.25 ------- 7.25 ------- 10
#       major     minor       exact

EXACT_THRESHOLD: float = 7.25
MINOR_THRESHOLD: float = 1.25


# ---------------------------------------------------------------------------
# LLM defaults
# ---------------------------------------------------------------------------

LLM_PROVIDER: str = "anthropic"
LLM_MODEL: str = "claude-sonnet-4-6"
LLM_TEMPERATURE: float = 0.0

# Reviewer LLM. ``None`` means "same as ``LLM_MODEL``".
REVIEW_ENABLED: bool = True
REVIEW_MODEL: str | None = None
REVIEW_SIM_THRESHOLD: float = 50.0

# Citation-parser fallback LLM. ``None`` means "same as ``LLM_MODEL``".
LLM_PARSE_ENABLED: bool = True
LLM_PARSE_MODEL: str | None = None


# ---------------------------------------------------------------------------
# External-API cascade
# ---------------------------------------------------------------------------

TRY_ARXIV: bool = True
TRY_WEB_SEARCH: bool = True
ACCEPT_BEST_WEB_SEARCH: bool = True

# The web-search retrieval stage uses OpenAI's ``web_search_preview``
# tool internally regardless of ``LLM_PROVIDER``.
WEB_SEARCH_MODEL: str = "gpt-5.4"
WEB_SEARCH_CANDIDATES: int = 3

MIN_TITLE_SIMILARITY: float = 55.0
FALLBACK_CONFIDENCE_THRESHOLD: float = 70.0
API_TIMEOUT: float = 15.0
INTER_API_DELAY: float = 0.5


# ---------------------------------------------------------------------------
# Evaluation / splits
# ---------------------------------------------------------------------------

DEFAULT_SEED: int = 42

# Where the canonical dataset lives, relative to the repo root.
DEFAULT_DATASET_PATH: Path = (
    Path(__file__).resolve().parents[2] / "dataset" / "corruption_metadata.json"
)
