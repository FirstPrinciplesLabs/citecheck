#!/usr/bin/env bash
#
# CiteCheck full-evaluation wrapper.
#
# Mirrors the structure of the original SLURM evaluation job used to
# produce the paper's headline numbers, but invokes a single
# `citecheck-evaluate` call.  Every flag below has a default in
# `citecheck.config` that matches the paper, so this script is purely
# illustrative — most invocations can drop most of these flags.
#
# Usage
# -----
#     # Defaults: Anthropic / claude-sonnet-4-6, save under results/claude
#     bash examples/evaluate.sh
#
#     # Pick a different model / output dir
#     MODEL=claude-opus-4-1 SAVE_DIR=results/claude_opus bash examples/evaluate.sh
#
#     # Switch providers (override PROVIDER + MODEL together)
#     PROVIDER=openai    MODEL=gpt-5.4               bash examples/evaluate.sh
#     PROVIDER=google    MODEL=gemini-2.5-pro        bash examples/evaluate.sh
#
#     # Reuse a previously-computed dev/test split (so detector +
#     # baselines are evaluated on the same test set)
#     SPLITS_PATH=results/claude/splits.json bash examples/evaluate.sh
#
# To wrap this in SLURM, prepend the usual headers
# (#SBATCH --job-name=citecheck-eval, --time=…, --mem=…, etc.) and
# call this script in the body.
#
# All required keys live in .env (copy .env.example -> .env first).

set -euo pipefail

# ── Tunable parameters (override via environment) ──────────────────
PROVIDER="${PROVIDER:-anthropic}"
MODEL="${MODEL:-claude-sonnet-4-6}"
SAVE_DIR="${SAVE_DIR:-results/${PROVIDER}}"
SPLITS_PATH="${SPLITS_PATH:-}"
SEED="${SEED:-42}"

EXACT_THRESHOLD="${EXACT_THRESHOLD:-7.25}"
MINOR_THRESHOLD="${MINOR_THRESHOLD:-1.25}"
WEB_SEARCH_MODEL="${WEB_SEARCH_MODEL:-gpt-5.4}"

# ── Build the argument list ────────────────────────────────────────
ARGS=(
    --save-dir         "$SAVE_DIR"
    --seed             "$SEED"
    --provider         "$PROVIDER"
    --model            "$MODEL"
    --review-model     "$MODEL"
    --llm-parse-model  "$MODEL"
    --web-search-model "$WEB_SEARCH_MODEL"
    --exact-threshold  "$EXACT_THRESHOLD"
    --minor-threshold  "$MINOR_THRESHOLD"
)

# Optional: share a pre-computed dev/test split with the baselines.
if [[ -n "${SPLITS_PATH}" ]]; then
    ARGS+=( --splits-path "$SPLITS_PATH" )
fi

# Note: --no-arxiv / --no-web-search / --no-accept-best-web-search /
# --no-review / --no-llm-parse / --quiet are all available if you want
# to *disable* the corresponding stage; the defaults match the paper's
# headline configuration.

echo ">>> citecheck-evaluate ${ARGS[*]}"
citecheck-evaluate "${ARGS[@]}"
