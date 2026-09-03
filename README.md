# CiteCheck

**Retrieval-grounded detection of LLM citation hallucinations in
scientific text.**

CiteCheck verifies whether a citation produced by an LLM (a) refers
to a real publication and (b) describes that publication's metadata
accurately. The pipeline retrieves candidate publications from
external scholarly sources (CrossRef, Semantic Scholar, OpenAlex,
arXiv) with an open-web fallback, compares each candidate against
the citation using a structured-output LLM verifier, optionally
invokes a second-pass reviewer LLM on weak matches, and maps the
verifier score to one of three discrete labels:

| label | meaning |
|---|---|
| `exact_match` | the citation correctly identifies a real paper, with at most trivial formatting differences |
| `minor_hallucination` | the citation refers to a recognisable real paper but has small metadata errors (year, initials, partial title, …) |
| `major_hallucination` | the citation describes a different paper from what its identifiers point to, or the paper does not exist |

> Companion artefact for the paper *CiteCheck: Retrieval-Grounded
> Detection of LLM Citation Hallucinations in Scientific Text*. The
> default configuration of `citecheck-evaluate` reproduces the
> headline numbers from the paper's main table.

## Pipeline

```
                      ┌────────────────────────────────┐
                      │ 1. Citation parser             │
   citation string ─▶ │   regex + LLM fallback         │
                      └──────────────┬─────────────────┘
                                     ▼
                      ┌────────────────────────────────┐
                      │ 2. External-API cascade        │
                      │    arXiv → CrossRef →          │
                      │    Semantic Scholar → OpenAlex │
                      │    → open web                  │
                      └──────────────┬─────────────────┘
                                     ▼
                      ┌────────────────────────────────┐
                      │ 3. LLM verifier (structured)   │
                      │    score 0–10 + label          │
                      └──────────────┬─────────────────┘
                                     ▼
                      ┌────────────────────────────────┐
                      │ 4. Reviewer LLM (optional)     │
                      │    overrides on weak matches   │
                      └──────────────┬─────────────────┘
                                     ▼
                      ┌────────────────────────────────┐
                      │ 5. Threshold classifier        │
                      │    score → discrete label      │
                      └────────────────────────────────┘
```

The reviewer is triggered only when the cascade returns an
identifier-based match whose title similarity is below
`review_sim_threshold`; otherwise the verifier's label is used as-is.

## Repository layout

```
citecheck/
├── README.md
├── pyproject.toml
├── .env.example                       API key template
├── configs/                           sample YAML configs
│   ├── evaluate-claude.yaml             paper-default detector run
│   ├── baseline-claude.yaml             one LLM-only baseline
│   └── baseline-matrix.yaml             comparison matrix
├── examples/
│   └── evaluate.sh                    batch wrapper around citecheck-evaluate
├── dataset/
│   ├── corruption_metadata.json       42 collections, 982 citations
│   └── SCHEMA.md                      schema reference + provenance
└── src/citecheck/
    ├── config.py                      package-wide defaults
    ├── core/                          citation parser, similarity,
    │                                  external-API cascade, classifier,
    │                                  detector orchestrator
    ├── llm/                           structured verifier + reviewer
    ├── baselines/                     LLM-only comparison harness
    ├── eval/                          dataset, ground-truth, metrics,
    │                                  splits, threshold tuning, harness
    ├── cli/                           three console scripts
    └── monitoring.py                  optional resource monitor
```

## Installation

CiteCheck targets Python 3.10+.

### Option A — pip install (recommended)

```bash
pip install -e .
# optional extras
pip install -e ".[monitoring,plot]"
```

This pulls in the third-party dependencies and registers the
`citecheck-evaluate`, `citecheck-baseline`, and `citecheck-compare`
console scripts on your `$PATH`.

### Option B — no install, run as a module

If you'd rather not install the package (e.g. you just want to run a
script after copying the directory somewhere), install the
dependencies directly and put `src/` on `PYTHONPATH`:

```bash
pip install httpx openai anthropic \
            langchain-core langchain-openai langchain-anthropic \
            langchain-google-genai google-genai \
            pydantic numpy scikit-learn pyyaml python-dotenv tqdm

# from the package root (the directory containing pyproject.toml + src/)
export PYTHONPATH="$PWD/src"

python -m citecheck.cli.evaluate         --save-dir results/claude
python -m citecheck.cli.baseline_run     --config configs/baseline-claude.yaml --save-dir results/baselines/zero
python -m citecheck.cli.baseline_compare --config configs/baseline-matrix.yaml --save-dir results/compare
```

`PYTHONPATH` must point to the directory **that contains** the
`citecheck/` package folder (i.e. `src/`, not the outer dir). Verify
with `python -c "import citecheck; print(citecheck.__file__)"`.

Direct file execution (`python src/citecheck/cli/evaluate.py …`) does
**not** work — the CLI modules use relative imports (`from .. import
config`), which Python only resolves when the file is launched via
`-m` as part of a package.

### API keys

Either way, set up the API keys for the LLM provider(s) you intend
to use:

```bash
cp .env.example .env
# edit .env and fill in OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY
```

`python-dotenv` loads the `.env` file automatically the first time
the package is imported.

## Quickstart

The package ships three console scripts, all driven by the bundled
benchmark dataset under `dataset/corruption_metadata.json`. Every
flag has a sensible default that matches the paper's headline run,
so the simplest invocations are typically enough.

### 1. Reproduce the paper's evaluation

```bash
citecheck-evaluate --save-dir results/claude
```

Builds a deterministic dev/test split from the bundled dataset, runs
the detector on both, tunes `(minor, exact)` thresholds on dev, and
evaluates the test set at both default (paper) and tuned thresholds.
Drops every artefact under `results/claude/`:

```
results/claude/
├── splits.json
├── dev_predictions.json
├── test_predictions.json
├── threshold_analysis.json
├── dev_evaluation.json
├── test_evaluation.json            ← headline metrics live here
└── metrics_summary.txt
```

Use `--config configs/evaluate-claude.yaml` to load every parameter
from a YAML file (CLI flags still override).

To swap providers or models without remembering every flag, use the
batch wrapper at `examples/evaluate.sh`. It mirrors the structure of
the original SLURM `evaluate.sh` job, exposes the common knobs as
environment variables, and just calls `citecheck-evaluate` underneath:

```bash
# Defaults: Anthropic / Claude, save under results/anthropic
bash examples/evaluate.sh

# Switch provider + model
PROVIDER=openai MODEL=gpt-5.4 SAVE_DIR=results/gpt5 bash examples/evaluate.sh
PROVIDER=google MODEL=gemini-2.5-pro SAVE_DIR=results/gemini bash examples/evaluate.sh
```

### 2. Run one LLM-only baseline

```bash
citecheck-baseline \
    --provider anthropic --model claude-sonnet-4-6 \
    --prompting fewshot --web-search \
    --save-dir results/baselines/claude_few_ws
```

Or, equivalently, drive everything from a YAML file:

```bash
citecheck-baseline --config configs/baseline-claude.yaml \
                   --save-dir results/baselines/claude_zero_nows
```

### 3. Run a comparison matrix

```bash
citecheck-compare --config configs/baseline-matrix.yaml
```

The matrix YAML lists multiple `(provider × model × prompting ×
web-search)` rows under a single `runs:` key. The CLI executes each,
optionally pulls in a row from a previous `citecheck-evaluate` run
(via `include_main:` in the YAML), and writes:

```
results/compare/
├── <run_name>/                      per-run baseline artefacts
│   ├── config.json
│   ├── predictions.json
│   ├── metrics_summary.txt
│   ├── cost_summary.json
│   └── evaluation_results.json
├── comparison.json                  aggregated rows + raw cost numbers
└── comparison_table.txt             human-readable comparison
```

Use `--max-citations 8` to smoke-test the whole matrix on a small
sample before launching the real run, and `--skip-existing` to resume
after a partial failure.

### 4. Zero-shot vs few-shot

Both the main detector and the LLM-only baselines support **zero-shot**
(default) and **few-shot** prompting. Few-shot exemplars are drawn
deterministically from the dev split — one labelled exemplar per
`(dev collection × class label)` cell — and the dev tuning set is
automatically de-duplicated so that no example appears as both an
exemplar and a graded item.

**Main detector — `citecheck-evaluate`** — toggle with the `--fewshot`
flag (or `fewshot: true` in the YAML):

```bash
# zero-shot (default)
citecheck-evaluate --save-dir results/claude_zero

# few-shot: appends the dev exemplar block to the verifier prompt and
# removes those exemplars from the dev tuning set
citecheck-evaluate --save-dir results/claude_few --fewshot
```

In few-shot mode, a `fewshot_block.txt` is written under `--save-dir`
so you can audit the exact prompt suffix used by the verifier, and
the prediction/evaluation files are suffixed `_fewshot` to keep the
two modes side by side.

**Baselines — `citecheck-baseline`** — toggle with `--prompting`:

```bash
# zero-shot (default)
citecheck-baseline --provider anthropic --model claude-sonnet-4-6 \
                   --prompting zeroshot \
                   --save-dir results/baselines/claude_zero_nows

# few-shot
citecheck-baseline --provider anthropic --model claude-sonnet-4-6 \
                   --prompting fewshot \
                   --save-dir results/baselines/claude_few_nows
```

**Comparison matrix — `citecheck-compare`** — set `prompting:
zeroshot|fewshot` per row in the YAML. The shipped
`configs/baseline-matrix.yaml` already mixes both modes (see the
`*_zero_*` and `*_few_*` rows for each provider).

To make few-shot results comparable across the detector and the
baselines, share the same splits file (so they all draw exemplars
from the *same* dev partition):

```bash
citecheck-evaluate --save-dir results/claude_few --fewshot
citecheck-baseline --config configs/baseline-claude.yaml \
                   --prompting fewshot \
                   --splits-path results/claude_few/splits.json \
                   --save-dir results/baselines/claude_few_nows
```

## Configuration system

For every CLI, parameters are resolved with the precedence:

1. **CLI flags** (highest)
2. **YAML file** passed via `--config` *(optional)*
3. **Package defaults** in `citecheck.config` (which match the
   paper's `evaluate.sh`)

Sample configs live under `configs/`:

- `configs/evaluate-claude.yaml` — full detector configuration
  matching the paper's headline Anthropic / Claude run.
- `configs/baseline-claude.yaml` — single-config baseline (zero-shot
  Claude, no web search).
- `configs/baseline-matrix.yaml` — 9-row comparison matrix
  (3 providers × 3 prompting + web-search combinations).

To share the dev/test split across `citecheck-evaluate` and the
baselines (so the headline numbers and the LLM-only baseline numbers
are computed on **exactly** the same test set), point both at the
same `splits.json`:

```bash
citecheck-evaluate  --save-dir results/claude
citecheck-baseline  --config configs/baseline-claude.yaml \
                    --splits-path results/claude/splits.json \
                    --save-dir results/baselines/claude_zero_nows
```

## Programmatic API

The CLIs cover the common workflows; every layer underneath is also
importable for custom pipelines or notebook exploration.

### Run the full evaluation harness

```python
import citecheck

citecheck.evaluate_dataset(save_dir="results/claude")
```

### Run the detector on your own citation strings

```python
from citecheck import detect_citations
from citecheck.core import parse_citation

raw = [
    "[Vaswani et al., 2017, Attention Is All You Need](https://arxiv.org/abs/1706.03762)",
    "[Doe, J., 2023, A Paper That Does Not Exist](https://example.org/fake)",
]
citations = [parse_citation(i, t) for i, t in enumerate(raw, start=1)]
predictions = detect_citations(citations)

for p in predictions:
    print(p["predicted_label"], p["llm_score"], "—", p["citation_text"])
```

### Lower-level access

```python
from citecheck.core import find_closest_reference, classify_score
from citecheck.llm import build_structured_llm, verify_with_llm
from citecheck.eval import load_dataset, make_splits, make_fewshot_exemplars
from citecheck.baselines import (
    run_baseline_config,
    evaluate_baseline_predictions,
    parse_baseline_output,
)
```

## Dataset

The benchmark dataset ships with the package:

```
dataset/corruption_metadata.json   # 42 collections, 982 citations
dataset/SCHEMA.md                  # schema reference + provenance
```

The split is balanced across three classes (`valid`,
`hallucinated_minor`, `hallucinated_major`) and nine scientific
topics (astrophysics, biophysics, condensed matter, gravitational
physics, nuclear physics, particle physics, plasma physics, quantum
computing, soft matter physics).

`citecheck.load_dataset()` returns a typed `Dataset` object;
`citecheck.eval.make_splits(...)` produces a deterministic dev/test
partition (one dev collection per topic, the rest test); and
`citecheck.eval.make_fewshot_exemplars(...)` produces a 27-element
exemplar pool (one per topic × class) suitable for both the
verifier's few-shot block and the baseline harness.

## Citing

```bibtex
@article{khajavi2026citecheck,
  title   = {{CiteCheck}: Retrieval-Grounded Detection of {LLM} Citation Hallucinations in Scientific Text},
  author  = {Khajavi, Khashayar and Sadeghi, Shaghayegh and Adhikari, Rise and Tessier, Alexander},
  journal = {arXiv preprint arXiv:2605.27700},
  year    = {2026},
  url     = {https://arxiv.org/abs/2605.27700}
}
```

## License

The source code is released under the [MIT License](LICENSE). See
[dataset/SCHEMA.md](dataset/SCHEMA.md) for dataset licensing notes.

