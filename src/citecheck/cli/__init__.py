"""Command-line entry points.

Exposes three console scripts (wired up in ``pyproject.toml``):

- ``citecheck-evaluate`` -- dev/test evaluation harness with
  threshold tuning.
- ``citecheck-baseline`` -- run one LLM-only baseline configuration.
- ``citecheck-compare``  -- run a YAML matrix of baselines and
  aggregate them into a comparison table (optionally including the
  main detector's row).
"""
