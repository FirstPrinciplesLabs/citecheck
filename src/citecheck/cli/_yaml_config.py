"""Tiny helper for loading YAML configs used by ``citecheck-evaluate`` and
``citecheck-compare``.

Resolution order
----------------

For every option, the value is taken from:

1. The command-line flag if explicitly provided.
2. The YAML file's ``key`` (if loaded).
3. The package default (from :mod:`citecheck.config`).

Unknown keys in the YAML are reported as warnings, not errors -- this
keeps configs forward-compatible across minor version bumps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional


__all__ = ("load_yaml_config", "merge_config")


def load_yaml_config(path: Optional[Path]) -> Dict[str, Any]:
    """Load *path* into a dict, or return ``{}`` if *path* is ``None``."""
    if path is None:
        return {}
    import yaml  # type: ignore[import-untyped]
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"YAML config at {path} must be a mapping at the top level, "
            f"got {type(raw).__name__}"
        )
    return raw


def merge_config(
    *,
    defaults: Mapping[str, Any],
    yaml_cfg: Mapping[str, Any],
    cli_overrides: Mapping[str, Any],
    valid_keys: Optional[set[str]] = None,
) -> Dict[str, Any]:
    """Merge defaults with YAML and CLI overrides.

    - Values are taken from *cli_overrides* first (when not ``None``),
      then from *yaml_cfg*, then from *defaults*.
    - If *valid_keys* is provided, any key in *yaml_cfg* not in the set
      raises a :class:`ValueError`.  This catches typos in user
      configs without preventing the CLI from forwarding extra
      kwargs.
    """
    if valid_keys is not None:
        unknown = set(yaml_cfg) - set(valid_keys)
        if unknown:
            raise ValueError(
                f"Unknown YAML config keys: {sorted(unknown)}. "
                f"Valid keys: {sorted(valid_keys)}"
            )

    merged: Dict[str, Any] = dict(defaults)
    merged.update({k: v for k, v in yaml_cfg.items() if v is not None})
    merged.update({k: v for k, v in cli_overrides.items() if v is not None})
    return merged
