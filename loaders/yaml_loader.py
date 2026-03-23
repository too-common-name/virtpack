"""YAML configuration loaders (HLD §3.2 – §3.4).

Each loader reads a YAML file, parses it with ``yaml.safe_load``, and
validates the result through the corresponding Pydantic model.

Design decisions
----------------
* **``PlanConfig``** has sensible defaults for every field, so an empty
  or missing file is valid — the loader returns defaults.
* **``InventoryConfig``** allows an empty profile list (pure greenfield),
  so a missing file is also valid.
* **``CatalogConfig``** requires at least one profile when provided, but
  the file itself is optional — ``None`` means inventory-only mode
  (pure brownfield, no greenfield expansion).
* All loaders raise clear, actionable errors on I/O or validation failure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from pathlib import Path

    from models.config import CatalogConfig, InventoryConfig, PlanConfig


class ConfigLoadError(Exception):
    """Raised when a YAML config file cannot be loaded or validated.

    Attributes
    ----------
    path : Path
        The file that caused the error.
    reason : str
        Human-readable explanation.
    """

    def __init__(self, path: Path, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


# ═══════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════


def _read_yaml(path: Path) -> dict[str, object]:
    """Read and parse a YAML file, returning the top-level mapping.

    Returns an empty dict if the file is empty or contains only
    ``null`` / ``---``.

    Raises
    ------
    ConfigLoadError
        If the file doesn't exist, isn't valid YAML, or the top-level
        element is not a mapping (dict).
    """
    if not path.exists():
        raise ConfigLoadError(path, "file not found")

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigLoadError(path, f"cannot read file: {exc}") from exc

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigLoadError(path, f"invalid YAML: {exc}") from exc

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ConfigLoadError(
            path,
            f"expected a YAML mapping at top level, got {type(data).__name__}",
        )

    return data


# ═══════════════════════════════════════════════════════════════════════
# Public loaders
# ═══════════════════════════════════════════════════════════════════════


def load_plan_config(path: Path | None = None) -> PlanConfig:
    """Load ``config.yaml`` → :class:`PlanConfig`.

    If *path* is ``None`` or the file is empty, all-defaults config is
    returned (every section has sensible defaults).

    Raises
    ------
    ConfigLoadError
        On I/O or validation failure.
    """
    from models.config import PlanConfig as _PlanConfig

    if path is None:
        return _PlanConfig()

    data = _read_yaml(path)

    try:
        return _PlanConfig.model_validate(data)
    except Exception as exc:
        raise ConfigLoadError(path, f"validation failed: {exc}") from exc


def load_inventory_config(path: Path | None = None) -> InventoryConfig:
    """Load ``inventory.yaml`` → :class:`InventoryConfig`.

    If *path* is ``None`` or the file is empty, returns an empty
    inventory (no brownfield nodes).

    Raises
    ------
    ConfigLoadError
        On I/O or validation failure.
    """
    from models.config import InventoryConfig as _InventoryConfig

    if path is None:
        return _InventoryConfig()

    data = _read_yaml(path)

    try:
        return _InventoryConfig.model_validate(data)
    except Exception as exc:
        raise ConfigLoadError(path, f"validation failed: {exc}") from exc


def load_catalog_config(path: Path | None = None) -> CatalogConfig | None:
    """Load ``catalog.yaml`` → :class:`CatalogConfig`, or ``None``.

    If *path* is ``None``, returns ``None`` — the engine runs in
    **inventory-only mode** (pure brownfield).  VMs that don't fit on
    existing nodes are added to the unplaced list without expansion.

    If a file *is* provided, it must contain at least one profile.

    Raises
    ------
    ConfigLoadError
        On I/O or validation failure.
    """
    from models.config import CatalogConfig as _CatalogConfig

    if path is None:
        return None

    data = _read_yaml(path)

    if not data:
        raise ConfigLoadError(path, "catalog file must not be empty")

    try:
        return _CatalogConfig.model_validate(data)
    except Exception as exc:
        raise ConfigLoadError(path, f"validation failed: {exc}") from exc
