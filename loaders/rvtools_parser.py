"""RVTools Excel parser — ETL for VM workloads (HLD §3.1, §3.3.1).

Reads the ``vInfo`` sheet from an RVTools ``.xlsx`` export and produces
a list of raw VM records (pre-normalization).

Optionally reads the ``vHost`` sheet for host auto-discovery, producing
:class:`InventoryProfile` objects that represent the existing physical
hardware (HLD §3.3.1).

Design decisions
----------------
* **pandas** is used for robust Excel parsing (column name lookup, dtype
  handling, missing-value semantics).
* The parser returns *raw* VM tuples — normalization (overcommit) is done
  separately by :func:`core.normalizer.normalize_vm`.
* Column names are case-insensitive to handle variations across RVTools
  versions and customer exports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════
# Data containers (pre-normalization)
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class RawVM:
    """A VM record extracted from the RVTools vInfo sheet.

    These are *raw* values — CPU overcommit normalization is applied
    later by the Normalizer.
    """

    name: str
    cpu: int
    memory_mb: int


@dataclass(frozen=True, slots=True)
class RawHost:
    """A physical host record extracted from the RVTools vHost sheet.

    Used for host auto-discovery (HLD §3.3.1).

    Carries full CPU topology so the caller can build a
    :class:`CpuTopology` directly::

        CpuTopology(
            sockets=host.sockets,
            cores_per_socket=host.cores_per_socket,
            threads_per_core=2 if host.ht_active else 1,
        )
    """

    name: str
    sockets: int  # vHost "# CPU"
    cores_per_socket: int  # vHost "Cores per CPU"
    ht_active: bool  # vHost "HT Active"
    memory_mb: int  # vHost "# Memory" (raw MB)


# ═══════════════════════════════════════════════════════════════════════
# Errors
# ═══════════════════════════════════════════════════════════════════════


class RVToolsParseError(Exception):
    """Raised when the RVTools file cannot be parsed."""

    def __init__(self, path: Path, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


# ═══════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════

# Required vInfo columns (lowercase for case-insensitive matching)
_VINFO_REQUIRED = {"vm", "cpus", "memory", "powerstate"}
_VINFO_FILTER_COLS = {"srm placeholder", "template"}

# Required vHost columns (lowercase)
# Note: "ht active" is intentionally optional — not all RVTools exports
# include it.  When absent, we default to ht_active=False (conservative).
_VHOST_REQUIRED = {"host", "# cpu", "cores per cpu", "# memory"}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase and strip whitespace from column names."""
    df.columns = pd.Index([str(c).strip().lower() for c in df.columns])
    return df


def _read_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    """Read a single sheet from an Excel file, normalizing column names.

    Raises
    ------
    RVToolsParseError
        If the file cannot be read or the sheet doesn't exist.
    """
    try:
        df = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
    except FileNotFoundError as exc:
        raise RVToolsParseError(path, "file not found") from exc
    except ValueError as exc:
        raise RVToolsParseError(path, f"sheet '{sheet_name}' not found") from exc
    except Exception as exc:
        raise RVToolsParseError(path, f"cannot read Excel file: {exc}") from exc
    return _normalize_columns(df)


def _check_columns(
    df: pd.DataFrame,
    required: set[str],
    sheet: str,
    path: Path,
) -> None:
    """Verify that all required columns exist in the DataFrame."""
    missing = required - set(df.columns)
    if missing:
        raise RVToolsParseError(
            path,
            f"sheet '{sheet}' is missing required columns: {sorted(missing)}",
        )


# ═══════════════════════════════════════════════════════════════════════
# vInfo parsing (VM workloads)
# ═══════════════════════════════════════════════════════════════════════


def parse_vinfo(path: Path) -> list[RawVM]:
    """Parse the ``vInfo`` sheet and return filtered VM records.

    Filtering rules (HLD §3.1):
      * ``Powerstate == poweredOn``
      * ``SRM Placeholder`` is falsy (``False`` / ``No`` / empty / NaN)
      * ``Template`` is falsy (``False`` / ``No`` / empty / NaN)

    Columns ``SRM Placeholder`` and ``Template`` are optional — if missing,
    no filtering is applied on that dimension (conservative: include the VM).

    Parameters
    ----------
    path : Path
        Path to the RVTools ``.xlsx`` file.

    Returns
    -------
    list[RawVM]
        Raw VM records (pre-normalization), sorted by name for
        deterministic output.

    Raises
    ------
    RVToolsParseError
        On I/O failure or missing required columns.
    """
    df = _read_sheet(path, "vInfo")
    _check_columns(df, _VINFO_REQUIRED, "vInfo", path)

    # ── Filter: poweredOn only ────────────────────────────────────────
    mask = df["powerstate"].astype(str).str.strip().str.lower() == "poweredon"

    # ── Filter: SRM Placeholder == false / empty / NaN ────────────────
    if "srm placeholder" in df.columns:
        srm = df["srm placeholder"].fillna("").astype(str).str.strip().str.lower()
        mask = mask & ~srm.isin({"true", "yes", "1"})

    # ── Filter: Template == false / empty / NaN ───────────────────────
    if "template" in df.columns:
        tmpl = df["template"].fillna("").astype(str).str.strip().str.lower()
        mask = mask & ~tmpl.isin({"true", "yes", "1"})

    filtered = df[mask].copy()

    # ── Build RawVM list ──────────────────────────────────────────────
    vms: list[RawVM] = []
    for _, row in filtered.iterrows():
        raw_name = row["vm"]
        if pd.isna(raw_name):
            continue
        name = str(raw_name).strip()
        if not name:
            continue
        vms.append(
            RawVM(
                name=name,
                cpu=int(row["cpus"]),
                memory_mb=int(row["memory"]),
            )
        )

    return sorted(vms, key=lambda v: v.name)


# ═══════════════════════════════════════════════════════════════════════
# vHost parsing (host auto-discovery — HLD §3.3.1)
# ═══════════════════════════════════════════════════════════════════════


def parse_vhost(path: Path) -> list[RawHost]:
    """Parse the ``vHost`` sheet for physical host auto-discovery.

    Extracts full CPU topology (sockets, cores per socket, HT status)
    and raw memory (MB) for each physical host.  These records are used
    to build inventory profiles automatically, eliminating the need for
    a manual ``inventory.yaml``.

    RVTools vHost column mapping:

    ============== ========================= ========= =================
    RawHost field  vHost column              Example   Required?
    ============== ========================= ========= =================
    sockets        ``# CPU``                 2         Yes
    cores_per_socket ``Cores per CPU``       8         Yes
    ht_active      ``HT Active``             True      No (default False)
    memory_mb      ``# Memory``              524253    Yes
    ============== ========================= ========= =================

    Parameters
    ----------
    path : Path
        Path to the RVTools ``.xlsx`` file.

    Returns
    -------
    list[RawHost]
        Raw host records, sorted by name for deterministic output.

    Raises
    ------
    RVToolsParseError
        On I/O failure or missing required columns.
    """
    df = _read_sheet(path, "vHost")
    _check_columns(df, _VHOST_REQUIRED, "vHost", path)

    hosts: list[RawHost] = []
    for _, row in df.iterrows():
        raw_name = row["host"]
        if pd.isna(raw_name):
            continue
        name = str(raw_name).strip()
        if not name:
            continue

        sockets = int(row["# cpu"])
        cores_per_socket = int(row["cores per cpu"])
        memory_mb = int(row["# memory"])

        # HT Active is optional — default to False (conservative)
        if "ht active" in df.columns and not pd.isna(row["ht active"]):
            ht_raw = str(row["ht active"]).strip().lower()
            ht_active = ht_raw in {"true", "yes", "1"}
        else:
            ht_active = False

        hosts.append(
            RawHost(
                name=name,
                sockets=sockets,
                cores_per_socket=cores_per_socket,
                ht_active=ht_active,
                memory_mb=memory_mb,
            )
        )

    return sorted(hosts, key=lambda h: h.name)
