"""Placement audit CSV export (HLD §8.5).

Writes ``placement_map.csv`` — a deterministic, sorted audit trail that
proves a geometrically valid placement exists for the workload.

Columns::

    VM_Name,vCPU,RAM_MB,Target_Node,Node_Profile

The file is sorted by ``(Target_Node, VM_Name)`` for reproducibility.
"""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from core.cluster_state import ClusterState
    from models.vm import VM

# CSV column order — matches HLD §8.5
_FIELDNAMES: list[str] = [
    "VM_Name",
    "vCPU",
    "RAM_MB",
    "Target_Node",
    "Node_Profile",
]


def export_placement_csv(
    *,
    state: ClusterState,
    vms: list[VM],
    path: Path,
) -> int:
    """Write the placement map to *path* as CSV.

    Only **placed** VMs appear in the output (unplaced VMs are excluded).

    Parameters
    ----------
    state : ClusterState
        Post-placement cluster state with final mappings.
    vms : list[VM]
        All VMs (placed + unplaced).  Only those present in
        ``state.placement_map`` are written.
    path : Path
        Destination file path.  Parent directories must already exist.

    Returns
    -------
    int
        Number of rows written (excluding the header).
    """
    placement_map = state.placement_map  # vm_name → node_id
    node_lookup = {n.id: n for n in state.nodes}

    # Build rows for placed VMs only
    rows: list[dict[str, str | float | int]] = []
    for vm in vms:
        node_id = placement_map.get(vm.name)
        if node_id is None:
            continue  # unplaced
        node = node_lookup[node_id]
        rows.append(
            {
                "VM_Name": vm.name,
                "vCPU": vm.cpu,
                "RAM_MB": vm.memory_mb,
                "Target_Node": node.id,
                "Node_Profile": node.profile,
            }
        )

    # Deterministic sort: by target node, then VM name
    rows.sort(key=lambda r: (str(r["Target_Node"]), str(r["VM_Name"])))

    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)
