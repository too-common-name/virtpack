"""Tests for report/csv_exporter.py — placement audit CSV (HLD §8.5)."""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING

from core.cluster_state import ClusterState
from models.node import Node
from models.vm import VM
from report.csv_exporter import export_placement_csv

if TYPE_CHECKING:
    from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def _make_node(node_id: str, profile: str, *, is_inventory: bool = True) -> Node:
    """Create a test node with generous capacity."""
    cost = 0.0 if is_inventory else 1.0
    return Node(
        id=node_id,
        profile=profile,
        cpu_total=100.0,
        memory_total=500_000.0,
        pods_total=250,
        cost_weight=cost,
        is_inventory=is_inventory,
    )


def _make_vm(name: str, cpu: float = 2.0, memory_mb: float = 4096.0) -> VM:
    return VM(name=name, cpu=cpu, memory_mb=memory_mb)


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read CSV into a list of dicts."""
    with open(path) as fh:
        return list(csv.DictReader(fh))


# ═══════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════


class TestExportPlacementCSV:
    """Tests for the ``export_placement_csv`` function."""

    def test_basic_export(self, tmp_path: Path) -> None:
        """Place 2 VMs on 1 node → CSV has 2 data rows."""
        node = _make_node("node-01", "r740")
        vm1 = _make_vm("web01")
        vm2 = _make_vm("db01", cpu=4.0, memory_mb=8192.0)

        state = ClusterState([node])
        state.place(vm1, node)
        state.place(vm2, node)

        csv_path = tmp_path / "placement_map.csv"
        count = export_placement_csv(state=state, vms=[vm1, vm2], path=csv_path)

        assert count == 2
        assert csv_path.exists()

        rows = _read_csv(csv_path)
        assert len(rows) == 2
        # Sorted by (Target_Node, VM_Name) → db01 before web01
        assert rows[0]["VM_Name"] == "db01"
        assert rows[1]["VM_Name"] == "web01"

    def test_header_columns(self, tmp_path: Path) -> None:
        """CSV header matches HLD §8.5 spec."""
        node = _make_node("n1", "prof")
        vm = _make_vm("vm1")
        state = ClusterState([node])
        state.place(vm, node)

        csv_path = tmp_path / "out.csv"
        export_placement_csv(state=state, vms=[vm], path=csv_path)

        with open(csv_path) as fh:
            reader = csv.reader(fh)
            header = next(reader)

        assert header == ["VM_Name", "vCPU", "RAM_MB", "Target_Node", "Node_Profile"]

    def test_unplaced_vms_excluded(self, tmp_path: Path) -> None:
        """Unplaced VMs should not appear in the CSV."""
        node = _make_node("node-01", "r740")
        placed = _make_vm("placed-vm")
        unplaced = _make_vm("unplaced-vm")

        state = ClusterState([node])
        state.place(placed, node)
        # unplaced is NOT placed

        csv_path = tmp_path / "out.csv"
        count = export_placement_csv(state=state, vms=[placed, unplaced], path=csv_path)

        assert count == 1
        rows = _read_csv(csv_path)
        assert len(rows) == 1
        assert rows[0]["VM_Name"] == "placed-vm"

    def test_empty_cluster(self, tmp_path: Path) -> None:
        """No VMs placed → CSV has only the header row."""
        state = ClusterState()
        csv_path = tmp_path / "out.csv"
        count = export_placement_csv(state=state, vms=[], path=csv_path)

        assert count == 0
        rows = _read_csv(csv_path)
        assert len(rows) == 0

    def test_deterministic_sort(self, tmp_path: Path) -> None:
        """Rows are sorted by (Target_Node, VM_Name) for reproducibility."""
        n1 = _make_node("node-02", "r760")
        n2 = _make_node("node-01", "r740")

        vms = [
            _make_vm("zz-vm"),
            _make_vm("aa-vm"),
            _make_vm("mm-vm"),
        ]

        state = ClusterState([n1, n2])
        # Place zz and mm on node-02, aa on node-01
        state.place(vms[0], n1)  # zz → node-02
        state.place(vms[1], n2)  # aa → node-01
        state.place(vms[2], n1)  # mm → node-02

        csv_path = tmp_path / "out.csv"
        export_placement_csv(state=state, vms=vms, path=csv_path)

        rows = _read_csv(csv_path)
        vm_names = [r["VM_Name"] for r in rows]
        # Expected: node-01/aa-vm, node-02/mm-vm, node-02/zz-vm
        assert vm_names == ["aa-vm", "mm-vm", "zz-vm"]

    def test_correct_resource_values(self, tmp_path: Path) -> None:
        """CPU and RAM values in CSV match the VM's actual resources."""
        node = _make_node("n1", "prof")
        vm = _make_vm("big-db", cpu=16.5, memory_mb=131072.0)

        state = ClusterState([node])
        state.place(vm, node)

        csv_path = tmp_path / "out.csv"
        export_placement_csv(state=state, vms=[vm], path=csv_path)

        rows = _read_csv(csv_path)
        assert rows[0]["vCPU"] == "16.5"
        assert rows[0]["RAM_MB"] == "131072.0"

    def test_multi_node_profiles(self, tmp_path: Path) -> None:
        """VMs on different node profiles show the correct profile name."""
        inv = _make_node("inv-01", "r740-existing")
        cat = _make_node("cat-01", "r760-new", is_inventory=False)

        vm1 = _make_vm("vm-inv")
        vm2 = _make_vm("vm-cat")

        state = ClusterState([inv, cat])
        state.place(vm1, inv)
        state.place(vm2, cat)

        csv_path = tmp_path / "out.csv"
        export_placement_csv(state=state, vms=[vm1, vm2], path=csv_path)

        rows = _read_csv(csv_path)
        profiles = {r["VM_Name"]: r["Node_Profile"] for r in rows}
        assert profiles["vm-inv"] == "r740-existing"
        assert profiles["vm-cat"] == "r760-new"

    def test_returns_placed_count(self, tmp_path: Path) -> None:
        """Return value equals number of placed VMs written."""
        node = _make_node("n1", "p")
        vms = [_make_vm(f"vm-{i}") for i in range(5)]

        state = ClusterState([node])
        for vm in vms[:3]:
            state.place(vm, node)

        csv_path = tmp_path / "out.csv"
        count = export_placement_csv(state=state, vms=vms, path=csv_path)
        assert count == 3
