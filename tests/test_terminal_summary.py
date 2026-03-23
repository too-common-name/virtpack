"""Tests for report/terminal_summary.py — Rich summary output (HLD §8.1–8.3)."""

from __future__ import annotations

import pytest

from core.cluster_state import ClusterState
from core.ha_injector import HAResult
from models.node import Node
from models.vm import VM
from report.terminal_summary import (
    PlanSummary,
    _compute_cfi,
    _determine_bottleneck,
    _node_pressure,
    _percentile,
    compute_summary,
    render_summary,
)

# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def _make_node(
    node_id: str,
    profile: str = "r740",
    *,
    cpu_total: float = 100.0,
    memory_total: float = 500_000.0,
    pods_total: int = 250,
    is_inventory: bool = True,
) -> Node:
    cost = 0.0 if is_inventory else 1.0
    return Node(
        id=node_id,
        profile=profile,
        cpu_total=cpu_total,
        memory_total=memory_total,
        pods_total=pods_total,
        cost_weight=cost,
        is_inventory=is_inventory,
    )


def _make_vm(name: str, cpu: float = 2.0, memory_mb: float = 4096.0) -> VM:
    return VM(name=name, cpu=cpu, memory_mb=memory_mb)


# ═══════════════════════════════════════════════════════════════════════
# Pure metric function tests
# ═══════════════════════════════════════════════════════════════════════


class TestNodePressure:
    """Tests for _node_pressure (§8.3)."""

    def test_empty_node(self) -> None:
        node = _make_node("n1")
        assert _node_pressure(node) == 0.0

    def test_cpu_dominant(self) -> None:
        node = _make_node("n1", cpu_total=100.0, memory_total=100_000.0)
        node.cpu_used = 80.0  # 80% CPU
        node.memory_used = 50_000.0  # 50% memory
        assert _node_pressure(node) == pytest.approx(0.8)

    def test_memory_dominant(self) -> None:
        node = _make_node("n1", cpu_total=100.0, memory_total=100_000.0)
        node.cpu_used = 30.0
        node.memory_used = 90_000.0  # 90% memory
        assert _node_pressure(node) == pytest.approx(0.9)


class TestPercentile:
    """Tests for _percentile helper."""

    def test_empty_list(self) -> None:
        assert _percentile([], 95) == 0.0

    def test_single_value(self) -> None:
        assert _percentile([0.5], 95) == 0.5

    def test_p95_of_100(self) -> None:
        values = sorted(float(i) / 100.0 for i in range(1, 101))
        assert _percentile(values, 95) == pytest.approx(0.95)

    def test_p50_median(self) -> None:
        values = sorted([0.1, 0.2, 0.3, 0.4, 0.5])
        assert _percentile(values, 50) == pytest.approx(0.3)


class TestComputeCFI:
    """Tests for Cluster Fragmentation Index (§8.2)."""

    def test_empty_nodes(self) -> None:
        assert _compute_cfi([]) == 0.0

    def test_fully_loaded_nodes(self) -> None:
        """Nodes at 100% memory → frag penalty = 0 → CFI = 0."""
        n = _make_node("n1", memory_total=100_000.0)
        n.memory_used = 100_000.0
        assert _compute_cfi([n]) == pytest.approx(0.0)

    def test_empty_nodes_high_frag(self) -> None:
        """Empty nodes → frag penalty = (1.0)² = 1.0 → CFI = 1.0."""
        n = _make_node("n1", memory_total=100_000.0)
        assert _compute_cfi([n]) == pytest.approx(1.0)

    def test_average_of_two_nodes(self) -> None:
        """One full, one empty → CFI = (0.0 + 1.0) / 2 = 0.5."""
        full = _make_node("n1", memory_total=100_000.0)
        full.memory_used = 100_000.0
        empty = _make_node("n2", memory_total=100_000.0)
        assert _compute_cfi([full, empty]) == pytest.approx(0.5)


class TestDetermineBottleneck:
    """Tests for _determine_bottleneck."""

    def test_cpu_bottleneck(self) -> None:
        assert _determine_bottleneck(0.85, 0.60) == "CPU"

    def test_memory_bottleneck(self) -> None:
        assert _determine_bottleneck(0.60, 0.85) == "MEMORY"

    def test_balanced_within_threshold(self) -> None:
        assert _determine_bottleneck(0.80, 0.81) == "BALANCED"

    def test_balanced_exact(self) -> None:
        assert _determine_bottleneck(0.75, 0.75) == "BALANCED"


# ═══════════════════════════════════════════════════════════════════════
# compute_summary integration tests
# ═══════════════════════════════════════════════════════════════════════


class TestComputeSummary:
    """Tests for the ``compute_summary`` builder function."""

    def test_basic_summary(self) -> None:
        """A simple scenario: 2 nodes, 3 VMs placed, 1 unplaced."""
        n1 = _make_node("inv-01", "r740", is_inventory=True)
        n2 = _make_node("cat-01", "r760", is_inventory=False)

        vm1 = _make_vm("vm1", cpu=10.0, memory_mb=50_000.0)
        vm2 = _make_vm("vm2", cpu=20.0, memory_mb=100_000.0)
        vm3 = _make_vm("vm3", cpu=5.0, memory_mb=25_000.0)
        vm_unplaced = _make_vm("vm-big", cpu=999.0, memory_mb=999_999.0)

        state = ClusterState([n1, n2])
        state.place(vm1, n1)
        state.place(vm2, n2)
        state.place(vm3, n1)

        all_vms = [vm1, vm2, vm3, vm_unplaced]
        unplaced = [vm_unplaced]

        summary = compute_summary(state=state, vms=all_vms, unplaced=unplaced)

        assert summary.total_nodes == 2
        assert summary.inventory_nodes == 1
        assert summary.catalog_nodes == 1
        assert summary.total_vms == 4
        assert summary.placed_vms == 3
        assert summary.unplaced_vms == 1
        assert summary.unplaced_vm_names == ["vm-big"]

    def test_cluster_utilization(self) -> None:
        """Cluster utilization aggregated across all nodes."""
        n1 = _make_node("n1", cpu_total=100.0, memory_total=200_000.0)
        n2 = _make_node("n2", cpu_total=100.0, memory_total=200_000.0)

        vm1 = _make_vm("vm1", cpu=50.0, memory_mb=100_000.0)
        vm2 = _make_vm("vm2", cpu=25.0, memory_mb=50_000.0)

        state = ClusterState([n1, n2])
        state.place(vm1, n1)
        state.place(vm2, n2)

        summary = compute_summary(state=state, vms=[vm1, vm2], unplaced=[])

        # Total CPU: 200, used: 75 → 37.5%
        assert summary.cluster_cpu_util == pytest.approx(0.375)
        # Total MEM: 400k, used: 150k → 37.5%
        assert summary.cluster_memory_util == pytest.approx(0.375)
        assert summary.bottleneck == "BALANCED"

    def test_peak_utilization(self) -> None:
        """Peak is per-node max, not the cluster average."""
        n1 = _make_node("n1", cpu_total=100.0, memory_total=100_000.0)
        n2 = _make_node("n2", cpu_total=100.0, memory_total=100_000.0)

        # Load n1 heavily
        vm_heavy = _make_vm("heavy", cpu=90.0, memory_mb=80_000.0)
        vm_light = _make_vm("light", cpu=10.0, memory_mb=10_000.0)

        state = ClusterState([n1, n2])
        state.place(vm_heavy, n1)
        state.place(vm_light, n2)

        summary = compute_summary(state=state, vms=[vm_heavy, vm_light], unplaced=[])

        assert summary.peak_cpu_util == pytest.approx(0.9)
        assert summary.peak_memory_util == pytest.approx(0.8)

    def test_ha_result_integration(self) -> None:
        """HA result data flows into the summary."""
        n1 = _make_node("n1")
        state = ClusterState([n1])

        ha_node = _make_node("ha-01", is_inventory=False)
        ha = HAResult(nodes_added=[ha_node], deficit_cpu=0.0, deficit_memory=0.0)

        summary = compute_summary(state=state, vms=[], unplaced=[], ha_result=ha)

        assert summary.ha_nodes == 1
        assert summary.ha_fully_covered is True

    def test_ha_deficit_reported(self) -> None:
        """Uncovered HA deficit is reported in the summary."""
        state = ClusterState([_make_node("n1")])
        ha = HAResult(deficit_cpu=10.0, deficit_memory=5000.0)

        summary = compute_summary(state=state, vms=[], unplaced=[], ha_result=ha)

        assert summary.ha_fully_covered is False
        assert summary.ha_deficit_cpu == pytest.approx(10.0)
        assert summary.ha_deficit_memory == pytest.approx(5000.0)

    def test_empty_cluster(self) -> None:
        """Empty cluster produces safe defaults."""
        state = ClusterState()
        summary = compute_summary(state=state, vms=[], unplaced=[])

        assert summary.total_nodes == 0
        assert summary.cluster_cpu_util == 0.0
        assert summary.cluster_memory_util == 0.0
        assert summary.cfi == 0.0
        assert summary.pressure_p95 == 0.0
        assert summary.pressure_max == 0.0
        assert summary.bottleneck == "BALANCED"

    def test_headroom_calculation(self) -> None:
        """Headroom = 1 - cluster utilization."""
        n = _make_node("n1", cpu_total=100.0, memory_total=100_000.0)
        vm = _make_vm("vm1", cpu=60.0, memory_mb=70_000.0)

        state = ClusterState([n])
        state.place(vm, n)

        summary = compute_summary(state=state, vms=[vm], unplaced=[])

        assert summary.headroom_cpu == pytest.approx(0.4)
        assert summary.headroom_memory == pytest.approx(0.3)


# ═══════════════════════════════════════════════════════════════════════
# Render smoke test
# ═══════════════════════════════════════════════════════════════════════


class TestRenderSummary:
    """Smoke tests for ``render_summary`` — it should not crash."""

    def test_render_basic(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Rendering a basic summary should print without errors."""
        summary = PlanSummary(
            total_nodes=3,
            inventory_nodes=2,
            catalog_nodes=1,
            ha_nodes=0,
            total_vms=10,
            placed_vms=9,
            unplaced_vms=1,
            cluster_cpu_util=0.65,
            cluster_memory_util=0.78,
            peak_cpu_util=0.82,
            peak_memory_util=0.91,
            bottleneck="MEMORY",
            headroom_cpu=0.35,
            headroom_memory=0.22,
            cfi=0.12,
            pressure_p95=0.85,
            pressure_max=0.91,
            ha_fully_covered=True,
            ha_deficit_cpu=0.0,
            ha_deficit_memory=0.0,
            unplaced_vm_names=["monster-vm"],
        )
        render_summary(summary)
        captured = capsys.readouterr()
        assert "Cluster Plan Summary" in captured.out
        assert "MEMORY" in captured.out
        assert "monster-vm" in captured.out

    def test_render_no_unplaced(self, capsys: pytest.CaptureFixture[str]) -> None:
        """No unplaced VMs → no red warnings."""
        summary = PlanSummary(
            total_nodes=2,
            inventory_nodes=2,
            catalog_nodes=0,
            ha_nodes=0,
            total_vms=5,
            placed_vms=5,
            unplaced_vms=0,
            cluster_cpu_util=0.5,
            cluster_memory_util=0.5,
            peak_cpu_util=0.5,
            peak_memory_util=0.5,
            bottleneck="BALANCED",
            headroom_cpu=0.5,
            headroom_memory=0.5,
            cfi=0.05,
            pressure_p95=0.5,
            pressure_max=0.5,
            ha_fully_covered=True,
            ha_deficit_cpu=0.0,
            ha_deficit_memory=0.0,
        )
        render_summary(summary)
        captured = capsys.readouterr()
        assert "Unplaced VMs: 0" in captured.out

    def test_render_ha_deficit(self, capsys: pytest.CaptureFixture[str]) -> None:
        """HA deficit triggers a warning panel."""
        summary = PlanSummary(
            total_nodes=1,
            inventory_nodes=1,
            catalog_nodes=0,
            ha_nodes=0,
            total_vms=1,
            placed_vms=1,
            unplaced_vms=0,
            cluster_cpu_util=0.9,
            cluster_memory_util=0.9,
            peak_cpu_util=0.9,
            peak_memory_util=0.9,
            bottleneck="BALANCED",
            headroom_cpu=0.1,
            headroom_memory=0.1,
            cfi=0.01,
            pressure_p95=0.9,
            pressure_max=0.9,
            ha_fully_covered=False,
            ha_deficit_cpu=20.0,
            ha_deficit_memory=10000.0,
        )
        render_summary(summary)
        captured = capsys.readouterr()
        assert "HA" in captured.out
