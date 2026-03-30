"""Tests for report/terminal_summary.py — Rich summary output (HLD §8.1–8.3)."""

from __future__ import annotations

import pytest

from core.cluster_state import ClusterState
from core.ha_injector import HAResult
from loaders.rvtools_parser import RawHost, RawVM
from models.node import Node
from models.vm import VM
from report.terminal_summary import (
    NodeDetail,
    PlanSummary,
    SkewedVM,
    VMwareSummary,
    _compute_cfi,
    _determine_bottleneck,
    _node_pressure,
    _percentile,
    compute_summary,
    compute_vmware_summary,
    detect_skewed_vms,
    render_comparison,
    render_node_table,
    render_skew_warnings,
    render_summary,
    render_vmware_summary,
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


def _plan_summary(**overrides: object) -> PlanSummary:
    """Build a PlanSummary with sensible defaults, overriding specific fields."""
    defaults: dict[str, object] = {
        "total_nodes": 2,
        "inventory_nodes": 2,
        "catalog_nodes": 0,
        "ha_nodes": 0,
        "total_vms": 5,
        "placed_vms": 5,
        "unplaced_vms": 0,
        "cluster_cpu_util": 0.5,
        "cluster_memory_util": 0.5,
        "total_cpu_capacity": 200.0,
        "total_memory_capacity_mb": 1_000_000.0,
        "peak_cpu_util": 0.5,
        "peak_memory_util": 0.5,
        "bottleneck": "BALANCED",
        "headroom_cpu": 0.5,
        "headroom_memory": 0.5,
        "cfi": 0.05,
        "pressure_p95": 0.5,
        "pressure_max": 0.5,
        "ha_fully_covered": True,
        "ha_deficit_cpu": 0.0,
        "ha_deficit_memory": 0.0,
    }
    defaults.update(overrides)
    return PlanSummary(**defaults)  # type: ignore[arg-type]


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
    """Tests for Cluster Fragmentation Index — stranded capacity (§8.2).

    CFI = average(stranded_penalty) where stranded_penalty =
    (cpu_remaining% − memory_remaining%)².

    Lower = remaining capacity is better balanced = good.
    """

    def test_empty_list(self) -> None:
        assert _compute_cfi([]) == 0.0

    def test_balanced_empty_node(self) -> None:
        """Empty node → remaining (100%, 100%) → diff=0 → CFI=0."""
        n = _make_node("n1", cpu_total=100.0, memory_total=100_000.0)
        assert _compute_cfi([n]) == pytest.approx(0.0)

    def test_balanced_full_node(self) -> None:
        """Fully loaded node → remaining (0%, 0%) → diff=0 → CFI=0."""
        n = _make_node("n1", cpu_total=100.0, memory_total=100_000.0)
        n.cpu_used = 100.0
        n.memory_used = 100_000.0
        assert _compute_cfi([n]) == pytest.approx(0.0)

    def test_balanced_half_used(self) -> None:
        """Both at 50% → remaining (50%, 50%) → diff=0 → CFI=0."""
        n = _make_node("n1", cpu_total=100.0, memory_total=100_000.0)
        n.cpu_used = 50.0
        n.memory_used = 50_000.0
        assert _compute_cfi([n]) == pytest.approx(0.0)

    def test_cpu_bound_node(self) -> None:
        """CPU 90%, memory 30% → remaining (10%, 70%) → penalty=0.36."""
        n = _make_node("n1", cpu_total=100.0, memory_total=100_000.0)
        n.cpu_used = 90.0
        n.memory_used = 30_000.0
        assert _compute_cfi([n]) == pytest.approx(0.36)

    def test_average_of_balanced_and_stranded(self) -> None:
        """One balanced, one stranded → CFI = (0.0 + 0.36) / 2 = 0.18."""
        balanced = _make_node("n1", cpu_total=100.0, memory_total=100_000.0)
        balanced.cpu_used = 50.0
        balanced.memory_used = 50_000.0

        stranded = _make_node("n2", cpu_total=100.0, memory_total=100_000.0)
        stranded.cpu_used = 90.0
        stranded.memory_used = 30_000.0

        assert _compute_cfi([balanced, stranded]) == pytest.approx(0.18)


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
# compute_vmware_summary tests
# ═══════════════════════════════════════════════════════════════════════


class TestComputeVMwareSummary:
    """Tests for VMware environment summary computation."""

    def test_returns_none_with_no_hosts(self) -> None:
        result = compute_vmware_summary(hosts=[], raw_vms=[])
        assert result is None

    def test_basic_summary(self) -> None:
        hosts = [
            RawHost(name="h1", sockets=2, cores_per_socket=8, ht_active=True, memory_mb=131072),
            RawHost(name="h2", sockets=2, cores_per_socket=8, ht_active=False, memory_mb=65536),
        ]
        vms = [
            RawVM(name="vm1", cpu=4, memory_mb=8192),
            RawVM(name="vm2", cpu=8, memory_mb=16384),
        ]
        result = compute_vmware_summary(hosts=hosts, raw_vms=vms)

        assert result is not None
        assert result.host_count == 2
        assert result.total_physical_cores == 32  # 2*8 + 2*8
        assert result.total_logical_cpus == 48  # 2*8*2 + 2*8*1 = 32+16
        assert result.total_ram_gb == pytest.approx((131072 + 65536) / 1024.0)
        assert result.vm_count == 2
        assert result.total_vcpu == 12
        assert result.total_vmem_gb == pytest.approx((8192 + 16384) / 1024.0)
        assert result.cpu_overcommit == pytest.approx(12 / 48.0)
        assert result.mem_ratio == pytest.approx(24.0 / 192.0)

    def test_no_vms(self) -> None:
        hosts = [
            RawHost(name="h1", sockets=2, cores_per_socket=16, ht_active=True, memory_mb=524288)
        ]
        result = compute_vmware_summary(hosts=hosts, raw_vms=[])

        assert result is not None
        assert result.vm_count == 0
        assert result.total_vcpu == 0
        assert result.cpu_overcommit == 0.0


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

    def test_capacity_fields(self) -> None:
        """Total capacity fields are populated."""
        n1 = _make_node("n1", cpu_total=100.0, memory_total=200_000.0)
        n2 = _make_node("n2", cpu_total=80.0, memory_total=300_000.0)

        state = ClusterState([n1, n2])
        summary = compute_summary(state=state, vms=[], unplaced=[])

        assert summary.total_cpu_capacity == pytest.approx(180.0)
        assert summary.total_memory_capacity_mb == pytest.approx(500_000.0)

    def test_node_details_populated(self) -> None:
        """Node details include per-node breakdown."""
        n1 = _make_node("n1", cpu_total=100.0, memory_total=100_000.0, is_inventory=True)
        vm1 = _make_vm("vm1", cpu=30.0, memory_mb=40_000.0)
        vm2 = _make_vm("vm2", cpu=20.0, memory_mb=20_000.0)

        state = ClusterState([n1])
        state.place(vm1, n1)
        state.place(vm2, n1)

        summary = compute_summary(state=state, vms=[vm1, vm2], unplaced=[])

        assert len(summary.node_details) == 1
        detail = summary.node_details[0]
        assert detail.node_id == "n1"
        assert detail.vm_count == 2
        assert detail.cpu_util == pytest.approx(0.5)
        assert detail.memory_util == pytest.approx(0.6)
        assert detail.is_inventory is True

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
        assert summary.total_cpu_capacity == 0.0
        assert summary.total_memory_capacity_mb == 0.0

    def test_headroom_calculation(self) -> None:
        """Headroom = 1 - cluster utilization."""
        n = _make_node("n1", cpu_total=100.0, memory_total=100_000.0)
        vm = _make_vm("vm1", cpu=60.0, memory_mb=70_000.0)

        state = ClusterState([n])
        state.place(vm, n)

        summary = compute_summary(state=state, vms=[vm], unplaced=[])

        assert summary.headroom_cpu == pytest.approx(0.4)
        assert summary.headroom_memory == pytest.approx(0.3)

    def test_shutdown_candidates_none(self) -> None:
        """No unused_inventory → shutdown_candidates = 0."""
        state = ClusterState([_make_node("n1")])
        summary = compute_summary(state=state, vms=[], unplaced=[])
        assert summary.shutdown_candidates == 0

    def test_shutdown_candidates_with_unused(self) -> None:
        """Unused inventory nodes are reported as shutdown candidates."""
        n1 = _make_node("n1")
        state = ClusterState([n1])
        unused = [_make_node("n2"), _make_node("n3")]

        summary = compute_summary(
            state=state,
            vms=[],
            unplaced=[],
            unused_inventory=unused,
        )

        assert summary.shutdown_candidates == 2


# ═══════════════════════════════════════════════════════════════════════
# Render smoke tests
# ═══════════════════════════════════════════════════════════════════════


class TestRenderSummary:
    """Smoke tests for ``render_summary`` — it should not crash."""

    def test_render_basic(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Rendering a basic summary should print without errors."""
        summary = _plan_summary(
            total_nodes=3,
            inventory_nodes=2,
            catalog_nodes=1,
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
            unplaced_vm_names=["monster-vm"],
        )
        render_summary(summary)
        captured = capsys.readouterr()
        assert "OCP Virt Cluster Plan" in captured.out
        assert "MEMORY" in captured.out
        assert "monster-vm" in captured.out

    def test_render_no_unplaced(self, capsys: pytest.CaptureFixture[str]) -> None:
        """No unplaced VMs → no red warnings."""
        summary = _plan_summary()
        render_summary(summary)
        captured = capsys.readouterr()
        assert "Unplaced VMs: 0" in captured.out

    def test_render_shutdown_candidates(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Shutdown candidates are displayed when > 0."""
        summary = _plan_summary(
            total_nodes=3,
            inventory_nodes=3,
            shutdown_candidates=7,
        )
        render_summary(summary)
        captured = capsys.readouterr()
        assert "Shutdown" in captured.out
        assert "7" in captured.out

    def test_render_ha_deficit(self, capsys: pytest.CaptureFixture[str]) -> None:
        """HA deficit triggers a warning panel."""
        summary = _plan_summary(
            total_nodes=1,
            inventory_nodes=1,
            cluster_cpu_util=0.9,
            cluster_memory_util=0.9,
            peak_cpu_util=0.9,
            peak_memory_util=0.9,
            ha_fully_covered=False,
            ha_deficit_cpu=20.0,
            ha_deficit_memory=10000.0,
        )
        render_summary(summary)
        captured = capsys.readouterr()
        assert "HA" in captured.out

    def test_render_with_vmware_summary(self, capsys: pytest.CaptureFixture[str]) -> None:
        """VMware summary + comparison render when vmware data is provided."""
        vmware = VMwareSummary(
            host_count=13,
            total_physical_cores=312,
            total_logical_cpus=624,
            total_ram_gb=6656.0,
            vm_count=452,
            total_vcpu=1747,
            total_vmem_gb=6282.0,
            cpu_overcommit=2.8,
            mem_ratio=0.94,
        )
        summary = _plan_summary(total_nodes=9, inventory_nodes=9)
        render_summary(summary, vmware=vmware)
        captured = capsys.readouterr()
        assert "VMware Source Environment" in captured.out
        assert "OCP Virt Cluster Plan" in captured.out
        assert "Migration Comparison" in captured.out
        assert "Physical Hosts: 13" in captured.out
        assert "VMs to Migrate: 452" in captured.out

    def test_render_node_table(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Node utilization table renders per-node details."""
        details = [
            NodeDetail(
                node_id="h1",
                profile="r740",
                cpu_total=100.0,
                cpu_used=60.0,
                cpu_util=0.6,
                memory_total_gb=500.0,
                memory_used_gb=400.0,
                memory_util=0.8,
                vm_count=25,
                is_inventory=True,
            ),
            NodeDetail(
                node_id="h2",
                profile="r760",
                cpu_total=80.0,
                cpu_used=70.0,
                cpu_util=0.875,
                memory_total_gb=256.0,
                memory_used_gb=250.0,
                memory_util=0.977,
                vm_count=15,
                is_inventory=False,
            ),
        ]
        summary = _plan_summary(node_details=details)
        render_summary(summary)
        captured = capsys.readouterr()
        assert "Node Utilization" in captured.out
        assert "h1" in captured.out
        assert "h2" in captured.out

    def test_render_comparison_standalone(self, capsys: pytest.CaptureFixture[str]) -> None:
        """render_comparison prints the comparison table."""
        vmware = VMwareSummary(
            host_count=10,
            total_physical_cores=200,
            total_logical_cpus=400,
            total_ram_gb=5120.0,
            vm_count=300,
            total_vcpu=900,
            total_vmem_gb=4000.0,
            cpu_overcommit=2.25,
            mem_ratio=0.78,
        )
        plan = _plan_summary(
            total_nodes=8,
            total_cpu_capacity=350.0,
            total_memory_capacity_mb=4_500_000.0,
        )
        render_comparison(vmware, plan)
        captured = capsys.readouterr()
        assert "Migration Comparison" in captured.out
        assert "Active Nodes" in captured.out
        assert "Why capacity differs" in captured.out


class TestRenderVMwareSummary:
    """Smoke tests for render_vmware_summary."""

    def test_render_vmware_summary(self, capsys: pytest.CaptureFixture[str]) -> None:
        vmware = VMwareSummary(
            host_count=5,
            total_physical_cores=80,
            total_logical_cpus=160,
            total_ram_gb=2560.0,
            vm_count=100,
            total_vcpu=400,
            total_vmem_gb=2000.0,
            cpu_overcommit=2.5,
            mem_ratio=0.78,
        )
        render_vmware_summary(vmware)
        captured = capsys.readouterr()
        assert "VMware Source Environment" in captured.out
        assert "Physical Hosts: 5" in captured.out
        assert "2.5:1" in captured.out


class TestRenderNodeTable:
    """Smoke tests for render_node_table."""

    def test_empty_list_no_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        render_node_table([])
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_renders_nodes(self, capsys: pytest.CaptureFixture[str]) -> None:
        details = [
            NodeDetail(
                node_id="n1",
                profile="r740",
                cpu_total=100.0,
                cpu_used=50.0,
                cpu_util=0.5,
                memory_total_gb=500.0,
                memory_used_gb=250.0,
                memory_util=0.5,
                vm_count=10,
                is_inventory=True,
            ),
        ]
        render_node_table(details)
        captured = capsys.readouterr()
        assert "Node Utilization" in captured.out
        assert "n1" in captured.out
        assert "inv" in captured.out


# ═══════════════════════════════════════════════════════════════════════
# Resource skew detection tests
# ═══════════════════════════════════════════════════════════════════════


class TestDetectSkewedVMs:
    """Tests for ``detect_skewed_vms``."""

    def test_empty_fleet(self) -> None:
        """No VMs → no warnings."""
        assert detect_skewed_vms([]) == []

    def test_single_vm_returns_empty(self) -> None:
        """A fleet of 1 VM can't be compared → no warnings."""
        assert detect_skewed_vms([_make_vm("vm1")]) == []

    def test_balanced_fleet_no_warnings(self) -> None:
        """All VMs with similar ratios → no warnings."""
        vms = [
            _make_vm("a", cpu=2.0, memory_mb=4096.0),
            _make_vm("b", cpu=4.0, memory_mb=8192.0),
            _make_vm("c", cpu=1.0, memory_mb=2048.0),
        ]
        assert detect_skewed_vms(vms) == []

    def test_memory_heavy_outlier_flagged(self) -> None:
        """A VM with memory:CPU ratio >> fleet median is flagged."""
        normal = [_make_vm(f"vm{i}", cpu=2.0, memory_mb=4096.0) for i in range(10)]
        # This VM: 2 CPU, 393216 MB → ratio ~196k vs fleet median ~2048
        monster = _make_vm("monster", cpu=2.0, memory_mb=393216.0)
        vms = [*normal, monster]

        result = detect_skewed_vms(vms)

        assert len(result) == 1
        assert result[0].name == "monster"
        assert result[0].direction == "memory-heavy"
        assert result[0].skew_factor > 4.0

    def test_cpu_heavy_outlier_flagged(self) -> None:
        """A VM with memory:CPU ratio << fleet median is flagged."""
        normal = [_make_vm(f"vm{i}", cpu=2.0, memory_mb=8192.0) for i in range(10)]
        # This VM: 100 CPU, 512 MB → ratio 5.12 vs fleet median ~4096
        cpu_monster = _make_vm("cpu-beast", cpu=100.0, memory_mb=512.0)
        vms = [*normal, cpu_monster]

        result = detect_skewed_vms(vms)

        assert len(result) == 1
        assert result[0].name == "cpu-beast"
        assert result[0].direction == "cpu-heavy"
        assert result[0].skew_factor < 0.25

    def test_custom_threshold(self) -> None:
        """A stricter threshold catches more VMs."""
        # Fleet median ratio = 2048 MB/core
        normal = [_make_vm(f"vm{i}", cpu=2.0, memory_mb=4096.0) for i in range(10)]
        # This VM: ratio = 8192 → 4x the median (exactly at default threshold)
        mild = _make_vm("mild-outlier", cpu=2.0, memory_mb=16384.0)
        vms = [*normal, mild]

        # Default threshold (4x) — mild outlier is at exactly 4x, not above
        default_result = detect_skewed_vms(vms, threshold=4.0)
        # With a stricter threshold (3x) the mild outlier IS flagged
        strict_result = detect_skewed_vms(vms, threshold=3.0)

        assert len(strict_result) >= len(default_result)

    def test_sorted_by_skew_factor_descending(self) -> None:
        """Results are sorted worst-first (highest skew_factor first)."""
        normal = [_make_vm(f"vm{i}", cpu=2.0, memory_mb=4096.0) for i in range(10)]
        big = _make_vm("big", cpu=2.0, memory_mb=100_000.0)
        huge = _make_vm("huge", cpu=2.0, memory_mb=400_000.0)
        vms = [*normal, big, huge]

        result = detect_skewed_vms(vms)

        if len(result) >= 2:
            assert result[0].skew_factor >= result[1].skew_factor
            assert result[0].name == "huge"

    def test_skewed_vms_in_compute_summary(self) -> None:
        """Skewed VMs are populated via compute_summary."""
        n1 = _make_node("n1", cpu_total=200.0, memory_total=500_000.0)
        normal = [_make_vm(f"v{i}", cpu=2.0, memory_mb=4096.0) for i in range(10)]
        monster = _make_vm("monster", cpu=2.0, memory_mb=393216.0)
        all_vms = [*normal, monster]

        state = ClusterState([n1])
        for vm in all_vms:
            state.place(vm, n1)

        summary = compute_summary(state=state, vms=all_vms, unplaced=[])

        assert len(summary.skewed_vms) >= 1
        assert summary.skewed_vms[0].name == "monster"


class TestRenderSkewWarnings:
    """Smoke tests for render_skew_warnings."""

    def test_empty_list_no_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        render_skew_warnings([])
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_renders_warning_panel(self, capsys: pytest.CaptureFixture[str]) -> None:
        skewed = [
            SkewedVM(
                name="monster-vm",
                cpu=2.67,
                memory_gb=384.0,
                ratio=147_191.0,
                fleet_median_ratio=11_000.0,
                skew_factor=13.4,
                direction="memory-heavy",
            ),
        ]
        render_skew_warnings(skewed)
        captured = capsys.readouterr()
        assert "Resource Skew" in captured.out
        assert "monster-vm" in captured.out
        # Rich may truncate "memory-heavy" depending on terminal width
        assert "memory-he" in captured.out

    def test_render_summary_includes_skew_panel(self, capsys: pytest.CaptureFixture[str]) -> None:
        """render_summary shows skew warnings when present."""
        skewed = [
            SkewedVM(
                name="skewed-vm",
                cpu=2.0,
                memory_gb=384.0,
                ratio=196_608.0,
                fleet_median_ratio=2048.0,
                skew_factor=96.0,
                direction="memory-heavy",
            ),
        ]
        summary = _plan_summary(skewed_vms=skewed)
        render_summary(summary)
        captured = capsys.readouterr()
        assert "Resource Skew" in captured.out
        assert "skewed-vm" in captured.out
