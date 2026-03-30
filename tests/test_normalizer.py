"""Tests for core.normalizer — resource normalization functions.

Test categories:
  1. VM normalization (overcommit ratios)
  2. MCO kubelet CPU reservation step function
  3. MCO kubelet memory reservation step function
  4. Effective CPU (HT adjustment)
  5. Usable capacity (overhead subtraction)
  6. Full node normalization pipeline (overheads + safety)
  7. Build inventory nodes
  8. Build catalog node
"""

from __future__ import annotations

import pytest

from core.normalizer import (
    build_catalog_node,
    build_inventory_nodes,
    compute_effective_cpu,
    compute_usable_capacity,
    kubelet_reserved_cpu,
    kubelet_reserved_memory_mb,
    normalize_node_capacity,
    normalize_vm,
)
from models.config import (
    CatalogProfile,
    CpuTopology,
    InventoryConfig,
    InventoryProfile,
    OvercommitConfig,
    PlanConfig,
    VirtOverheads,
)

# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def _topo(
    sockets: int = 2,
    cores_per_socket: int = 32,
    threads_per_core: int = 2,
) -> CpuTopology:
    return CpuTopology(
        sockets=sockets,
        cores_per_socket=cores_per_socket,
        threads_per_core=threads_per_core,
    )


def _overheads(**kwargs: float) -> VirtOverheads:
    return VirtOverheads(**kwargs)


# ═══════════════════════════════════════════════════════════════════════
# 1. VM Normalization
# ═══════════════════════════════════════════════════════════════════════


class TestNormalizeVM:
    """VM overcommit ratio application."""

    def test_cpu_overcommit_8_to_1(self) -> None:
        vm = normalize_vm(
            name="db01",
            raw_cpu=8.0,
            raw_memory_mb=32768.0,
            overcommit=OvercommitConfig(cpu_ratio=8.0),
        )
        assert vm.name == "db01"
        assert vm.cpu == pytest.approx(1.0)  # 8 / 8
        assert vm.memory_mb == pytest.approx(32768.0)  # 32768 / 1.0

    def test_cpu_overcommit_4_to_1(self) -> None:
        vm = normalize_vm(
            name="app01",
            raw_cpu=4.0,
            raw_memory_mb=8192.0,
            overcommit=OvercommitConfig(cpu_ratio=4.0),
        )
        assert vm.cpu == pytest.approx(1.0)  # 4 / 4

    def test_no_memory_overcommit_by_default(self) -> None:
        vm = normalize_vm(
            name="web01",
            raw_cpu=2.0,
            raw_memory_mb=4096.0,
            overcommit=OvercommitConfig(),
        )
        assert vm.memory_mb == pytest.approx(4096.0)

    def test_memory_overcommit_1_5(self) -> None:
        vm = normalize_vm(
            name="web01",
            raw_cpu=2.0,
            raw_memory_mb=3072.0,
            overcommit=OvercommitConfig(memory_ratio=1.5),
        )
        assert vm.memory_mb == pytest.approx(2048.0)  # 3072 / 1.5

    def test_vm_is_frozen(self) -> None:
        vm = normalize_vm(
            name="db01",
            raw_cpu=8.0,
            raw_memory_mb=32768.0,
            overcommit=OvercommitConfig(),
        )
        with pytest.raises(Exception):  # noqa: B017
            vm.cpu = 999.0  # type: ignore[misc]

    def test_pods_default_to_one(self) -> None:
        vm = normalize_vm(
            name="vm01",
            raw_cpu=4.0,
            raw_memory_mb=8192.0,
            overcommit=OvercommitConfig(),
        )
        assert vm.pods == 1


# ═══════════════════════════════════════════════════════════════════════
# 2. MCO Kubelet CPU Reservation
# ═══════════════════════════════════════════════════════════════════════


class TestKubeletReservedCPU:
    """Step function: 6% first 1c, 1% next 1c, 0.5% next 2c, 0.25% rest."""

    def test_1_core(self) -> None:
        # 1 * 0.06 = 0.06
        assert kubelet_reserved_cpu(1.0) == pytest.approx(0.06)

    def test_2_cores(self) -> None:
        # 0.06 + 1 * 0.01 = 0.07
        assert kubelet_reserved_cpu(2.0) == pytest.approx(0.07)

    def test_4_cores(self) -> None:
        # 0.06 + 0.01 + 2 * 0.005 = 0.08
        assert kubelet_reserved_cpu(4.0) == pytest.approx(0.08)

    def test_8_cores(self) -> None:
        # 0.08 + 4 * 0.0025 = 0.09
        assert kubelet_reserved_cpu(8.0) == pytest.approx(0.09)

    def test_64_cores(self) -> None:
        # 0.08 + 60 * 0.0025 = 0.08 + 0.15 = 0.23
        assert kubelet_reserved_cpu(64.0) == pytest.approx(0.23)

    def test_96_cores(self) -> None:
        # 0.08 + 92 * 0.0025 = 0.08 + 0.23 = 0.31
        assert kubelet_reserved_cpu(96.0) == pytest.approx(0.31)

    def test_128_cores(self) -> None:
        # 0.08 + 124 * 0.0025 = 0.08 + 0.31 = 0.39
        assert kubelet_reserved_cpu(128.0) == pytest.approx(0.39)

    def test_zero_cores(self) -> None:
        assert kubelet_reserved_cpu(0.0) == pytest.approx(0.0)

    def test_fractional_cores(self) -> None:
        # 0.5 * 0.06 = 0.03
        assert kubelet_reserved_cpu(0.5) == pytest.approx(0.03)


# ═══════════════════════════════════════════════════════════════════════
# 3. MCO Kubelet Memory Reservation
# ═══════════════════════════════════════════════════════════════════════


class TestKubeletReservedMemory:
    """Step function: 25% first 4G, 20% next 4G, 10% next 8G, 6% next 112G, 2% rest."""

    def test_64_gib(self) -> None:
        # 4*0.25 + 4*0.20 + 8*0.10 + 48*0.06 = 1.0+0.8+0.8+2.88 = 5.48 GiB
        total_mb = 64 * 1024.0
        expected_mb = 5.48 * 1024.0  # 5,611.52
        assert kubelet_reserved_memory_mb(total_mb) == pytest.approx(expected_mb)

    def test_256_gib(self) -> None:
        # 1.0 + 0.8 + 0.8 + 112*0.06 + 128*0.02 = 9.32 + 2.56 = 11.88 GiB
        total_mb = 256 * 1024.0
        expected_mb = 11.88 * 1024.0  # 12,165.12
        assert kubelet_reserved_memory_mb(total_mb) == pytest.approx(expected_mb)

    def test_512_gib(self) -> None:
        # 1.0 + 0.8 + 0.8 + 6.72 + 384*0.02 = 9.32 + 7.68 = 17.0 GiB
        total_mb = 512 * 1024.0
        expected_mb = 17.0 * 1024.0  # 17,408.0
        assert kubelet_reserved_memory_mb(total_mb) == pytest.approx(expected_mb)

    def test_1024_gib(self) -> None:
        # 1.0 + 0.8 + 0.8 + 6.72 + 896*0.02 = 9.32 + 17.92 = 27.24 GiB
        total_mb = 1024 * 1024.0
        expected_mb = 27.24 * 1024.0  # 27,893.76
        assert kubelet_reserved_memory_mb(total_mb) == pytest.approx(expected_mb)

    def test_4_gib(self) -> None:
        # 4 * 0.25 = 1.0 GiB
        total_mb = 4 * 1024.0
        expected_mb = 1.0 * 1024.0
        assert kubelet_reserved_memory_mb(total_mb) == pytest.approx(expected_mb)

    def test_16_gib(self) -> None:
        # 1.0 + 0.8 + 0.8 = 2.6 GiB
        total_mb = 16 * 1024.0
        expected_mb = 2.6 * 1024.0
        assert kubelet_reserved_memory_mb(total_mb) == pytest.approx(expected_mb)

    def test_zero_memory(self) -> None:
        assert kubelet_reserved_memory_mb(0.0) == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════════
# 4. Effective CPU (HT Adjustment)
# ═══════════════════════════════════════════════════════════════════════


class TestComputeEffectiveCPU:
    """Stage 1 — HT efficiency factor application."""

    def test_no_ht_returns_physical_cores(self) -> None:
        # 2 sockets × 24 cores, no HT → 48
        topo = _topo(sockets=2, cores_per_socket=24, threads_per_core=1)
        result = compute_effective_cpu(topo, _overheads())
        assert result == pytest.approx(48.0)

    def test_ht_default_factor_1_5(self) -> None:
        # 2s × 32c × 2t → physical_cores=64, effective = 64 × 1.5 = 96
        topo = _topo(sockets=2, cores_per_socket=32, threads_per_core=2)
        result = compute_effective_cpu(topo, _overheads(ht_efficiency_factor=1.5))
        assert result == pytest.approx(96.0)

    def test_ht_factor_2_0_counts_all_threads(self) -> None:
        # physical=64, 64 × 2.0 = 128 (same as logical_cpus)
        topo = _topo(sockets=2, cores_per_socket=32, threads_per_core=2)
        result = compute_effective_cpu(topo, _overheads(ht_efficiency_factor=2.0))
        assert result == pytest.approx(128.0)

    def test_ht_factor_1_0_physical_cores_only(self) -> None:
        # physical=64, 64 × 1.0 = 64
        topo = _topo(sockets=2, cores_per_socket=32, threads_per_core=2)
        result = compute_effective_cpu(topo, _overheads(ht_efficiency_factor=1.0))
        assert result == pytest.approx(64.0)

    def test_single_socket_no_ht(self) -> None:
        topo = _topo(sockets=1, cores_per_socket=16, threads_per_core=1)
        result = compute_effective_cpu(topo, _overheads())
        assert result == pytest.approx(16.0)


# ═══════════════════════════════════════════════════════════════════════
# 5. Usable Capacity (Overhead Subtraction)
# ═══════════════════════════════════════════════════════════════════════


class TestComputeUsableCapacity:
    """Stage 2 — kubelet + ocp_virt overhead subtraction."""

    def test_typical_server_2s_32c_ht_512g(self) -> None:
        """2s × 32c × 2t, 512 GiB, default overheads."""
        topo = _topo(sockets=2, cores_per_socket=32, threads_per_core=2)
        cpu, mem = compute_usable_capacity(
            topology=topo,
            total_memory_mb=512 * 1024.0,
            overheads=VirtOverheads(),
        )
        # effective_cpu = 64 × 1.5 = 96
        # kubelet_cpu = 0.08 + 92*0.0025 = 0.31
        # usable_cpu = 96 - 0.31 - 2.0 = 93.69
        assert cpu == pytest.approx(93.69)

        # total_mem = 512 * 1024 = 524,288 MB
        # kubelet_mem = 17.0 * 1024 = 17,408 MB
        # usable_mem = 524,288 - 17,408 - 100 - 360 = 506,420 MB
        assert mem == pytest.approx(506_420.0)

    def test_small_server_1s_16c_noht_64g(self) -> None:
        """1s × 16c × 1t, 64 GiB, default overheads."""
        topo = _topo(sockets=1, cores_per_socket=16, threads_per_core=1)
        cpu, mem = compute_usable_capacity(
            topology=topo,
            total_memory_mb=64 * 1024.0,
            overheads=VirtOverheads(),
        )
        # effective_cpu = 16 (no HT)
        # kubelet_cpu = 0.08 + 12*0.0025 = 0.11
        # usable_cpu = 16 - 0.11 - 2.0 = 13.89
        assert cpu == pytest.approx(13.89)

        # total_mem = 64 * 1024 = 65,536 MB
        # kubelet_mem = 5.48 * 1024 = 5,611.52 MB
        # usable_mem = 65,536 - 5,611.52 - 100 - 360 = 59,464.48 MB
        assert mem == pytest.approx(59_464.48)

    def test_rejects_negative_usable_cpu(self) -> None:
        """Node too small to survive overhead subtraction."""
        topo = _topo(sockets=1, cores_per_socket=1, threads_per_core=1)
        with pytest.raises(ValueError, match=r"Negative usable CPU"):
            compute_usable_capacity(
                topology=topo,
                total_memory_mb=512 * 1024.0,
                overheads=VirtOverheads(ocp_virt_core=10.0),  # huge overhead
            )

    def test_rejects_negative_usable_memory(self) -> None:
        """Node too small to survive overhead subtraction."""
        topo = _topo(sockets=2, cores_per_socket=16, threads_per_core=1)
        with pytest.raises(ValueError, match=r"Negative usable memory"):
            compute_usable_capacity(
                topology=topo,
                total_memory_mb=1024.0,  # 1 GiB = 1024 MB, less than overheads
                overheads=VirtOverheads(ocp_virt_memory_mb=900.0, eviction_hard_mb=500.0),
            )


# ═══════════════════════════════════════════════════════════════════════
# 6. Full Node Normalization Pipeline
# ═══════════════════════════════════════════════════════════════════════


class TestNormalizeNodeCapacity:
    """End-to-end: overheads + safety margins + pod limit."""

    def test_default_config_2s_32c_ht_512g(self) -> None:
        """Standard server with default PlanConfig."""
        topo = _topo(sockets=2, cores_per_socket=32, threads_per_core=2)
        config = PlanConfig()
        cpu, mem, pods = normalize_node_capacity(
            topology=topo,
            total_memory_mb=512 * 1024.0,
            config=config,
        )
        # usable_cpu = 93.69 (from TestComputeUsableCapacity)
        # schedulable_cpu = 93.69 × 0.85 = 79.6365
        assert cpu == pytest.approx(93.69 * 0.85)

        # usable_mem = 506,420.0 MB
        # schedulable_mem = 506,420.0 × 0.80 = 405,136.0
        assert mem == pytest.approx(506_420.0 * 0.80)

        # Pods from default cluster_limits
        assert pods == 250

    def test_custom_utilization_targets(self) -> None:
        """Verify custom utilization targets are applied."""
        topo = _topo(sockets=2, cores_per_socket=32, threads_per_core=2)
        config = PlanConfig(
            safety_margins={  # type: ignore[arg-type]
                "utilization_targets": {"cpu": 100, "memory": 100},
            },
        )
        cpu, mem, pods = normalize_node_capacity(
            topology=topo,
            total_memory_mb=512 * 1024.0,
            config=config,
        )
        # 100% utilization = no safety margin
        assert cpu == pytest.approx(93.69)
        assert mem == pytest.approx(506_420.0)
        assert pods == 250

    def test_custom_pod_limit(self) -> None:
        topo = _topo(sockets=2, cores_per_socket=32, threads_per_core=2)
        config = PlanConfig(cluster_limits={"max_pods_per_node": 110})  # type: ignore[arg-type]
        _, _, pods = normalize_node_capacity(
            topology=topo, total_memory_mb=512 * 1024.0, config=config
        )
        assert pods == 110


# ═══════════════════════════════════════════════════════════════════════
# 7. Build Inventory Nodes
# ═══════════════════════════════════════════════════════════════════════


class TestBuildInventoryNodes:
    """Inventory node creation from profiles."""

    def test_single_profile_quantity_3(self) -> None:
        inv = InventoryConfig(
            profiles=[
                InventoryProfile(
                    profile_name="r740-existing",
                    cpu_topology=_topo(sockets=2, cores_per_socket=24, threads_per_core=1),
                    ram_gb=512,
                    quantity=3,
                )
            ]
        )
        nodes = build_inventory_nodes(inv, PlanConfig())
        assert len(nodes) == 3
        assert nodes[0].id == "r740-existing-01"
        assert nodes[1].id == "r740-existing-02"
        assert nodes[2].id == "r740-existing-03"

        # All nodes identical capacity
        assert all(n.profile == "r740-existing" for n in nodes)
        assert all(n.is_inventory for n in nodes)
        assert all(n.cost_weight == 0.0 for n in nodes)
        assert all(n.cpu_total > 0 for n in nodes)
        assert all(n.memory_total > 0 for n in nodes)
        assert all(n.pods_total == 250 for n in nodes)

    def test_multiple_profiles(self) -> None:
        inv = InventoryConfig(
            profiles=[
                InventoryProfile(
                    profile_name="small",
                    cpu_topology=_topo(1, 16, 1),
                    ram_gb=64,
                    quantity=2,
                ),
                InventoryProfile(
                    profile_name="large",
                    cpu_topology=_topo(2, 32, 2),
                    ram_gb=1024,
                    quantity=1,
                ),
            ]
        )
        nodes = build_inventory_nodes(inv, PlanConfig())
        assert len(nodes) == 3
        assert nodes[0].id == "small-01"
        assert nodes[1].id == "small-02"
        assert nodes[2].id == "large-01"

        # Large node should have more resources
        assert nodes[2].cpu_total > nodes[0].cpu_total
        assert nodes[2].memory_total > nodes[0].memory_total

    def test_empty_inventory(self) -> None:
        inv = InventoryConfig(profiles=[])
        nodes = build_inventory_nodes(inv, PlanConfig())
        assert nodes == []


# ═══════════════════════════════════════════════════════════════════════
# 8. Build Catalog Node
# ═══════════════════════════════════════════════════════════════════════


class TestBuildCatalogNode:
    """Catalog node creation for Expander."""

    def test_basic_catalog_node(self) -> None:
        profile = CatalogProfile(
            profile_name="r760-new",
            cpu_topology=_topo(2, 32, 2),
            ram_gb=1024,
            cost_weight=1.0,
        )
        node = build_catalog_node(profile, index=1, config=PlanConfig())

        assert node.id == "r760-new-01"
        assert node.profile == "r760-new"
        assert node.is_inventory is False
        assert node.cost_weight == 1.0
        assert node.cpu_total > 0
        assert node.memory_total > 0
        assert node.pods_total == 250

        # Usage should be zero (fresh node)
        assert node.cpu_used == 0.0
        assert node.memory_used == 0.0
        assert node.pods_used == 0

    def test_catalog_index_increments(self) -> None:
        profile = CatalogProfile(
            profile_name="r760-new",
            cpu_topology=_topo(2, 32, 2),
            ram_gb=1024,
            cost_weight=1.0,
        )
        n1 = build_catalog_node(profile, index=1, config=PlanConfig())
        n2 = build_catalog_node(profile, index=2, config=PlanConfig())
        assert n1.id == "r760-new-01"
        assert n2.id == "r760-new-02"
        # Same profile → same capacity
        assert n1.cpu_total == n2.cpu_total
        assert n1.memory_total == n2.memory_total

    def test_catalog_node_with_high_cost(self) -> None:
        profile = CatalogProfile(
            profile_name="expensive",
            cpu_topology=_topo(4, 64, 2),
            ram_gb=2048,
            cost_weight=5.0,
        )
        node = build_catalog_node(profile, index=1, config=PlanConfig())
        assert node.cost_weight == 5.0
