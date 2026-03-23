"""Unit tests for models.config (PlanConfig, Inventory, Catalog)."""

import pytest
from pydantic import ValidationError

from models.config import (
    AlgorithmWeights,
    CatalogConfig,
    CatalogProfile,
    ClusterLimits,
    CpuTopology,
    InventoryConfig,
    InventoryProfile,
    OvercommitConfig,
    PlanConfig,
    SafetyMargins,
    UtilizationTargets,
    VirtOverheads,
)


# ═══════════════════════════════════════════════════════════════════════
# PlanConfig  –  sensible defaults
# ═══════════════════════════════════════════════════════════════════════


class TestPlanConfigDefaults:
    """An empty config.yaml should produce valid defaults (HLD §3.2)."""

    def test_all_defaults(self) -> None:
        cfg = PlanConfig()

        assert cfg.cluster_limits.max_pods_per_node == 250
        assert cfg.overcommit.cpu_ratio == 8.0
        assert cfg.overcommit.memory_ratio == 1.0
        assert cfg.virt_overheads.ocp_virt_core == 2.0
        assert cfg.virt_overheads.ocp_virt_memory_mb == 360.0
        assert cfg.virt_overheads.eviction_hard_mb == 100.0
        assert cfg.safety_margins.utilization_targets.cpu == 85.0
        assert cfg.safety_margins.utilization_targets.memory == 80.0
        assert cfg.safety_margins.ha_failures_to_tolerate == 1

    def test_frozen(self) -> None:
        cfg = PlanConfig()
        with pytest.raises(ValidationError):
            cfg.overcommit = OvercommitConfig(cpu_ratio=4.0)  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════
# AlgorithmWeights  –  sum-to-one validation
# ═══════════════════════════════════════════════════════════════════════


class TestAlgorithmWeights:
    """Weight validation per HLD §3.5."""

    def test_defaults_sum_to_one(self) -> None:
        w = AlgorithmWeights()
        total = w.alpha_balance + w.beta_spread + w.gamma_pod_headroom + w.delta_frag_penalty
        assert total == pytest.approx(1.0)

    def test_custom_weights_summing_to_one(self) -> None:
        w = AlgorithmWeights(
            alpha_balance=0.25,
            beta_spread=0.25,
            gamma_pod_headroom=0.25,
            delta_frag_penalty=0.25,
        )
        assert w.alpha_balance == 0.25

    def test_rejects_weights_over_one(self) -> None:
        with pytest.raises(ValidationError, match="sum to 1.0"):
            AlgorithmWeights(
                alpha_balance=0.5,
                beta_spread=0.5,
                gamma_pod_headroom=0.5,
                delta_frag_penalty=0.5,
            )

    def test_rejects_weights_under_one(self) -> None:
        with pytest.raises(ValidationError, match="sum to 1.0"):
            AlgorithmWeights(
                alpha_balance=0.1,
                beta_spread=0.1,
                gamma_pod_headroom=0.1,
                delta_frag_penalty=0.1,
            )

    def test_rejects_negative_weight(self) -> None:
        with pytest.raises(ValidationError):
            AlgorithmWeights(
                alpha_balance=-0.1,
                beta_spread=0.4,
                gamma_pod_headroom=0.3,
                delta_frag_penalty=0.4,
            )

    def test_frozen(self) -> None:
        w = AlgorithmWeights()
        with pytest.raises(ValidationError):
            w.alpha_balance = 0.5  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════
# UtilizationTargets  –  boundary validation
# ═══════════════════════════════════════════════════════════════════════


class TestUtilizationTargets:
    """Utilization must be (0, 100]."""

    def test_defaults(self) -> None:
        t = UtilizationTargets()
        assert t.cpu == 85.0
        assert t.memory == 80.0

    def test_rejects_zero_cpu(self) -> None:
        with pytest.raises(ValidationError):
            UtilizationTargets(cpu=0)

    def test_rejects_over_100_memory(self) -> None:
        with pytest.raises(ValidationError):
            UtilizationTargets(memory=101)

    def test_accepts_100_percent(self) -> None:
        t = UtilizationTargets(cpu=100, memory=100)
        assert t.cpu == 100.0

    def test_accepts_yaml_int_coercion(self) -> None:
        """YAML produces int 85 for cpu; non-strict config coerces to float."""
        t = UtilizationTargets(cpu=85, memory=80)
        assert isinstance(t.cpu, float)


# ═══════════════════════════════════════════════════════════════════════
# CpuTopology  –  computed field
# ═══════════════════════════════════════════════════════════════════════


class TestCpuTopology:
    """logical_cpus and physical_cores computed fields."""

    def test_no_hyperthreading(self) -> None:
        topo = CpuTopology(sockets=2, cores_per_socket=24, threads_per_core=1)
        assert topo.physical_cores == 48
        assert topo.logical_cpus == 48

    def test_with_hyperthreading(self) -> None:
        topo = CpuTopology(sockets=2, cores_per_socket=32, threads_per_core=2)
        assert topo.physical_cores == 64
        assert topo.logical_cpus == 128

    def test_single_socket(self) -> None:
        topo = CpuTopology(sockets=1, cores_per_socket=8)
        assert topo.physical_cores == 8
        assert topo.logical_cpus == 8

    def test_rejects_zero_sockets(self) -> None:
        with pytest.raises(ValidationError):
            CpuTopology(sockets=0, cores_per_socket=24)

    def test_rejects_zero_cores(self) -> None:
        with pytest.raises(ValidationError):
            CpuTopology(sockets=2, cores_per_socket=0)

    def test_frozen(self) -> None:
        topo = CpuTopology(sockets=2, cores_per_socket=24)
        with pytest.raises(ValidationError):
            topo.sockets = 4  # type: ignore[misc]

    def test_computed_fields_in_serialization(self) -> None:
        """Both computed_fields should appear in model_dump()."""
        topo = CpuTopology(sockets=2, cores_per_socket=16, threads_per_core=2)
        data = topo.model_dump()
        assert data["physical_cores"] == 32
        assert data["logical_cpus"] == 64


# ═══════════════════════════════════════════════════════════════════════
# CatalogConfig  –  requires at least one profile
# ═══════════════════════════════════════════════════════════════════════

_TOPO = CpuTopology(sockets=2, cores_per_socket=32, threads_per_core=2)


class TestCatalogConfig:
    """catalog.yaml must have ≥ 1 profile for greenfield expansion."""

    def test_valid_single_profile(self) -> None:
        cat = CatalogConfig(
            profiles=[CatalogProfile(profile_name="r760", cpu_topology=_TOPO, ram_gb=1024)]
        )
        assert len(cat.profiles) == 1

    def test_rejects_empty_profiles(self) -> None:
        with pytest.raises(ValidationError):
            CatalogConfig(profiles=[])

    def test_rejects_zero_ram(self) -> None:
        with pytest.raises(ValidationError):
            CatalogProfile(profile_name="bad", cpu_topology=_TOPO, ram_gb=0)

    def test_rejects_zero_cost_weight(self) -> None:
        with pytest.raises(ValidationError):
            CatalogProfile(
                profile_name="bad", cpu_topology=_TOPO, ram_gb=512, cost_weight=0.0
            )

    def test_frozen(self) -> None:
        cat = CatalogConfig(
            profiles=[CatalogProfile(profile_name="r760", cpu_topology=_TOPO, ram_gb=1024)]
        )
        with pytest.raises(ValidationError):
            cat.profiles = []  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════
# InventoryConfig  –  allows empty
# ═══════════════════════════════════════════════════════════════════════


class TestInventoryConfig:
    """inventory.yaml is optional (empty = no brownfield hardware)."""

    def test_empty_profiles_allowed(self) -> None:
        inv = InventoryConfig()
        assert inv.profiles == []

    def test_valid_profiles(self) -> None:
        inv = InventoryConfig(
            profiles=[
                InventoryProfile(
                    profile_name="r740",
                    cpu_topology=CpuTopology(sockets=2, cores_per_socket=24),
                    ram_gb=512,
                    quantity=12,
                ),
            ]
        )
        assert len(inv.profiles) == 1
        assert inv.profiles[0].quantity == 12

    def test_default_quantity_is_one(self) -> None:
        p = InventoryProfile(
            profile_name="r740",
            cpu_topology=CpuTopology(sockets=2, cores_per_socket=24),
            ram_gb=256,
        )
        assert p.quantity == 1


# ═══════════════════════════════════════════════════════════════════════
# VirtOverheads / ClusterLimits / SafetyMargins  –  spot checks
# ═══════════════════════════════════════════════════════════════════════


class TestVirtOverheads:

    def test_defaults(self) -> None:
        v = VirtOverheads()
        assert v.ht_efficiency_factor == 1.5
        assert v.ocp_virt_core == 2.0
        assert v.ocp_virt_memory_mb == 360.0
        assert v.eviction_hard_mb == 100.0

    def test_allows_zero_overheads(self) -> None:
        """ge=0: overhead of 0 is valid (e.g. testing without overheads)."""
        v = VirtOverheads(ocp_virt_core=0, ocp_virt_memory_mb=0, eviction_hard_mb=0)
        assert v.ocp_virt_core == 0.0

    def test_ht_factor_custom(self) -> None:
        """User can override HT factor (e.g. 2.0 to count all threads)."""
        v = VirtOverheads(ht_efficiency_factor=2.0)
        assert v.ht_efficiency_factor == 2.0

    def test_ht_factor_physical_only(self) -> None:
        """Factor of 1.0 means ignore HT entirely (physical cores only)."""
        v = VirtOverheads(ht_efficiency_factor=1.0)
        assert v.ht_efficiency_factor == 1.0

    def test_ht_factor_rejects_below_one(self) -> None:
        with pytest.raises(ValidationError):
            VirtOverheads(ht_efficiency_factor=0.5)

    def test_ht_factor_rejects_above_two(self) -> None:
        with pytest.raises(ValidationError):
            VirtOverheads(ht_efficiency_factor=2.5)


class TestClusterLimits:

    def test_rejects_zero_pods(self) -> None:
        with pytest.raises(ValidationError):
            ClusterLimits(max_pods_per_node=0)

    def test_rejects_negative_pods(self) -> None:
        with pytest.raises(ValidationError):
            ClusterLimits(max_pods_per_node=-1)


class TestSafetyMargins:

    def test_defaults(self) -> None:
        s = SafetyMargins()
        assert s.ha_failures_to_tolerate == 1
        assert s.utilization_targets.cpu == 85.0

    def test_ha_zero_allowed(self) -> None:
        """ha=0 is valid (no HA requirement)."""
        s = SafetyMargins(ha_failures_to_tolerate=0)
        assert s.ha_failures_to_tolerate == 0
