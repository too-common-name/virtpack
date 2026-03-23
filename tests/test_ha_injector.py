"""Tests for core.ha_injector — HA spare-capacity injection (HLD §7)."""

from __future__ import annotations

from core.cluster_state import ClusterState
from core.ha_injector import (
    HARequirement,
    HAResult,
    compute_current_spare,
    compute_ha_requirements,
    inject_ha_nodes,
)
from models.config import (
    CatalogConfig,
    CatalogProfile,
    CpuTopology,
    PlanConfig,
    SafetyMargins,
)
from models.node import Node
from models.vm import VM

# ── Helpers ──────────────────────────────────────────────────────────────


def _topology(sockets: int = 2, cores: int = 32, threads: int = 2) -> CpuTopology:
    return CpuTopology(sockets=sockets, cores_per_socket=cores, threads_per_core=threads)


def _catalog_profile(
    name: str = "r760",
    ram_gb: int = 512,
    cost: float = 1.0,
) -> CatalogProfile:
    return CatalogProfile(
        profile_name=name,
        cpu_topology=_topology(),
        ram_gb=ram_gb,
        cost_weight=cost,
    )


def _catalog(*profiles: CatalogProfile) -> CatalogConfig:
    if not profiles:
        profiles = (_catalog_profile(),)
    return CatalogConfig(profiles=list(profiles))


def _config(ha_failures: int = 1) -> PlanConfig:
    return PlanConfig(safety_margins=SafetyMargins(ha_failures_to_tolerate=ha_failures))


def _inv_node(
    profile: str = "inv",
    index: int = 1,
    cpu_total: float = 100.0,
    memory_total: float = 400_000.0,
    pods_total: int = 250,
) -> Node:
    return Node.new_inventory(
        profile=profile,
        index=index,
        cpu_total=cpu_total,
        memory_total=memory_total,
        pods_total=pods_total,
    )


def _vm(name: str, cpu: float = 2.0, memory_mb: float = 4096.0) -> VM:
    return VM(name=name, cpu=cpu, memory_mb=memory_mb)


def _place_vm(state: ClusterState, vm: VM, node: Node) -> None:
    """Helper: place a VM on a node."""
    state.place(vm, node)


# ═══════════════════════════════════════════════════════════════════════
# HARequirement / HAResult containers
# ═══════════════════════════════════════════════════════════════════════


class TestContainers:
    def test_ha_requirement_fields(self) -> None:
        r = HARequirement(required_spare_cpu=10.0, required_spare_memory=5000.0)
        assert r.required_spare_cpu == 10.0
        assert r.required_spare_memory == 5000.0

    def test_ha_result_defaults(self) -> None:
        r = HAResult()
        assert r.nodes_added == []
        assert r.deficit_cpu == 0.0
        assert r.deficit_memory == 0.0
        assert r.fully_covered is True

    def test_ha_result_deficit(self) -> None:
        r = HAResult(deficit_cpu=5.0, deficit_memory=1000.0)
        assert r.fully_covered is False


# ═══════════════════════════════════════════════════════════════════════
# compute_ha_requirements
# ═══════════════════════════════════════════════════════════════════════


class TestComputeHARequirements:
    def test_zero_failures_returns_zeros(self) -> None:
        state = ClusterState([_inv_node()])
        req = compute_ha_requirements(state, 0)
        assert req.required_spare_cpu == 0.0
        assert req.required_spare_memory == 0.0

    def test_negative_failures_returns_zeros(self) -> None:
        state = ClusterState([_inv_node()])
        req = compute_ha_requirements(state, -1)
        assert req.required_spare_cpu == 0.0
        assert req.required_spare_memory == 0.0

    def test_empty_cluster_returns_zeros(self) -> None:
        state = ClusterState()
        req = compute_ha_requirements(state, 1)
        assert req.required_spare_cpu == 0.0
        assert req.required_spare_memory == 0.0

    def test_no_active_nodes_returns_zeros(self) -> None:
        """Nodes exist but nothing placed → no active nodes."""
        state = ClusterState([_inv_node()])
        req = compute_ha_requirements(state, 1)
        assert req.required_spare_cpu == 0.0
        assert req.required_spare_memory == 0.0

    def test_single_failure_single_node(self) -> None:
        node = _inv_node()
        state = ClusterState([node])
        vm = _vm("vm1", cpu=20.0, memory_mb=50_000.0)
        state.place(vm, node)

        req = compute_ha_requirements(state, 1)
        assert req.required_spare_cpu == 20.0
        assert req.required_spare_memory == 50_000.0

    def test_single_failure_picks_heaviest(self) -> None:
        """N=1: picks the single node with the most used in each dimension."""
        n1 = _inv_node(index=1)
        n2 = _inv_node(index=2)
        state = ClusterState([n1, n2])

        # n1: 30 CPU used, 80_000 MB used
        state.place(_vm("heavy-cpu", cpu=30.0, memory_mb=80_000.0), n1)
        # n2: 10 CPU used, 120_000 MB used
        state.place(_vm("heavy-mem", cpu=10.0, memory_mb=120_000.0), n2)

        req = compute_ha_requirements(state, 1)
        assert req.required_spare_cpu == 30.0  # from n1
        assert req.required_spare_memory == 120_000.0  # from n2

    def test_two_failures_sums_top_two(self) -> None:
        """N=2: sums the two most loaded nodes per dimension."""
        n1 = _inv_node(index=1)
        n2 = _inv_node(index=2)
        n3 = _inv_node(index=3)
        state = ClusterState([n1, n2, n3])

        state.place(_vm("a", cpu=30.0, memory_mb=100_000.0), n1)
        state.place(_vm("b", cpu=20.0, memory_mb=80_000.0), n2)
        state.place(_vm("c", cpu=10.0, memory_mb=60_000.0), n3)

        req = compute_ha_requirements(state, 2)
        # Top-2 CPU: 30 + 20 = 50
        assert req.required_spare_cpu == 50.0
        # Top-2 memory: 100k + 80k = 180k
        assert req.required_spare_memory == 180_000.0

    def test_n_exceeds_active_clamps(self) -> None:
        """If N > active nodes, clamp to the actual count."""
        n1 = _inv_node(index=1)
        state = ClusterState([n1])
        state.place(_vm("a", cpu=10.0, memory_mb=50_000.0), n1)

        req = compute_ha_requirements(state, 5)
        assert req.required_spare_cpu == 10.0
        assert req.required_spare_memory == 50_000.0


# ═══════════════════════════════════════════════════════════════════════
# compute_current_spare
# ═══════════════════════════════════════════════════════════════════════


class TestComputeCurrentSpare:
    def test_empty_cluster(self) -> None:
        state = ClusterState()
        assert compute_current_spare(state) == (0.0, 0.0)

    def test_unused_nodes_full_spare(self) -> None:
        n1 = _inv_node(index=1, cpu_total=100.0, memory_total=400_000.0)
        n2 = _inv_node(index=2, cpu_total=80.0, memory_total=300_000.0)
        state = ClusterState([n1, n2])
        spare_cpu, spare_mem = compute_current_spare(state)
        assert spare_cpu == 180.0
        assert spare_mem == 700_000.0

    def test_partial_usage(self) -> None:
        n1 = _inv_node(index=1, cpu_total=100.0, memory_total=400_000.0)
        state = ClusterState([n1])
        state.place(_vm("a", cpu=40.0, memory_mb=100_000.0), n1)
        spare_cpu, spare_mem = compute_current_spare(state)
        assert spare_cpu == 60.0
        assert spare_mem == 300_000.0


# ═══════════════════════════════════════════════════════════════════════
# inject_ha_nodes
# ═══════════════════════════════════════════════════════════════════════


class TestInjectHANodes:
    def test_ha_zero_no_injection(self) -> None:
        """ha_failures_to_tolerate=0 → no injection."""
        state = ClusterState([_inv_node()])
        result = inject_ha_nodes(state=state, config=_config(ha_failures=0))
        assert result.nodes_added == []
        assert result.fully_covered is True

    def test_spare_already_sufficient(self) -> None:
        """Cluster has enough spare → no HA nodes added."""
        # Two nodes with 100 CPU each, one lightly loaded
        n1 = _inv_node(index=1, cpu_total=100.0, memory_total=400_000.0)
        n2 = _inv_node(index=2, cpu_total=100.0, memory_total=400_000.0)
        state = ClusterState([n1, n2])
        # Place a small VM on n1 → n1 used: 5 CPU, 10k MB
        state.place(_vm("small", cpu=5.0, memory_mb=10_000.0), n1)

        # Spare: 195 CPU, 790k MB — easily covers worst case (5 CPU, 10k MB)
        result = inject_ha_nodes(state=state, config=_config(ha_failures=1))
        assert result.nodes_added == []
        assert result.fully_covered is True

    def test_no_active_nodes_no_injection(self) -> None:
        """Empty nodes → nothing to protect → no injection."""
        state = ClusterState([_inv_node()])
        result = inject_ha_nodes(
            state=state,
            config=_config(ha_failures=1),
            catalog=_catalog(),
        )
        assert result.nodes_added == []
        assert result.fully_covered is True

    def test_deficit_adds_catalog_nodes(self) -> None:
        """Single tight node → deficit → HA node added."""
        # One small node nearly full
        n1 = _inv_node(index=1, cpu_total=50.0, memory_total=100_000.0)
        state = ClusterState([n1])
        state.place(_vm("big", cpu=45.0, memory_mb=90_000.0), n1)
        # Spare: 5 CPU, 10k MB
        # Required: 45 CPU, 90k MB → deficit: 40 CPU, 80k MB

        result = inject_ha_nodes(
            state=state,
            config=_config(ha_failures=1),
            catalog=_catalog(),
        )
        assert len(result.nodes_added) >= 1
        assert result.fully_covered is True
        # HA nodes have "ha-" prefix
        assert all(n.id.startswith("ha-") for n in result.nodes_added)

    def test_ha_nodes_added_to_state(self) -> None:
        """Injected HA nodes must be present in the cluster state."""
        n1 = _inv_node(index=1, cpu_total=50.0, memory_total=100_000.0)
        state = ClusterState([n1])
        state.place(_vm("big", cpu=45.0, memory_mb=90_000.0), n1)

        initial_count = len(state.nodes)
        result = inject_ha_nodes(
            state=state,
            config=_config(ha_failures=1),
            catalog=_catalog(),
        )
        assert len(state.nodes) == initial_count + len(result.nodes_added)

    def test_no_catalog_reports_deficit(self) -> None:
        """No catalog → deficit is reported, no nodes added."""
        n1 = _inv_node(index=1, cpu_total=50.0, memory_total=100_000.0)
        state = ClusterState([n1])
        state.place(_vm("big", cpu=45.0, memory_mb=90_000.0), n1)

        result = inject_ha_nodes(
            state=state,
            config=_config(ha_failures=1),
            catalog=None,
        )
        assert result.nodes_added == []
        assert result.fully_covered is False
        assert result.deficit_cpu > 0.0
        assert result.deficit_memory > 0.0

    def test_two_failures_covers_both(self) -> None:
        """N=2: cluster must tolerate loss of 2 heaviest nodes."""
        n1 = _inv_node(index=1, cpu_total=50.0, memory_total=100_000.0)
        n2 = _inv_node(index=2, cpu_total=50.0, memory_total=100_000.0)
        state = ClusterState([n1, n2])
        state.place(_vm("a", cpu=45.0, memory_mb=90_000.0), n1)
        state.place(_vm("b", cpu=40.0, memory_mb=80_000.0), n2)

        # Spare before HA: (5 + 10) = 15 CPU, (10k + 20k) = 30k MB
        # Required N=2: (45 + 40) = 85 CPU, (90k + 80k) = 170k MB
        # Deficit: 70 CPU, 140k MB → multiple HA nodes needed
        result = inject_ha_nodes(
            state=state,
            config=_config(ha_failures=2),
            catalog=_catalog(),
        )
        assert len(result.nodes_added) >= 1
        assert result.fully_covered is True

    def test_ha_node_ids_sequential(self) -> None:
        """HA nodes get sequential IDs: ha-profile-01, ha-profile-02, etc."""
        n1 = _inv_node(index=1, cpu_total=10.0, memory_total=20_000.0)
        state = ClusterState([n1])
        state.place(_vm("a", cpu=9.0, memory_mb=19_000.0), n1)

        result = inject_ha_nodes(
            state=state,
            config=_config(ha_failures=1),
            catalog=_catalog(_catalog_profile(name="r760")),
        )
        assert result.fully_covered is True
        # First HA node should be ha-r760-01
        assert result.nodes_added[0].id == "ha-r760-01"

    def test_picks_cheapest_profile(self) -> None:
        """When multiple profiles exist, the cheapest is used for HA."""
        n1 = _inv_node(index=1, cpu_total=10.0, memory_total=20_000.0)
        state = ClusterState([n1])
        state.place(_vm("a", cpu=9.0, memory_mb=19_000.0), n1)

        expensive = _catalog_profile(name="big", ram_gb=1024, cost=5.0)
        cheap = _catalog_profile(name="small", ram_gb=512, cost=1.0)
        result = inject_ha_nodes(
            state=state,
            config=_config(ha_failures=1),
            catalog=_catalog(expensive, cheap),
        )
        assert result.fully_covered is True
        # All HA nodes should be from the cheap profile
        assert all("small" in n.id for n in result.nodes_added)

    def test_ha_nodes_are_catalog_type(self) -> None:
        """HA nodes are catalog nodes (is_inventory=False)."""
        n1 = _inv_node(index=1, cpu_total=10.0, memory_total=20_000.0)
        state = ClusterState([n1])
        state.place(_vm("a", cpu=9.0, memory_mb=19_000.0), n1)

        result = inject_ha_nodes(
            state=state,
            config=_config(ha_failures=1),
            catalog=_catalog(),
        )
        assert all(not n.is_inventory for n in result.nodes_added)
        assert all(n.cost_weight > 0 for n in result.nodes_added)
