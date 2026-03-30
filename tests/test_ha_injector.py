"""Tests for core.ha_injector — HA spare-capacity injection (HLD §7)."""

from __future__ import annotations

from core.cluster_state import ClusterState
from core.ha_injector import (
    HARequirement,
    HAResult,
    _simulate_failure,
    compute_current_spare,
    compute_ha_deficit,
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
# _simulate_failure
# ═══════════════════════════════════════════════════════════════════════


class TestSimulateFailure:
    """Tests for the greedy first-fit re-placement simulation."""

    def test_no_displaced_vms(self) -> None:
        """No VMs to re-place → zero deficit."""
        n1 = _inv_node(index=1)
        assert _simulate_failure([n1], []) == (0.0, 0.0)

    def test_all_vms_fit(self) -> None:
        """Survivors have plenty of room → zero deficit."""
        n1 = _inv_node(index=1, cpu_total=100.0, memory_total=400_000.0)
        vms = [_vm("v1", cpu=5.0, memory_mb=10_000.0)]
        assert _simulate_failure([n1], vms) == (0.0, 0.0)

    def test_no_survivors(self) -> None:
        """No surviving nodes → all VMs are unplaced."""
        vms = [_vm("v1", cpu=5.0, memory_mb=10_000.0)]
        d_cpu, d_mem = _simulate_failure([], vms)
        assert d_cpu == 5.0
        assert d_mem == 10_000.0

    def test_partial_fit(self) -> None:
        """Some VMs fit, others don't."""
        n1 = _inv_node(index=1, cpu_total=10.0, memory_total=50_000.0)
        vms = [
            _vm("big", cpu=8.0, memory_mb=40_000.0),
            _vm("small", cpu=5.0, memory_mb=20_000.0),
        ]
        d_cpu, d_mem = _simulate_failure([n1], vms)
        assert d_cpu == 5.0
        assert d_mem == 20_000.0

    def test_stranded_cpu_not_counted(self) -> None:
        """CPU free but memory full → VMs cannot land there."""
        n1 = _inv_node(index=1, cpu_total=100.0, memory_total=50_000.0)
        state = ClusterState([n1])
        state.place(_vm("filler", cpu=10.0, memory_mb=50_000.0), n1)

        displaced = [_vm("v1", cpu=5.0, memory_mb=10_000.0)]
        d_cpu, d_mem = _simulate_failure([n1], displaced)
        assert d_cpu == 5.0
        assert d_mem == 10_000.0

    def test_stranded_mem_not_counted(self) -> None:
        """Memory free but CPU full → VMs cannot land there."""
        n1 = _inv_node(index=1, cpu_total=10.0, memory_total=400_000.0)
        state = ClusterState([n1])
        state.place(_vm("filler", cpu=10.0, memory_mb=1000.0), n1)

        displaced = [_vm("v1", cpu=2.0, memory_mb=5_000.0)]
        d_cpu, d_mem = _simulate_failure([n1], displaced)
        assert d_cpu == 2.0
        assert d_mem == 5_000.0

    def test_spreads_across_multiple_survivors(self) -> None:
        """VMs spread across multiple survivors when one fills up."""
        n1 = _inv_node(index=1, cpu_total=10.0, memory_total=50_000.0)
        n2 = _inv_node(index=2, cpu_total=10.0, memory_total=50_000.0)
        vms = [
            _vm("v1", cpu=8.0, memory_mb=40_000.0),
            _vm("v2", cpu=8.0, memory_mb=40_000.0),
        ]
        d_cpu, d_mem = _simulate_failure([n1, n2], vms)
        assert d_cpu == 0.0
        assert d_mem == 0.0


# ═══════════════════════════════════════════════════════════════════════
# compute_ha_deficit
# ═══════════════════════════════════════════════════════════════════════


class TestComputeHADeficit:
    """Tests for the simulation-based HA deficit computation."""

    def test_zero_failures(self) -> None:
        state = ClusterState([_inv_node()])
        assert compute_ha_deficit(state, 0) == (0.0, 0.0)

    def test_no_active_nodes(self) -> None:
        state = ClusterState([_inv_node()])
        assert compute_ha_deficit(state, 1) == (0.0, 0.0)

    def test_single_failure_sufficient_spare(self) -> None:
        """Two large nodes, one lightly loaded → surviving node absorbs VMs."""
        n1 = _inv_node(index=1, cpu_total=100.0, memory_total=400_000.0)
        n2 = _inv_node(index=2, cpu_total=100.0, memory_total=400_000.0)
        state = ClusterState([n1, n2])
        state.place(_vm("small", cpu=5.0, memory_mb=10_000.0), n1)

        assert compute_ha_deficit(state, 1) == (0.0, 0.0)

    def test_single_failure_deficit(self) -> None:
        """Single tight node → failure loses everything, no survivors absorb."""
        n1 = _inv_node(index=1, cpu_total=50.0, memory_total=100_000.0)
        state = ClusterState([n1])
        state.place(_vm("big", cpu=45.0, memory_mb=90_000.0), n1)

        d_cpu, d_mem = compute_ha_deficit(state, 1)
        assert d_cpu == 45.0
        assert d_mem == 90_000.0

    def test_two_failures_worst_case(self) -> None:
        """N=2: checks all pairs, returns worst scenario."""
        n1 = _inv_node(index=1, cpu_total=50.0, memory_total=100_000.0)
        n2 = _inv_node(index=2, cpu_total=50.0, memory_total=100_000.0)
        n3 = _inv_node(index=3, cpu_total=50.0, memory_total=100_000.0)
        state = ClusterState([n1, n2, n3])
        state.place(_vm("a", cpu=40.0, memory_mb=90_000.0), n1)
        state.place(_vm("b", cpu=35.0, memory_mb=80_000.0), n2)
        state.place(_vm("c", cpu=5.0, memory_mb=10_000.0), n3)

        d_cpu, d_mem = compute_ha_deficit(state, 2)
        assert d_cpu > 0.0 or d_mem > 0.0

    def test_empty_ha_node_provides_capacity(self) -> None:
        """An empty HA node in the state provides absorption capacity."""
        n1 = _inv_node(index=1, cpu_total=50.0, memory_total=100_000.0)
        n_ha = _inv_node(index=2, cpu_total=100.0, memory_total=400_000.0)
        state = ClusterState([n1, n_ha])
        state.place(_vm("big", cpu=45.0, memory_mb=90_000.0), n1)

        assert compute_ha_deficit(state, 1) == (0.0, 0.0)

    def test_stranded_capacity_causes_deficit(self) -> None:
        """Regression: stranded CPU on memory-full node must NOT satisfy HA.

        Old code summed spare independently → reported 0 deficit.
        Simulation detects the VM cannot actually be rescheduled.
        """
        n1 = _inv_node(index=1, cpu_total=100.0, memory_total=100_000.0)
        n2 = _inv_node(index=2, cpu_total=100.0, memory_total=100_000.0)
        state = ClusterState([n1, n2])
        state.place(_vm("cpu-light-mem-heavy", cpu=10.0, memory_mb=90_000.0), n1)
        state.place(_vm("fills-n2-mem", cpu=10.0, memory_mb=95_000.0), n2)

        d_cpu, d_mem = compute_ha_deficit(state, 1)
        assert d_cpu > 0.0 or d_mem > 0.0


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
        """No catalog and no unused pool → deficit is reported, no nodes added."""
        n1 = _inv_node(index=1, cpu_total=50.0, memory_total=100_000.0)
        state = ClusterState([n1])
        state.place(_vm("big", cpu=45.0, memory_mb=90_000.0), n1)

        result = inject_ha_nodes(
            state=state,
            config=_config(ha_failures=1),
            catalog=None,
        )
        assert result.nodes_added == []
        assert result.nodes_reclaimed == []
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


# ═══════════════════════════════════════════════════════════════════════
# Unused pool reclamation (consolidate mode HA)
# ═══════════════════════════════════════════════════════════════════════


class TestUnusedPoolReclamation:
    """HA should reclaim unused inventory nodes before buying catalog nodes."""

    def test_reclaims_from_pool_covers_deficit(self) -> None:
        """Unused pool has enough capacity → reclaim, no catalog needed."""
        # Single active node, nearly full
        n1 = _inv_node(index=1, cpu_total=50.0, memory_total=100_000.0)
        state = ClusterState([n1])
        state.place(_vm("big", cpu=45.0, memory_mb=90_000.0), n1)

        # 2 unused nodes in the pool (could be shut down)
        spare1 = _inv_node(index=10, cpu_total=100.0, memory_total=400_000.0)
        spare2 = _inv_node(index=11, cpu_total=80.0, memory_total=300_000.0)
        pool = [spare1, spare2]

        result = inject_ha_nodes(
            state=state,
            config=_config(ha_failures=1),
            catalog=None,  # no catalog!
            unused_pool=pool,
        )

        assert result.fully_covered is True
        assert result.nodes_added == []  # no catalog nodes
        assert len(result.nodes_reclaimed) >= 1
        # Pool should be mutated (reclaimed nodes removed)
        assert len(pool) < 2

    def test_reclaimed_nodes_added_to_state(self) -> None:
        """Reclaimed nodes must appear in the cluster state."""
        n1 = _inv_node(index=1, cpu_total=50.0, memory_total=100_000.0)
        state = ClusterState([n1])
        state.place(_vm("big", cpu=45.0, memory_mb=90_000.0), n1)

        spare = _inv_node(index=10, cpu_total=100.0, memory_total=400_000.0)
        pool = [spare]
        initial_count = len(state.nodes)

        result = inject_ha_nodes(
            state=state,
            config=_config(ha_failures=1),
            unused_pool=pool,
        )

        assert result.fully_covered is True
        assert len(state.nodes) == initial_count + len(result.nodes_reclaimed)
        # The reclaimed node should be in the state
        assert spare in state.nodes

    def test_reclaims_largest_first(self) -> None:
        """Reclamation should prefer the largest nodes to minimise count."""
        n1 = _inv_node(index=1, cpu_total=50.0, memory_total=100_000.0)
        state = ClusterState([n1])
        state.place(_vm("big", cpu=45.0, memory_mb=90_000.0), n1)

        # Small node won't cover the memory deficit alone
        small = _inv_node(index=10, cpu_total=20.0, memory_total=50_000.0)
        big = _inv_node(index=11, cpu_total=100.0, memory_total=400_000.0)
        pool = [small, big]

        result = inject_ha_nodes(
            state=state,
            config=_config(ha_failures=1),
            unused_pool=pool,
        )

        assert result.fully_covered is True
        # Big node should be reclaimed first
        assert big in result.nodes_reclaimed

    def test_pool_plus_catalog_hybrid(self) -> None:
        """Pool partially covers → catalog fills the rest."""
        n1 = _inv_node(index=1, cpu_total=50.0, memory_total=100_000.0)
        state = ClusterState([n1])
        state.place(_vm("big", cpu=45.0, memory_mb=90_000.0), n1)

        # Small pool node — not enough on its own
        small = _inv_node(index=10, cpu_total=10.0, memory_total=10_000.0)
        pool = [small]

        result = inject_ha_nodes(
            state=state,
            config=_config(ha_failures=1),
            catalog=_catalog(),
            unused_pool=pool,
        )

        assert result.fully_covered is True
        assert len(result.nodes_reclaimed) == 1  # the small node
        assert len(result.nodes_added) >= 1  # catalog fills the gap
        assert pool == []  # pool fully drained

    def test_empty_pool_falls_through_to_catalog(self) -> None:
        """Empty unused pool → behaves like the original (catalog only)."""
        n1 = _inv_node(index=1, cpu_total=50.0, memory_total=100_000.0)
        state = ClusterState([n1])
        state.place(_vm("big", cpu=45.0, memory_mb=90_000.0), n1)

        result = inject_ha_nodes(
            state=state,
            config=_config(ha_failures=1),
            catalog=_catalog(),
            unused_pool=[],
        )

        assert result.fully_covered is True
        assert result.nodes_reclaimed == []
        assert len(result.nodes_added) >= 1

    def test_pool_reduces_shutdown_candidates(self) -> None:
        """After reclamation, the pool length == actual shutdown candidates."""
        n1 = _inv_node(index=1, cpu_total=50.0, memory_total=100_000.0)
        state = ClusterState([n1])
        state.place(_vm("big", cpu=45.0, memory_mb=90_000.0), n1)

        spare1 = _inv_node(index=10, cpu_total=100.0, memory_total=400_000.0)
        spare2 = _inv_node(index=11, cpu_total=80.0, memory_total=300_000.0)
        spare3 = _inv_node(index=12, cpu_total=60.0, memory_total=200_000.0)
        pool = [spare1, spare2, spare3]

        inject_ha_nodes(
            state=state,
            config=_config(ha_failures=1),
            unused_pool=pool,
        )

        # Some nodes reclaimed, remainder can be shut down
        assert len(pool) < 3  # at least 1 reclaimed
        assert len(pool) >= 0  # some may remain

    def test_stranded_capacity_triggers_reclaim(self) -> None:
        """Regression: stranded CPU on memory-full survivors must trigger HA.

        Scenario: 2 active nodes, both memory-heavy.  If either fails,
        its 90 GB VM cannot land on the other (only 5 GB free).  The old
        independent-sum code saw 180 spare CPU and thought HA was fine.
        The simulation correctly detects the VM cannot be rescheduled.
        """
        n1 = _inv_node(index=1, cpu_total=100.0, memory_total=100_000.0)
        n2 = _inv_node(index=2, cpu_total=100.0, memory_total=100_000.0)
        state = ClusterState([n1, n2])
        state.place(_vm("a", cpu=10.0, memory_mb=95_000.0), n1)
        state.place(_vm("b", cpu=10.0, memory_mb=95_000.0), n2)

        spare = _inv_node(index=10, cpu_total=100.0, memory_total=400_000.0)
        pool = [spare]

        result = inject_ha_nodes(
            state=state,
            config=_config(ha_failures=1),
            unused_pool=pool,
        )

        assert result.fully_covered is True
        assert len(result.nodes_reclaimed) >= 1
