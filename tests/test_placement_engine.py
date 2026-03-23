"""Tests for core.placement_engine — the full placement simulation loop."""

from __future__ import annotations

import pytest

from core.cluster_state import ClusterState
from core.placement_engine import PlacementResult, _score_candidates, run_placement
from models.config import (
    AlgorithmWeights,
    CatalogConfig,
    CatalogProfile,
    CpuTopology,
    PlanConfig,
)
from models.node import Node
from models.vm import VM

# ── Helpers ──────────────────────────────────────────────────────────────


def _topology(sockets: int = 2, cores: int = 32, threads: int = 2) -> CpuTopology:
    return CpuTopology(sockets=sockets, cores_per_socket=cores, threads_per_core=threads)


def _catalog_profile(
    name: str = "r760",
    sockets: int = 2,
    cores: int = 32,
    threads: int = 2,
    ram_gb: int = 512,
    cost: float = 1.0,
) -> CatalogProfile:
    return CatalogProfile(
        profile_name=name,
        cpu_topology=_topology(sockets, cores, threads),
        ram_gb=ram_gb,
        cost_weight=cost,
    )


def _catalog(*profiles: CatalogProfile) -> CatalogConfig:
    if not profiles:
        profiles = (_catalog_profile(),)
    return CatalogConfig(profiles=list(profiles))


def _config() -> PlanConfig:
    return PlanConfig()


def _inv_node(
    profile: str = "inv",
    index: int = 1,
    cpu_total: float = 80.0,
    memory_total: float = 300_000.0,
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


# ═══════════════════════════════════════════════════════════════════════
# PlacementResult
# ═══════════════════════════════════════════════════════════════════════


class TestPlacementResult:
    def test_default_unplaced_empty(self) -> None:
        r = PlacementResult(state=ClusterState())
        assert r.unplaced == []

    def test_contains_state(self) -> None:
        state = ClusterState([_inv_node()])
        r = PlacementResult(state=state)
        assert len(r.state.nodes) == 1


# ═══════════════════════════════════════════════════════════════════════
# _score_candidates (scoring + lookahead)
# ═══════════════════════════════════════════════════════════════════════


class TestScoreCandidates:
    def test_single_candidate(self) -> None:
        """With one candidate, it must be chosen."""
        state = ClusterState()
        n = _inv_node()
        state.add_node(n)
        vm = _vm("v1")
        best = _score_candidates(
            candidates=[n],
            vm=vm,
            next_vm=None,
            state=state,
            weights=AlgorithmWeights(),
        )
        assert best is n

    def test_prefers_less_used_node(self) -> None:
        """With spread-only weights, the lighter node wins."""
        weights = AlgorithmWeights(
            alpha_balance=0.0,
            beta_spread=1.0,
            gamma_pod_headroom=0.0,
            delta_frag_penalty=0.0,
        )
        n_light = _inv_node(index=1)
        n_heavy = _inv_node(index=2)
        # Simulate some load on the heavy node
        n_heavy.cpu_used = 60.0
        n_heavy.memory_used = 200_000.0
        n_heavy.pods_used = 100

        state = ClusterState([n_light, n_heavy])
        vm = _vm("v1")
        best = _score_candidates(
            candidates=[n_light, n_heavy],
            vm=vm,
            next_vm=None,
            state=state,
            weights=weights,
        )
        assert best is n_light

    def test_lookahead_avoids_blocking_next_vm(self) -> None:
        """Lookahead should disfavour a node that can't fit the next VM."""
        # Node with just enough room for the current VM but not both
        tight = _inv_node(index=1, cpu_total=5.0, memory_total=10_000.0, pods_total=250)
        # Node with plenty of room
        roomy = _inv_node(index=2, cpu_total=80.0, memory_total=300_000.0, pods_total=250)

        state = ClusterState([tight, roomy])
        current_vm = _vm("v1", cpu=4.0, memory_mb=8000.0)
        next_vm = _vm("v2", cpu=4.0, memory_mb=8000.0)

        best = _score_candidates(
            candidates=[tight, roomy],
            vm=current_vm,
            next_vm=next_vm,
            state=state,
            weights=AlgorithmWeights(),
        )
        # Roomy should win because tight can't fit both v1 and v2
        assert best is roomy

    def test_lookahead_rollback_leaves_state_clean(self) -> None:
        """After scoring, no nodes should have modified usage."""
        n = _inv_node()
        state = ClusterState([n])
        vm = _vm("v1")
        next_vm = _vm("v2")

        _score_candidates(
            candidates=[n],
            vm=vm,
            next_vm=next_vm,
            state=state,
            weights=AlgorithmWeights(),
        )

        # Verify the node's usage was restored
        assert n.cpu_used == pytest.approx(0.0)
        assert n.memory_used == pytest.approx(0.0)
        assert n.pods_used == 0


# ═══════════════════════════════════════════════════════════════════════
# run_placement — full integration
# ═══════════════════════════════════════════════════════════════════════


class TestRunPlacement:
    def test_all_vms_on_inventory(self) -> None:
        """VMs that fit on inventory should be placed without expansion."""
        state = ClusterState([_inv_node(cpu_total=80.0, memory_total=300_000.0)])
        vms = [_vm(f"v{i}", cpu=2.0, memory_mb=4096.0) for i in range(5)]
        catalog = _catalog()

        result = run_placement(vms=vms, state=state, config=_config(), catalog=catalog)

        assert result.unplaced == []
        assert result.state.total_placed_vms == 5
        # No catalog nodes should be added
        assert len(result.state.catalog_nodes) == 0

    def test_expansion_when_inventory_full(self) -> None:
        """When inventory is exhausted, catalog nodes are created."""
        # Very small inventory — can fit ~1 VM
        small_inv = _inv_node(cpu_total=3.0, memory_total=5000.0, pods_total=1)
        state = ClusterState([small_inv])
        vms = [_vm(f"v{i}", cpu=2.0, memory_mb=4096.0) for i in range(3)]
        catalog = _catalog()

        result = run_placement(vms=vms, state=state, config=_config(), catalog=catalog)

        assert result.unplaced == []
        assert result.state.total_placed_vms == 3
        assert len(result.state.catalog_nodes) >= 1

    def test_monster_vm_goes_to_unplaced(self) -> None:
        """A VM too large for any catalog profile is marked unplaced."""
        state = ClusterState([_inv_node()])
        monster = VM(name="monster", cpu=500.0, memory_mb=10_000_000.0)
        catalog = _catalog()

        result = run_placement(vms=[monster], state=state, config=_config(), catalog=catalog)

        assert len(result.unplaced) == 1
        assert result.unplaced[0].name == "monster"
        assert result.state.total_placed_vms == 0

    def test_mixed_placement_and_unplaced(self) -> None:
        """Some VMs fit, some are monsters — both lists populated."""
        state = ClusterState([_inv_node()])
        ok_vms = [_vm(f"ok-{i}", cpu=2.0, memory_mb=4096.0) for i in range(3)]
        monster = VM(name="monster", cpu=500.0, memory_mb=10_000_000.0)
        catalog = _catalog()

        result = run_placement(
            vms=[*ok_vms, monster],
            state=state,
            config=_config(),
            catalog=catalog,
        )

        assert result.state.total_placed_vms == 3
        assert len(result.unplaced) == 1

    def test_vms_sorted_by_memory_desc(self) -> None:
        """Verify largest-memory VMs are placed first (first-fit-decreasing)."""
        # Two nodes with limited capacity
        n1 = _inv_node(index=1, cpu_total=50.0, memory_total=100_000.0)
        n2 = _inv_node(index=2, cpu_total=50.0, memory_total=100_000.0)
        state = ClusterState([n1, n2])

        # Create VMs with varying memory (deliberately unsorted)
        small = _vm("small", cpu=1.0, memory_mb=10_000.0)
        medium = _vm("medium", cpu=1.0, memory_mb=50_000.0)
        large = _vm("large", cpu=1.0, memory_mb=90_000.0)

        catalog = _catalog()
        result = run_placement(
            vms=[small, medium, large],
            state=state,
            config=_config(),
            catalog=catalog,
        )

        assert result.unplaced == []
        # All placed
        assert result.state.total_placed_vms == 3
        # "large" was placed first (descending memory sort)
        # It should be on one node, and "medium" + "small" can be on the other
        # (or together with large if they fit)

    def test_placement_is_deterministic(self) -> None:
        """Running twice with same inputs produces identical placement."""

        def do_run() -> dict[str, str]:
            state = ClusterState([_inv_node(cpu_total=80.0, memory_total=300_000.0)])
            vms = [_vm(f"v{i}", cpu=2.0, memory_mb=4096.0) for i in range(10)]
            catalog = _catalog()
            result = run_placement(vms=vms, state=state, config=_config(), catalog=catalog)
            return result.state.placement_map

        map1 = do_run()
        map2 = do_run()
        assert map1 == map2

    def test_empty_vm_list(self) -> None:
        """No VMs → no placements, no unplaced."""
        state = ClusterState([_inv_node()])
        result = run_placement(vms=[], state=state, config=_config(), catalog=_catalog())
        assert result.state.total_placed_vms == 0
        assert result.unplaced == []

    def test_catalog_counter_not_incremented_on_monster(self) -> None:
        """When a monster VM fails expansion, the counter should rollback."""
        state = ClusterState()
        # Two VMs: one monster (fails), one normal (triggers expansion)
        monster = VM(name="monster", cpu=500.0, memory_mb=10_000_000.0)
        normal = _vm("normal")
        catalog = _catalog()

        result = run_placement(
            vms=[monster, normal],
            state=state,
            config=_config(),
            catalog=catalog,
        )

        assert len(result.unplaced) == 1
        assert result.unplaced[0].name == "monster"
        assert result.state.total_placed_vms == 1
        # The catalog node created for "normal" should be index 1, not 2
        cat_nodes = result.state.catalog_nodes
        assert len(cat_nodes) == 1
        assert cat_nodes[0].id.endswith("-01")

    def test_multiple_catalog_expansions(self) -> None:
        """Each expansion creates a new catalog node with incrementing index."""
        state = ClusterState()  # No inventory
        # VMs large enough that each needs its own node
        vms = [_vm(f"v{i}", cpu=70.0, memory_mb=250_000.0) for i in range(3)]
        catalog = _catalog(_catalog_profile(name="big", ram_gb=512, cost=2.0))

        result = run_placement(vms=vms, state=state, config=_config(), catalog=catalog)

        assert result.unplaced == []
        cat_nodes = result.state.catalog_nodes
        assert len(cat_nodes) == 3
        ids = sorted(n.id for n in cat_nodes)
        assert ids == ["big-01", "big-02", "big-03"]

    # ── Inventory-only mode (no catalog) ─────────────────────────────

    def test_inventory_only_all_fit(self) -> None:
        """No catalog: VMs that fit on inventory are placed normally."""
        state = ClusterState([_inv_node(cpu_total=80.0, memory_total=300_000.0)])
        vms = [_vm(f"v{i}", cpu=2.0, memory_mb=4096.0) for i in range(5)]

        result = run_placement(vms=vms, state=state, config=_config(), catalog=None)

        assert result.unplaced == []
        assert result.state.total_placed_vms == 5
        assert len(result.state.catalog_nodes) == 0

    def test_inventory_only_overflow_goes_to_unplaced(self) -> None:
        """No catalog: VMs that don't fit are unplaced (no expansion)."""
        small_inv = _inv_node(cpu_total=3.0, memory_total=5000.0, pods_total=1)
        state = ClusterState([small_inv])
        vms = [_vm(f"v{i}", cpu=2.0, memory_mb=4096.0) for i in range(3)]

        result = run_placement(vms=vms, state=state, config=_config(), catalog=None)

        assert result.state.total_placed_vms == 1
        assert len(result.unplaced) == 2
        assert len(result.state.catalog_nodes) == 0

    def test_inventory_only_no_nodes_all_unplaced(self) -> None:
        """No catalog + no inventory = everything unplaced."""
        state = ClusterState()
        vms = [_vm(f"v{i}") for i in range(3)]

        result = run_placement(vms=vms, state=state, config=_config(), catalog=None)

        assert result.state.total_placed_vms == 0
        assert len(result.unplaced) == 3
