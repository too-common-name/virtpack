"""Tests for core.cluster_state.ClusterState.

Test categories:
  1. Construction & node management
  2. place() — counter mutation + map tracking
  3. unplace() — counter rollback + map cleanup
  4. place/unplace roundtrip (O(1) Lookahead simulation)
  5. get_candidate_nodes() — filtering
  6. Query helpers (inventory/catalog/active nodes)
  7. Multi-VM placement scenarios
"""

from __future__ import annotations

import pytest

from core.cluster_state import ClusterState
from models.node import Node
from models.vm import VM

# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════


def _node(
    node_id: str = "node-01",
    cpu_total: float = 80.0,
    memory_total: float = 400_000.0,
    pods_total: int = 250,
    *,
    is_inventory: bool = True,
    cost_weight: float = 0.0,
) -> Node:
    return Node(
        id=node_id,
        profile="test-profile",
        cpu_total=cpu_total,
        memory_total=memory_total,
        pods_total=pods_total,
        cost_weight=cost_weight,
        is_inventory=is_inventory,
    )


def _vm(
    name: str = "vm-01",
    cpu: float = 1.0,
    memory_mb: float = 4096.0,
) -> VM:
    return VM(name=name, cpu=cpu, memory_mb=memory_mb)


# ═══════════════════════════════════════════════════════════════════════
# 1. Construction & Node Management
# ═══════════════════════════════════════════════════════════════════════


class TestConstruction:
    def test_empty_cluster(self) -> None:
        state = ClusterState()
        assert state.nodes == []
        assert state.placement_map == {}
        assert state.node_vm_map == {}
        assert state.total_placed_vms == 0

    def test_initial_nodes(self) -> None:
        n1 = _node("n1")
        n2 = _node("n2")
        state = ClusterState([n1, n2])
        assert len(state.nodes) == 2
        assert state.node_vm_map == {"n1": [], "n2": []}

    def test_add_node(self) -> None:
        state = ClusterState([_node("n1")])
        n2 = _node("n2", is_inventory=False, cost_weight=1.0)
        state.add_node(n2)
        assert len(state.nodes) == 2
        assert "n2" in state.node_vm_map

    def test_nodes_is_ordered(self) -> None:
        nodes = [_node(f"n{i}") for i in range(5)]
        state = ClusterState(nodes)
        assert [n.id for n in state.nodes] == ["n0", "n1", "n2", "n3", "n4"]

    def test_initial_nodes_list_is_copied(self) -> None:
        """Modifying the original list does not affect the state."""
        original = [_node("n1")]
        state = ClusterState(original)
        original.append(_node("n2"))
        assert len(state.nodes) == 1


# ═══════════════════════════════════════════════════════════════════════
# 2. place() — Counter Mutation + Map Tracking
# ═══════════════════════════════════════════════════════════════════════


class TestPlace:
    def test_updates_node_counters(self) -> None:
        node = _node("n1")
        state = ClusterState([node])
        vm = _vm("db01", cpu=2.0, memory_mb=8192.0)

        state.place(vm, node)

        assert node.cpu_used == pytest.approx(2.0)
        assert node.memory_used == pytest.approx(8192.0)
        assert node.pods_used == 1

    def test_updates_placement_map(self) -> None:
        node = _node("n1")
        state = ClusterState([node])
        vm = _vm("db01")

        state.place(vm, node)

        assert state.placement_map == {"db01": "n1"}

    def test_updates_node_vm_map(self) -> None:
        node = _node("n1")
        state = ClusterState([node])
        vm = _vm("db01")

        state.place(vm, node)

        assert state.node_vm_map == {"n1": ["db01"]}

    def test_total_placed_vms_increments(self) -> None:
        node = _node("n1")
        state = ClusterState([node])
        state.place(_vm("v1"), node)
        state.place(_vm("v2"), node)
        assert state.total_placed_vms == 2

    def test_multiple_vms_same_node(self) -> None:
        node = _node("n1")
        state = ClusterState([node])
        state.place(_vm("v1", cpu=1.0, memory_mb=1000.0), node)
        state.place(_vm("v2", cpu=2.0, memory_mb=2000.0), node)

        assert node.cpu_used == pytest.approx(3.0)
        assert node.memory_used == pytest.approx(3000.0)
        assert node.pods_used == 2
        assert state.node_vm_map["n1"] == ["v1", "v2"]


# ═══════════════════════════════════════════════════════════════════════
# 3. unplace() — Counter Rollback + Map Cleanup
# ═══════════════════════════════════════════════════════════════════════


class TestUnplace:
    def test_restores_node_counters(self) -> None:
        node = _node("n1")
        state = ClusterState([node])
        vm = _vm("db01", cpu=2.0, memory_mb=8192.0)

        state.place(vm, node)
        state.unplace(vm, node)

        assert node.cpu_used == pytest.approx(0.0)
        assert node.memory_used == pytest.approx(0.0)
        assert node.pods_used == 0

    def test_removes_from_placement_map(self) -> None:
        node = _node("n1")
        state = ClusterState([node])
        vm = _vm("db01")

        state.place(vm, node)
        state.unplace(vm, node)

        assert state.placement_map == {}

    def test_removes_from_node_vm_map(self) -> None:
        node = _node("n1")
        state = ClusterState([node])
        vm = _vm("db01")

        state.place(vm, node)
        state.unplace(vm, node)

        assert state.node_vm_map == {"n1": []}

    def test_unplace_raises_on_missing_vm(self) -> None:
        node = _node("n1")
        state = ClusterState([node])
        vm = _vm("ghost")
        with pytest.raises(KeyError):
            state.unplace(vm, node)

    def test_partial_unplace_preserves_other_vms(self) -> None:
        node = _node("n1")
        state = ClusterState([node])
        v1 = _vm("v1", cpu=1.0, memory_mb=1000.0)
        v2 = _vm("v2", cpu=2.0, memory_mb=2000.0)

        state.place(v1, node)
        state.place(v2, node)
        state.unplace(v1, node)

        assert node.cpu_used == pytest.approx(2.0)
        assert node.memory_used == pytest.approx(2000.0)
        assert node.pods_used == 1
        assert state.placement_map == {"v2": "n1"}
        assert state.node_vm_map["n1"] == ["v2"]


# ═══════════════════════════════════════════════════════════════════════
# 4. Place/Unplace Roundtrip (Lookahead Simulation)
# ═══════════════════════════════════════════════════════════════════════


class TestLookaheadRoundtrip:
    """Simulates the Lookahead k=2 pattern from HLD §6.2."""

    def test_single_lookahead_roundtrip(self) -> None:
        """place → score → unplace leaves state unchanged."""
        node = _node("n1", cpu_total=80.0, memory_total=400_000.0)
        state = ClusterState([node])
        vm = _vm("lookahead-vm", cpu=4.0, memory_mb=16384.0)

        # Snapshot before
        cpu_before = node.cpu_used
        mem_before = node.memory_used
        pods_before = node.pods_used

        # Simulate
        state.place(vm, node)
        # (scorer would read node.cpu_util, node.memory_util here)
        state.unplace(vm, node)

        # Verify exact rollback
        assert node.cpu_used == pytest.approx(cpu_before)
        assert node.memory_used == pytest.approx(mem_before)
        assert node.pods_used == pods_before
        assert state.placement_map == {}
        assert state.total_placed_vms == 0

    def test_lookahead_with_existing_placement(self) -> None:
        """Lookahead doesn't disturb already-placed VMs."""
        node = _node("n1")
        state = ClusterState([node])

        # Permanently place v1
        v1 = _vm("v1", cpu=10.0, memory_mb=50_000.0)
        state.place(v1, node)

        # Lookahead with v2
        v2 = _vm("v2", cpu=5.0, memory_mb=20_000.0)
        state.place(v2, node)
        state.unplace(v2, node)

        # v1 still placed, counters back to v1-only state
        assert node.cpu_used == pytest.approx(10.0)
        assert node.memory_used == pytest.approx(50_000.0)
        assert node.pods_used == 1
        assert state.placement_map == {"v1": "n1"}
        assert state.node_vm_map["n1"] == ["v1"]

    def test_multiple_lookahead_candidates(self) -> None:
        """Lookahead across multiple candidate nodes — only final bind persists."""
        n1 = _node("n1")
        n2 = _node("n2")
        state = ClusterState([n1, n2])
        vm = _vm("test-vm", cpu=2.0, memory_mb=8000.0)

        # Lookahead on n1
        state.place(vm, n1)
        state.unplace(vm, n1)

        # Lookahead on n2
        state.place(vm, n2)
        state.unplace(vm, n2)

        # Final bind on n1
        state.place(vm, n1)

        assert n1.cpu_used == pytest.approx(2.0)
        assert n2.cpu_used == pytest.approx(0.0)
        assert state.placement_map == {"test-vm": "n1"}


# ═══════════════════════════════════════════════════════════════════════
# 5. get_candidate_nodes() — Filtering
# ═══════════════════════════════════════════════════════════════════════


class TestGetCandidateNodes:
    def test_all_nodes_fit(self) -> None:
        nodes = [_node("n1"), _node("n2")]
        state = ClusterState(nodes)
        vm = _vm("tiny", cpu=0.5, memory_mb=100.0)
        assert len(state.get_candidate_nodes(vm)) == 2

    def test_no_nodes_fit(self) -> None:
        node = _node("n1", cpu_total=1.0, memory_total=100.0, pods_total=1)
        state = ClusterState([node])
        vm = _vm("monster", cpu=999.0, memory_mb=999_999.0)
        assert state.get_candidate_nodes(vm) == []

    def test_partial_fit(self) -> None:
        small = _node("small", cpu_total=2.0, memory_total=4096.0)
        large = _node("large", cpu_total=80.0, memory_total=400_000.0)
        state = ClusterState([small, large])
        vm = _vm("medium", cpu=5.0, memory_mb=10_000.0)
        candidates = state.get_candidate_nodes(vm)
        assert len(candidates) == 1
        assert candidates[0].id == "large"

    def test_filtering_accounts_for_existing_load(self) -> None:
        node = _node("n1", cpu_total=10.0, memory_total=20_000.0)
        state = ClusterState([node])

        # Fill the node most of the way
        state.place(_vm("v1", cpu=8.0, memory_mb=15_000.0), node)

        # This VM still fits
        small = _vm("small", cpu=1.0, memory_mb=1000.0)
        assert len(state.get_candidate_nodes(small)) == 1

        # This VM doesn't fit
        big = _vm("big", cpu=5.0, memory_mb=10_000.0)
        assert len(state.get_candidate_nodes(big)) == 0

    def test_pods_limit_filters(self) -> None:
        node = _node("n1", pods_total=2)
        state = ClusterState([node])
        state.place(_vm("v1"), node)
        state.place(_vm("v2"), node)
        # Pod limit reached
        assert state.get_candidate_nodes(_vm("v3")) == []

    def test_filtering_includes_added_catalog_nodes(self) -> None:
        """Nodes added by Expander appear in candidate filtering."""
        state = ClusterState([_node("inv-01")])
        catalog_node = _node("cat-01", is_inventory=False, cost_weight=1.0)
        state.add_node(catalog_node)

        vm = _vm("v1", cpu=1.0, memory_mb=1000.0)
        candidates = state.get_candidate_nodes(vm)
        assert len(candidates) == 2


# ═══════════════════════════════════════════════════════════════════════
# 6. Query Helpers
# ═══════════════════════════════════════════════════════════════════════


class TestGetNodeVMs:
    """Tests for get_node_vms() — resolves VM names to VM objects."""

    def test_returns_placed_vms(self) -> None:
        node = _node("n1")
        state = ClusterState([node])
        v1 = _vm("v1", cpu=2.0, memory_mb=8000.0)
        v2 = _vm("v2", cpu=4.0, memory_mb=16000.0)
        state.place(v1, node)
        state.place(v2, node)

        result = state.get_node_vms("n1")
        assert len(result) == 2
        assert result[0].name == "v1"
        assert result[0].cpu == 2.0
        assert result[1].name == "v2"
        assert result[1].memory_mb == 16000.0

    def test_empty_node_returns_empty(self) -> None:
        node = _node("n1")
        state = ClusterState([node])
        assert state.get_node_vms("n1") == []

    def test_unknown_node_returns_empty(self) -> None:
        state = ClusterState()
        assert state.get_node_vms("nonexistent") == []

    def test_unplace_removes_from_registry(self) -> None:
        node = _node("n1")
        state = ClusterState([node])
        v1 = _vm("v1", cpu=2.0, memory_mb=8000.0)
        v2 = _vm("v2", cpu=4.0, memory_mb=16000.0)
        state.place(v1, node)
        state.place(v2, node)
        state.unplace(v1, node)

        result = state.get_node_vms("n1")
        assert len(result) == 1
        assert result[0].name == "v2"


class TestQueryHelpers:
    def test_inventory_vs_catalog_nodes(self) -> None:
        inv = _node("inv-01", is_inventory=True)
        cat = _node("cat-01", is_inventory=False, cost_weight=1.0)
        state = ClusterState([inv, cat])

        assert len(state.inventory_nodes) == 1
        assert state.inventory_nodes[0].id == "inv-01"
        assert len(state.catalog_nodes) == 1
        assert state.catalog_nodes[0].id == "cat-01"

    def test_active_nodes_empty_cluster(self) -> None:
        state = ClusterState([_node("n1"), _node("n2")])
        assert state.active_nodes == []

    def test_active_nodes_after_placement(self) -> None:
        n1 = _node("n1")
        n2 = _node("n2")
        state = ClusterState([n1, n2])
        state.place(_vm("v1"), n1)

        assert len(state.active_nodes) == 1
        assert state.active_nodes[0].id == "n1"

    def test_read_only_maps(self) -> None:
        """Returned maps are copies — mutations don't affect state."""
        node = _node("n1")
        state = ClusterState([node])
        state.place(_vm("v1"), node)

        pm = state.placement_map
        pm["ghost"] = "nowhere"
        assert "ghost" not in state.placement_map

        nvm = state.node_vm_map
        nvm["n1"].append("ghost")
        assert "ghost" not in state.node_vm_map["n1"]


# ═══════════════════════════════════════════════════════════════════════
# 7. Multi-VM Placement Scenario
# ═══════════════════════════════════════════════════════════════════════


class TestMultiVMScenario:
    """End-to-end placement sequence mimicking the engine loop."""

    def test_place_5_vms_across_2_nodes(self) -> None:
        n1 = _node("n1", cpu_total=10.0, memory_total=50_000.0)
        n2 = _node("n2", cpu_total=10.0, memory_total=50_000.0)
        state = ClusterState([n1, n2])

        vms = [_vm(f"vm-{i}", cpu=3.0, memory_mb=15_000.0) for i in range(5)]

        # Place first 3 on n1 (fills to 9.0/10.0 CPU)
        for vm in vms[:3]:
            state.place(vm, n1)

        # n1 can't fit another 3.0 CPU, only n2 is candidate
        assert state.get_candidate_nodes(vms[3]) == [n2]

        # Place remaining on n2
        for vm in vms[3:]:
            state.place(vm, n2)

        assert state.total_placed_vms == 5
        assert n1.pods_used == 3
        assert n2.pods_used == 2
        assert len(state.active_nodes) == 2
        assert set(state.placement_map.keys()) == {f"vm-{i}" for i in range(5)}
