"""Mutable cluster state for the placement simulation.

``ClusterState`` owns the list of nodes and the VM→Node placement map.
It provides O(1) ``place`` / ``unplace`` operations to enable fast
Lookahead rollback (LLD §2.1, HLD §6.2).

Usage by the placement engine::

    state = ClusterState(inventory_nodes)

    # Lookahead (temporary — cancelled by unplace)
    state.place(vm, node)
    score = scorer.score(node)
    state.unplace(vm, node)

    # Final bind (permanent)
    state.place(vm, best_node)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.node import Node
    from models.vm import VM


class ClusterState:
    """Tracks all nodes and VM placements during the simulation.

    Attributes
    ----------
    nodes : list[Node]
        All nodes (inventory + catalog), in insertion order.
    placement_map : dict[str, str]
        ``vm.name → node.id`` for every placed VM.
    node_vm_map : dict[str, list[str]]
        ``node.id → [vm.name, ...]`` for auditing / CSV export.
    """

    __slots__ = ("_node_vm_map", "_nodes", "_placement_map", "_vm_registry")

    def __init__(self, nodes: list[Node] | None = None) -> None:
        self._nodes: list[Node] = list(nodes) if nodes else []
        self._placement_map: dict[str, str] = {}
        # dict[str, None] instead of list[str]: O(1) delete in unplace()
        # (list.remove is O(k) per node). Keys stay insertion-ordered.
        self._node_vm_map: dict[str, dict[str, None]] = {n.id: {} for n in self._nodes}
        self._vm_registry: dict[str, VM] = {}

    # ── Node management ──────────────────────────────────────────────

    @property
    def nodes(self) -> list[Node]:
        """All nodes in the cluster (inventory + catalog)."""
        return self._nodes

    @property
    def placement_map(self) -> dict[str, str]:
        """Read-only view: ``vm_name → node_id`` for every placed VM."""
        return dict(self._placement_map)

    @property
    def node_vm_map(self) -> dict[str, list[str]]:
        """Read-only view: ``node_id → [vm_names]`` for every node."""
        return {nid: list(vms.keys()) for nid, vms in self._node_vm_map.items()}

    def add_node(self, node: Node) -> None:
        """Register a new node (used by Expander for catalog nodes)."""
        self._nodes.append(node)
        self._node_vm_map[node.id] = {}

    # ── O(1) place / unplace ─────────────────────────────────────────

    def place(self, vm: VM, node: Node) -> None:
        """Place a VM on a node — O(1) mutation.

        Updates the node's usage counters and records the mapping.
        Called by both the Lookahead (temporary) and the final bind
        (permanent).
        """
        node.cpu_used += vm.cpu
        node.memory_used += vm.memory_mb
        node.pods_used += vm.pods
        self._placement_map[vm.name] = node.id
        self._node_vm_map[node.id][vm.name] = None
        self._vm_registry[vm.name] = vm

    def unplace(self, vm: VM, node: Node) -> None:
        """Reverse a placement — O(1) rollback.

        Restores the node's usage counters and removes the mapping.
        Critical for Lookahead heuristic simulation.
        """
        node.cpu_used -= vm.cpu
        node.memory_used -= vm.memory_mb
        node.pods_used -= vm.pods
        del self._placement_map[vm.name]
        del self._node_vm_map[node.id][vm.name]
        del self._vm_registry[vm.name]

    # ── Filtering ────────────────────────────────────────────────────

    def get_candidate_nodes(self, vm: VM) -> list[Node]:
        """Return all nodes that can fit *vm* across all 3 dimensions.

        This is the **filter phase** of the K8s-like scheduling loop
        (HLD §6, step 1).
        """
        return [n for n in self._nodes if n.fits(vm)]

    # ── Query helpers (for reporting / HA injection) ─────────────────

    def get_node_vms(self, node_id: str) -> list[VM]:
        """Return the actual VM objects placed on *node_id*."""
        return [self._vm_registry[name] for name in self._node_vm_map.get(node_id, {})]

    @property
    def inventory_nodes(self) -> list[Node]:
        """Brownfield nodes (``is_inventory=True``)."""
        return [n for n in self._nodes if n.is_inventory]

    @property
    def catalog_nodes(self) -> list[Node]:
        """Greenfield nodes (``is_inventory=False``)."""
        return [n for n in self._nodes if not n.is_inventory]

    @property
    def active_nodes(self) -> list[Node]:
        """Nodes with at least one VM placed."""
        return [n for n in self._nodes if n.pods_used > 0]

    @property
    def total_placed_vms(self) -> int:
        """Number of VMs currently in the placement map."""
        return len(self._placement_map)
