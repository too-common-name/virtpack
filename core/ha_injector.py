"""HA Node Injection — post-placement spare-capacity guarantee (HLD §7).

After placement completes, the cluster must maintain enough spare capacity
to tolerate ``N`` simultaneous node failures (``ha_failures_to_tolerate``).

The injector uses **simulation-based** deficit computation: for every
possible N-node failure combination, it performs a greedy first-fit
re-placement of displaced VMs onto surviving nodes, checking that both
CPU *and* memory fit on the same node.  This avoids the pitfall of
counting stranded capacity (e.g., free CPU on a memory-saturated node)
as usable spare.

If the cluster's current spare capacity is insufficient, the injector
reclaims unused inventory nodes or adds the cheapest catalog nodes until
all failure scenarios are survivable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import TYPE_CHECKING

from core.normalizer import build_catalog_node

if TYPE_CHECKING:
    from core.cluster_state import ClusterState
    from models.config import CatalogConfig, CatalogProfile, PlanConfig
    from models.node import Node
    from models.vm import VM


# ═══════════════════════════════════════════════════════════════════════
# Data containers
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class HARequirement:
    """Worst-case spare capacity needed to survive ``n`` node failures."""

    required_spare_cpu: float
    required_spare_memory: float


@dataclass
class HAResult:
    """Outcome of the HA injection phase."""

    nodes_added: list[Node] = field(default_factory=list)
    nodes_reclaimed: list[Node] = field(default_factory=list)
    deficit_cpu: float = 0.0
    deficit_memory: float = 0.0

    @property
    def fully_covered(self) -> bool:
        """True if the HA requirement is fully satisfied."""
        return self.deficit_cpu <= 0.0 and self.deficit_memory <= 0.0


# ═══════════════════════════════════════════════════════════════════════
# Pure calculations
# ═══════════════════════════════════════════════════════════════════════


def compute_ha_requirements(
    state: ClusterState,
    n_failures: int,
) -> HARequirement:
    """Compute worst-case spare capacity for *n_failures* (HLD §7).

    Evaluates CPU and memory independently:

    * **CPU:** sum of ``cpu_used`` on the *N* most CPU-loaded active nodes.
    * **Memory:** sum of ``memory_used`` on the *N* most memory-loaded
      active nodes.

    The worst-case sets may overlap (the heaviest CPU node may also be
    the heaviest memory node) — this is correct because we must
    survive *either* worst-case independently.

    Returns zeros when ``n_failures <= 0`` or the cluster is empty.
    """
    if n_failures <= 0:
        return HARequirement(0.0, 0.0)

    active = state.active_nodes
    if not active:
        return HARequirement(0.0, 0.0)

    n = min(n_failures, len(active))

    top_cpu = sorted(active, key=lambda nd: nd.cpu_used, reverse=True)[:n]
    top_mem = sorted(active, key=lambda nd: nd.memory_used, reverse=True)[:n]

    return HARequirement(
        required_spare_cpu=sum(nd.cpu_used for nd in top_cpu),
        required_spare_memory=sum(nd.memory_used for nd in top_mem),
    )


def compute_current_spare(state: ClusterState) -> tuple[float, float]:
    """Return ``(spare_cpu, spare_memory_mb)`` across all nodes."""
    spare_cpu = sum(n.cpu_remaining for n in state.nodes)
    spare_mem = sum(n.memory_remaining for n in state.nodes)
    return (spare_cpu, spare_mem)


# ═══════════════════════════════════════════════════════════════════════
# Simulation-based deficit
# ═══════════════════════════════════════════════════════════════════════


def _simulate_failure(
    surviving_nodes: list[Node],
    displaced_vms: list[VM],
) -> tuple[float, float]:
    """Greedy best-fit re-placement of displaced VMs onto survivors.

    For each VM (sorted by memory desc — standard packing order), try to
    find a surviving node with enough **co-located** spare capacity in
    CPU, memory, *and* pods.  VMs that cannot fit anywhere accumulate as
    the unplaced deficit.

    Returns ``(unplaced_cpu, unplaced_mem)`` — the total resources of
    VMs that could not be rescheduled.  Pure function, no side effects.
    """
    spare: dict[str, tuple[float, float, int]] = {
        n.id: (n.cpu_remaining, n.memory_remaining, n.pods_remaining) for n in surviving_nodes
    }
    unplaced_cpu = 0.0
    unplaced_mem = 0.0

    for vm in sorted(displaced_vms, key=lambda v: v.memory_mb, reverse=True):
        # Best-fit: among nodes with enough co-located capacity in
        # ALL three dimensions, pick the one with the least spare memory
        # so large pockets of headroom are preserved for bigger VMs.
        best_id: str | None = None
        best_mem = float("inf")
        for node in surviving_nodes:
            s_cpu, s_mem, s_pods = spare[node.id]
            if (
                s_cpu >= vm.cpu
                and s_mem >= vm.memory_mb
                and s_pods >= vm.pods
                and s_mem < best_mem
            ):
                best_id = node.id
                best_mem = s_mem
        if best_id is not None:
            s_cpu, s_mem, s_pods = spare[best_id]
            spare[best_id] = (s_cpu - vm.cpu, s_mem - vm.memory_mb, s_pods - vm.pods)
        else:
            unplaced_cpu += vm.cpu
            unplaced_mem += vm.memory_mb

    return (unplaced_cpu, unplaced_mem)


def compute_ha_deficit(
    state: ClusterState,
    n_failures: int,
) -> tuple[float, float]:
    """Simulate worst-case *n_failures* node failure, return the deficit.

    Enumerates all ``C(active, N)`` failure combinations.  For each,
    collects the VMs on the failed nodes and runs
    :func:`_simulate_failure` against the surviving nodes.

    Returns the worst-case ``(deficit_cpu, deficit_mem)`` — the scenario
    requiring the most additional capacity.  ``(0.0, 0.0)`` means every
    failure scenario is survivable with current spare capacity.
    """
    if n_failures <= 0:
        return (0.0, 0.0)

    active = state.active_nodes
    if not active:
        return (0.0, 0.0)

    n = min(n_failures, len(active))
    worst_cpu, worst_mem = 0.0, 0.0

    for failed_set in combinations(active, n):
        failed_ids = {id(nd) for nd in failed_set}
        surviving = [nd for nd in state.nodes if id(nd) not in failed_ids]

        displaced: list[VM] = []
        for nd in failed_set:
            displaced.extend(state.get_node_vms(nd.id))

        deficit_cpu, deficit_mem = _simulate_failure(surviving, displaced)

        if deficit_cpu + deficit_mem > worst_cpu + worst_mem:
            worst_cpu, worst_mem = deficit_cpu, deficit_mem

    return (worst_cpu, worst_mem)


# ═══════════════════════════════════════════════════════════════════════
# HA injection
# ═══════════════════════════════════════════════════════════════════════


def inject_ha_nodes(
    *,
    state: ClusterState,
    config: PlanConfig,
    catalog: CatalogConfig | None = None,
    unused_pool: list[Node] | None = None,
) -> HAResult:
    """Ensure the cluster has enough spare capacity for HA.

    Uses **simulation-based** deficit computation: for every possible
    N-node failure combination, displaced VMs are re-placed onto
    surviving nodes with a greedy first-fit that checks *both* CPU
    and memory fit on the same node.

    Algorithm (iterative):

    1. ``compute_ha_deficit`` → simulate worst-case N-node failure.
    2. If deficit exists:

       a. **Reclaim** the largest unused inventory node (free, already
          owned) and re-simulate.
       b. If pool is exhausted and a catalog is available, add the
          cheapest catalog profile and re-simulate.

    3. Repeat until all failure scenarios are survivable, or no more
       capacity sources are available.

    Reclaimed nodes are *removed* from ``unused_pool`` (mutated in
    place) and added to ``state``.

    Parameters
    ----------
    state : ClusterState
        Post-placement cluster state.
    config : PlanConfig
        Global configuration (safety margins, overheads).
    catalog : CatalogConfig | None
        Available catalog profiles.  ``None`` = inventory-only mode.
    unused_pool : list[Node] | None
        Inventory nodes not activated during consolidation.  The list
        is **mutated** — reclaimed nodes are removed from it.

    Returns
    -------
    HAResult
        Contains reclaimed inventory nodes, added catalog nodes, and
        any remaining deficit.
    """
    n_failures = config.safety_margins.ha_failures_to_tolerate
    if n_failures <= 0:
        return HAResult()

    if not state.active_nodes:
        return HAResult()

    nodes_reclaimed: list[Node] = []
    nodes_added: list[Node] = []
    ha_counter = 0

    cheapest_profile = (
        min(catalog.profiles, key=lambda p: p.cost_weight)
        if catalog and catalog.profiles
        else None
    )

    pool_sorted = (
        sorted(unused_pool, key=lambda n: n.memory_total, reverse=True) if unused_pool else []
    )

    prev_deficit: tuple[float, float] | None = None

    while True:
        deficit_cpu, deficit_mem = compute_ha_deficit(state, n_failures)
        if deficit_cpu <= 0.0 and deficit_mem <= 0.0:
            break

        # Phase 1: Reclaim from unused pool (free, largest first)
        if pool_sorted:
            node = pool_sorted.pop(0)
            if unused_pool is not None:
                unused_pool.remove(node)
            state.add_node(node)
            nodes_reclaimed.append(node)
            prev_deficit = None  # reset after pool reclaim
            continue

        # Phase 2: Add catalog node
        if cheapest_profile is not None:
            if prev_deficit is not None and (deficit_cpu, deficit_mem) == prev_deficit:
                # Last catalog node didn't reduce the deficit — profile is
                # too small to absorb any displaced VM.  Stop to avoid an
                # infinite loop.
                return HAResult(
                    nodes_reclaimed=nodes_reclaimed,
                    nodes_added=nodes_added,
                    deficit_cpu=deficit_cpu,
                    deficit_memory=deficit_mem,
                )
            prev_deficit = (deficit_cpu, deficit_mem)
            ha_counter += 1
            node = _build_ha_node(cheapest_profile, ha_counter, config)
            state.add_node(node)
            nodes_added.append(node)
            continue

        # Neither available — report remaining deficit
        return HAResult(
            nodes_reclaimed=nodes_reclaimed,
            nodes_added=nodes_added,
            deficit_cpu=deficit_cpu,
            deficit_memory=deficit_mem,
        )

    return HAResult(
        nodes_reclaimed=nodes_reclaimed,
        nodes_added=nodes_added,
    )


# ═══════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════


def _build_ha_node(
    profile: CatalogProfile,
    index: int,
    config: PlanConfig,
) -> Node:
    """Build a catalog node with an ``ha-`` prefixed ID.

    Re-uses the standard normalization pipeline but overrides the
    node ID to distinguish HA spare nodes from placement catalog nodes.
    """
    node = build_catalog_node(profile, index, config)
    # Override the auto-generated ID with an ha- prefix
    node.id = f"ha-{profile.profile_name}-{index:02d}"
    return node
