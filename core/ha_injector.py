"""HA Node Injection — post-placement spare-capacity guarantee (HLD §7).

After placement completes, the cluster must maintain enough spare capacity
to tolerate ``N`` simultaneous node failures (``ha_failures_to_tolerate``).

Because CPU and memory are independent scheduling dimensions, worst-case
failure is evaluated separately:

    required_spare_cpu    = sum of cpu_used on the N most CPU-loaded nodes
    required_spare_memory = sum of memory_used on the N most memory-loaded nodes

If the cluster's current spare capacity is insufficient, the injector adds
the cheapest catalog nodes until both deficits are covered.

When no catalog is available (inventory-only mode), the injector reports
the uncovered deficit without adding nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from core.normalizer import build_catalog_node

if TYPE_CHECKING:
    from core.cluster_state import ClusterState
    from models.config import CatalogConfig, CatalogProfile, PlanConfig
    from models.node import Node


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
# HA injection
# ═══════════════════════════════════════════════════════════════════════


def inject_ha_nodes(
    *,
    state: ClusterState,
    config: PlanConfig,
    catalog: CatalogConfig | None = None,
) -> HAResult:
    """Add catalog nodes to satisfy the HA spare-capacity requirement.

    Algorithm:

    1. Compute worst-case spare needed for ``ha_failures_to_tolerate``.
    2. Compute current cluster spare.
    3. If deficit exists and a catalog is available, greedily add
       the cheapest catalog profile until both CPU and memory
       deficits are covered.

    HA nodes are added to ``state`` (via ``state.add_node``) and are
    identifiable by their ``"ha-"`` ID prefix.

    Parameters
    ----------
    state : ClusterState
        Post-placement cluster state.
    config : PlanConfig
        Global configuration (safety margins, overheads).
    catalog : CatalogConfig | None
        Available catalog profiles.  ``None`` = inventory-only mode
        (deficit is reported but no nodes are added).

    Returns
    -------
    HAResult
        Contains the list of added HA nodes and any remaining deficit.
    """
    n_failures = config.safety_margins.ha_failures_to_tolerate
    if n_failures <= 0:
        return HAResult()

    req = compute_ha_requirements(state, n_failures)
    spare_cpu, spare_mem = compute_current_spare(state)

    deficit_cpu = max(0.0, req.required_spare_cpu - spare_cpu)
    deficit_mem = max(0.0, req.required_spare_memory - spare_mem)

    # Already covered — no injection needed
    if deficit_cpu <= 0.0 and deficit_mem <= 0.0:
        return HAResult()

    # No catalog → report deficit without adding nodes
    if catalog is None or not catalog.profiles:
        return HAResult(deficit_cpu=deficit_cpu, deficit_memory=deficit_mem)

    # Greedy: add cheapest catalog profile until both deficits are covered
    # Sort profiles by cost_weight to always pick the cheapest option
    cheapest_profile = min(catalog.profiles, key=lambda p: p.cost_weight)

    nodes_added: list[Node] = []
    ha_counter = 0

    while deficit_cpu > 0.0 or deficit_mem > 0.0:
        ha_counter += 1
        node = _build_ha_node(cheapest_profile, ha_counter, config)
        state.add_node(node)
        nodes_added.append(node)
        deficit_cpu = max(0.0, deficit_cpu - node.cpu_total)
        deficit_mem = max(0.0, deficit_mem - node.memory_total)

    return HAResult(nodes_added=nodes_added, deficit_cpu=0.0, deficit_memory=0.0)


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
