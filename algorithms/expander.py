"""Catalog node expansion (HLD §6.2 step 2).

When no existing node can fit a VM, the Expander selects the cheapest
catalog profile that *could* fit the VM (on an empty node) and creates
a new normalized node.

Pure functions — no side effects on ClusterState (the engine does the
``add_node`` call).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.normalizer import build_catalog_node, normalize_node_capacity

if TYPE_CHECKING:
    from models.config import CatalogConfig, CatalogProfile, PlanConfig
    from models.node import Node
    from models.vm import VM


def _profile_fits_vm(
    profile: CatalogProfile,
    vm: VM,
    config: PlanConfig,
) -> bool:
    """Return True if an *empty* node of this profile can host *vm*."""
    cpu, mem, pods = normalize_node_capacity(
        topology=profile.cpu_topology,
        total_memory_mb=float(profile.ram_gb) * 1024.0,
        config=config,
    )
    return vm.cpu <= cpu and vm.memory_mb <= mem and vm.pods <= pods


def select_profile(
    vm: VM,
    catalog: CatalogConfig,
    config: PlanConfig,
) -> CatalogProfile | None:
    """Pick the cheapest catalog profile that can fit *vm* on an empty node.

    Returns ``None`` if no profile is large enough (monster VM).
    Profiles are compared by ``cost_weight`` (lower is cheaper).
    """
    eligible = [p for p in catalog.profiles if _profile_fits_vm(p, vm, config)]
    if not eligible:
        return None
    return min(eligible, key=lambda p: p.cost_weight)


def expand(
    vm: VM,
    catalog: CatalogConfig | None,
    config: PlanConfig,
    next_index: int,
) -> Node | None:
    """Create a new catalog node for *vm*, or ``None`` if impossible.

    This is the complete expand step: select profile → build node.
    The caller (PlacementEngine) is responsible for calling
    ``state.add_node(node)`` with the result.

    Returns ``None`` when:
    * ``catalog`` is ``None`` (inventory-only mode — no expansion).
    * No catalog profile is large enough for *vm* (monster VM).

    Parameters
    ----------
    vm : VM
        The VM that triggered expansion (no existing node fits).
    catalog : CatalogConfig | None
        Available hardware profiles, or ``None`` for inventory-only mode.
    config : PlanConfig
        Global configuration (overheads, safety margins, limits).
    next_index : int
        1-based sequence number for the new node's ID.
    """
    if catalog is None:
        return None
    profile = select_profile(vm, catalog, config)
    if profile is None:
        return None
    return build_catalog_node(profile, next_index, config)
