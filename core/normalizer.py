"""Resource normalization for VMs and Nodes.

Pipeline (HLD §4 + §5):
  1. **VM normalization** — apply overcommit ratios
  2. **Node normalization**
     a. Stage 1 — effective CPU (HT adjustment)
     b. Stage 2 — subtract kubelet + OCP Virt overheads
     c. Stage 3 — apply utilization target safety margins
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from models.node import Node
from models.vm import VM

if TYPE_CHECKING:
    from models.config import (
        CatalogProfile,
        CpuTopology,
        InventoryConfig,
        OvercommitConfig,
        PlanConfig,
        VirtOverheads,
    )

# ═══════════════════════════════════════════════════════════════════════
# VM Normalization  (HLD §4)
# ═══════════════════════════════════════════════════════════════════════


def normalize_vm(
    *,
    name: str,
    raw_cpu: float,
    raw_memory_mb: float,
    overcommit: OvercommitConfig,
) -> VM:
    """Apply overcommit ratios to raw VM resources.

    Returns an immutable :class:`VM` with effective resource requests::

        effective_cpu       = raw_cpu / cpu_ratio
        effective_memory_mb = raw_memory_mb / memory_ratio
    """
    return VM(
        name=name,
        cpu=raw_cpu / overcommit.cpu_ratio,
        memory_mb=raw_memory_mb / overcommit.memory_ratio,
    )


# ═══════════════════════════════════════════════════════════════════════
# MCO Kubelet Auto-Sizing Step Functions
# ═══════════════════════════════════════════════════════════════════════
#
# Source:
#   github.com/openshift/machine-config-operator
#   templates/common/_base/files/kubelet-auto-sizing.yaml
#
# These match the standard Kubernetes recommended system-reserved values
# used by GKE, EKS, and the OpenShift Machine Config Operator.
# ═══════════════════════════════════════════════════════════════════════

# CPU brackets: (upper_bound_cores, rate)
_CPU_BRACKETS: list[tuple[float, float]] = [
    (1.0, 0.06),  # 6%  of first  1 core
    (2.0, 0.01),  # 1%  of next   1 core
    (4.0, 0.005),  # 0.5% of next  2 cores
]
_CPU_TAIL_RATE: float = 0.0025  # 0.25% of anything above 4 cores

# Memory brackets: (upper_bound_gib, rate)
_MEM_BRACKETS: list[tuple[float, float]] = [
    (4.0, 0.25),  # 25%  of first   4 GiB
    (8.0, 0.20),  # 20%  of next    4 GiB
    (16.0, 0.10),  # 10%  of next    8 GiB
    (128.0, 0.06),  # 6%   of next  112 GiB
]
_MEM_TAIL_RATE: float = 0.02  # 2% of anything above 128 GiB

_GIB_TO_MB: float = 1024.0


def kubelet_reserved_cpu(total_cores: float) -> float:
    """Kubelet system-reserved CPU in cores.

    Step function::

        6%    of first  1 core
        1%    of next   1 core   (1–2)
        0.5%  of next   2 cores  (2–4)
        0.25% of remaining       (>4)
    """
    reserved = 0.0
    prev_bound = 0.0
    for upper, rate in _CPU_BRACKETS:
        bracket_size = upper - prev_bound
        consumed = min(max(total_cores - prev_bound, 0.0), bracket_size)
        reserved += consumed * rate
        prev_bound = upper
    # Tail: everything above last bracket
    if total_cores > prev_bound:
        reserved += (total_cores - prev_bound) * _CPU_TAIL_RATE
    return reserved


def kubelet_reserved_memory_mb(total_memory_mb: float) -> float:
    """Kubelet system-reserved memory in MB.

    Step function (brackets in GiB, input/output in MB)::

        25% of first   4 GiB
        20% of next    4 GiB   (4–8)
        10% of next    8 GiB   (8–16)
        6%  of next  112 GiB   (16–128)
        2%  of remaining        (>128)
    """
    total_gib = total_memory_mb / _GIB_TO_MB
    reserved_gib = 0.0
    prev_bound = 0.0
    for upper, rate in _MEM_BRACKETS:
        bracket_size = upper - prev_bound
        consumed = min(max(total_gib - prev_bound, 0.0), bracket_size)
        reserved_gib += consumed * rate
        prev_bound = upper
    # Tail: everything above last bracket
    if total_gib > prev_bound:
        reserved_gib += (total_gib - prev_bound) * _MEM_TAIL_RATE
    return reserved_gib * _GIB_TO_MB


# ═══════════════════════════════════════════════════════════════════════
# Node Normalization  (HLD §4 + §5)
# ═══════════════════════════════════════════════════════════════════════


def compute_effective_cpu(
    topology: CpuTopology,
    overheads: VirtOverheads,
) -> float:
    """Stage 1 — Effective CPU count accounting for HT efficiency.

    ::

        if threads_per_core > 1:
            effective_cpu = physical_cores × ht_efficiency_factor
        else:
            effective_cpu = physical_cores
    """
    if topology.threads_per_core > 1:
        return topology.physical_cores * overheads.ht_efficiency_factor
    return float(topology.physical_cores)


def compute_usable_capacity(
    *,
    topology: CpuTopology,
    ram_gb: int,
    overheads: VirtOverheads,
) -> tuple[float, float]:
    """Stage 2 — Subtract kubelet + OCP Virtualization overheads.

    Returns ``(usable_cpu, usable_memory_mb)`` *before* safety margins.

    ::

        usable_cpu    = effective_cpu − kubelet_cpu − ocp_virt_cpu
        usable_memory = node_memory   − kubelet_mem − eviction − ocp_virt_mem
    """
    # ── CPU ──────────────────────────────────────────────────────────
    effective_cpu = compute_effective_cpu(topology, overheads)
    kubelet_cpu = kubelet_reserved_cpu(effective_cpu)
    usable_cpu = effective_cpu - kubelet_cpu - overheads.ocp_virt_core

    # ── Memory (all in MB) ──────────────────────────────────────────
    total_memory_mb = float(ram_gb) * _GIB_TO_MB
    kubelet_mem = kubelet_reserved_memory_mb(total_memory_mb)
    usable_memory_mb = (
        total_memory_mb - kubelet_mem - overheads.eviction_hard_mb - overheads.ocp_virt_memory_mb
    )

    # Guard: overheads must not exceed raw capacity
    if usable_cpu <= 0:
        raise ValueError(
            f"Negative usable CPU ({usable_cpu:.2f}) after subtracting overheads "
            f"from {effective_cpu:.1f} effective cores. "
            f"Check VirtOverheads or node topology."
        )
    if usable_memory_mb <= 0:
        raise ValueError(
            f"Negative usable memory ({usable_memory_mb:.1f} MB) after subtracting "
            f"overheads from {ram_gb} GiB. "
            f"Check VirtOverheads or node topology."
        )

    return (usable_cpu, usable_memory_mb)


def normalize_node_capacity(
    *,
    topology: CpuTopology,
    ram_gb: int,
    config: PlanConfig,
) -> tuple[float, float, int]:
    """Full node normalization pipeline: overheads + safety margins.

    Returns ``(schedulable_cpu, schedulable_memory_mb, max_pods)``.

    HLD §4 (overheads) + §5 (safety margins)::

        schedulable = usable × (utilization_target / 100)
    """
    usable_cpu, usable_memory_mb = compute_usable_capacity(
        topology=topology,
        ram_gb=ram_gb,
        overheads=config.virt_overheads,
    )

    # Apply safety margins (HLD §5)
    targets = config.safety_margins.utilization_targets
    schedulable_cpu = usable_cpu * (targets.cpu / 100.0)
    schedulable_memory_mb = usable_memory_mb * (targets.memory / 100.0)

    # Pod limit from cluster config
    max_pods = config.cluster_limits.max_pods_per_node

    return (schedulable_cpu, schedulable_memory_mb, max_pods)


# ═══════════════════════════════════════════════════════════════════════
# Node Builders
# ═══════════════════════════════════════════════════════════════════════


def build_inventory_nodes(
    inventory: InventoryConfig,
    config: PlanConfig,
) -> list[Node]:
    """Create normalized inventory nodes from all profiles.

    Each :class:`InventoryProfile` with ``quantity=N`` produces *N*
    :class:`Node` instances with IDs ``"{profile_name}-01"`` …
    ``"{profile_name}-{N:02d}"``.
    """
    nodes: list[Node] = []
    for profile in inventory.profiles:
        cpu_total, memory_total, pods_total = normalize_node_capacity(
            topology=profile.cpu_topology,
            ram_gb=profile.ram_gb,
            config=config,
        )
        for i in range(1, profile.quantity + 1):
            nodes.append(
                Node.new_inventory(
                    profile=profile.profile_name,
                    index=i,
                    cpu_total=cpu_total,
                    memory_total=memory_total,
                    pods_total=pods_total,
                )
            )
    return nodes


def build_catalog_node(
    profile: CatalogProfile,
    index: int,
    config: PlanConfig,
) -> Node:
    """Create a single normalized catalog node.

    Called by the Expander each time it needs a new greenfield node.
    """
    cpu_total, memory_total, pods_total = normalize_node_capacity(
        topology=profile.cpu_topology,
        ram_gb=profile.ram_gb,
        config=config,
    )
    return Node.new_catalog(
        profile=profile.profile_name,
        index=index,
        cpu_total=cpu_total,
        memory_total=memory_total,
        pods_total=pods_total,
        cost_weight=profile.cost_weight,
    )
