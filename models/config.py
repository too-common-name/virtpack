"""Configuration models for virtpack plan inputs.

Three YAML sources are modeled here:
  1. **config.yaml**  → :class:`PlanConfig`
  2. **inventory.yaml** → :class:`InventoryConfig`
  3. **catalog.yaml**  → :class:`CatalogConfig`

All configuration models are **frozen** (immutable after construction).
``strict`` is intentionally omitted so that YAML integers (e.g. ``8``)
coerce cleanly to ``float`` fields during parsing.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

# ═══════════════════════════════════════════════════════════════════════
# 0. Placement strategy (HLD §1.1)
# ═══════════════════════════════════════════════════════════════════════


class PlacementStrategy(StrEnum):
    """Controls how inventory nodes are fed to the placement engine.

    * ``spread``  — All inventory nodes are added upfront; the scorer
      distributes VMs evenly across the full fleet.
    * ``consolidate`` — Inventory nodes are held in a pool and pulled
      lazily (one at a time, like catalog expansion).  Unused inventory
      nodes are reported as candidates for shutdown, saving OCP
      subscriptions.
    """

    SPREAD = "spread"
    CONSOLIDATE = "consolidate"


# ═══════════════════════════════════════════════════════════════════════
# 1. config.yaml  (HLD §3.2)
# ═══════════════════════════════════════════════════════════════════════


class ClusterLimits(BaseModel):
    """Hard scheduling limits applied cluster-wide."""

    model_config = ConfigDict(frozen=True)

    max_pods_per_node: int = Field(
        default=250,
        gt=0,
        description="Maximum pods per node (KubeVirt + system pods)",
    )


class OvercommitConfig(BaseModel):
    """CPU/Memory overcommit ratios for VM normalization.

    ``effective_cpu = vm_cpu / cpu_ratio``
    ``memory_ratio`` is typically 1.0 (no memory overcommit).
    """

    model_config = ConfigDict(frozen=True)

    cpu_ratio: float = Field(
        default=8.0,
        gt=0,
        description="CPU overcommit ratio (e.g. 8:1 means 8 vCPU → 1 physical core)",
    )
    memory_ratio: float = Field(
        default=1.0,
        gt=0,
        description="Memory overcommit ratio (1.0 = no overcommit)",
    )


class VirtOverheads(BaseModel):
    """System overheads subtracted from raw node capacity.

    Sources:
      - OpenShift Virtualization Cluster Sizing Guide
      - Kubelet Auto Sizing Rules (Machine Config Operator)

    The ``ht_efficiency_factor`` accounts for the fact that hyperthreaded
    cores do not deliver 2× the throughput of a physical core.  The Red Hat
    sizing guide recommends 1.5× as the effective multiplier::

        effective_cpus = physical_cores × ht_efficiency_factor   (if HT)
        effective_cpus = physical_cores                          (if no HT)

    Set to ``2.0`` to count every thread as a full core (raw logical count).
    Set to ``1.0`` to ignore hyperthreading entirely (physical cores only).
    """

    model_config = ConfigDict(frozen=True)

    ht_efficiency_factor: float = Field(
        default=1.5,
        ge=1.0,
        le=2.0,
        description=(
            "HT efficiency multiplier applied to physical cores when "
            "threads_per_core > 1 (Red Hat sizing guide default: 1.5)"
        ),
    )
    ocp_virt_core: float = Field(
        default=2.0,
        ge=0,
        description="CPU cores reserved for OCP Virtualization stack",
    )
    ocp_virt_memory_mb: float = Field(
        default=360.0,
        ge=0,
        description="Memory in MB reserved for OCP Virtualization stack",
    )
    eviction_hard_mb: float = Field(
        default=100.0,
        ge=0,
        description="Kubelet hard eviction threshold in MB",
    )


class UtilizationTargets(BaseModel):
    """Target utilization percentages used to derive safety buffers.

    ``target_capacity = usable_capacity × (target / 100)``
    """

    model_config = ConfigDict(frozen=True)

    cpu: float = Field(
        default=85.0,
        gt=0,
        le=100,
        description="CPU utilization target (percent, e.g. 85 = 85%)",
    )
    memory: float = Field(
        default=80.0,
        gt=0,
        le=100,
        description="Memory utilization target (percent, e.g. 80 = 80%)",
    )


class SafetyMargins(BaseModel):
    """Safety configuration: utilization targets and HA spare capacity."""

    model_config = ConfigDict(frozen=True)

    utilization_targets: UtilizationTargets = Field(
        default_factory=UtilizationTargets,
        description="Per-resource utilization ceiling before placement refuses",
    )
    ha_failures_to_tolerate: int = Field(
        default=1,
        ge=0,
        description="Number of simultaneous node failures the cluster must survive",
    )


class AlgorithmWeights(BaseModel):
    """Tunable weights for the K8s-like scoring function (HLD §6.1).

    ``score(node) = α·balance + β·spread + γ·pod_headroom − δ·frag_penalty``

    All weights must be non-negative and sum to 1.0 (±0.01 tolerance).
    """

    model_config = ConfigDict(frozen=True)

    alpha_balance: float = Field(
        default=0.3,
        ge=0,
        le=1.0,
        description="Weight: CPU/Memory balance (NodeResourcesBalancedAllocation)",
    )
    beta_alloc: float = Field(
        default=0.3,
        ge=0,
        le=1.0,
        description="Weight: allocation score (LeastAllocated / MostAllocated)",
    )
    gamma_pod_headroom: float = Field(
        default=0.1,
        ge=0,
        le=1.0,
        description="Weight: pod IP headroom preservation",
    )
    delta_frag_penalty: float = Field(
        default=0.3,
        ge=0,
        le=1.0,
        description="Weight: memory fragmentation penalty",
    )

    @model_validator(mode="after")
    def _weights_must_sum_to_one(self) -> Self:
        total = (
            self.alpha_balance
            + self.beta_alloc
            + self.gamma_pod_headroom
            + self.delta_frag_penalty
        )
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"Algorithm weights must sum to 1.0 (got {total:.4f}). "
                f"Adjust alpha_balance, beta_alloc, gamma_pod_headroom, "
                f"and delta_frag_penalty."
            )
        return self


class PlanConfig(BaseModel):
    """Top-level model for ``config.yaml``.

    Every section has sensible defaults so a minimal or empty config
    file is still valid.
    """

    model_config = ConfigDict(frozen=True)

    cluster_limits: ClusterLimits = Field(default_factory=ClusterLimits)
    overcommit: OvercommitConfig = Field(default_factory=OvercommitConfig)
    virt_overheads: VirtOverheads = Field(default_factory=VirtOverheads)
    safety_margins: SafetyMargins = Field(default_factory=SafetyMargins)
    algorithm_weights: AlgorithmWeights = Field(default_factory=AlgorithmWeights)
    placement_strategy: PlacementStrategy = Field(
        default=PlacementStrategy.SPREAD,
        description=(
            "How inventory nodes are introduced to the placement engine. "
            "'spread' adds all upfront (use all hardware); "
            "'consolidate' adds lazily (minimize active nodes / subscriptions)."
        ),
    )


# ═══════════════════════════════════════════════════════════════════════
# 2. Hardware Profile Models (shared by inventory & catalog)
# ═══════════════════════════════════════════════════════════════════════


class CpuTopology(BaseModel):
    """Physical CPU topology of a bare-metal server.

    ``logical_cpus = sockets × cores_per_socket × threads_per_core``
    ``physical_cores = sockets × cores_per_socket``

    The normalizer uses ``physical_cores`` together with
    ``VirtOverheads.ht_efficiency_factor`` to compute effective CPU
    capacity when hyperthreading is active.
    """

    model_config = ConfigDict(frozen=True)

    sockets: int = Field(..., gt=0, description="Number of physical CPU sockets")
    cores_per_socket: int = Field(..., gt=0, description="Physical cores per socket")
    threads_per_core: int = Field(
        default=1,
        gt=0,
        description="Threads per core (1 = no HT, 2 = HT enabled)",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def physical_cores(self) -> int:
        """Physical core count (sockets × cores_per_socket)."""
        return self.sockets * self.cores_per_socket

    @computed_field  # type: ignore[prop-decorator]
    @property
    def logical_cpus(self) -> int:
        """Total logical CPUs (threads) visible to the OS."""
        return self.sockets * self.cores_per_socket * self.threads_per_core


# ═══════════════════════════════════════════════════════════════════════
# 3. inventory.yaml  (HLD §3.3)
# ═══════════════════════════════════════════════════════════════════════


class InventoryProfile(BaseModel):
    """A brownfield hardware profile (existing nodes, cost = 0)."""

    model_config = ConfigDict(frozen=True)

    profile_name: str = Field(..., min_length=1, description="Profile identifier")
    cpu_topology: CpuTopology
    ram_gb: int = Field(..., gt=0, description="Total physical RAM in GB")
    quantity: int = Field(
        default=1,
        gt=0,
        description="Number of identical nodes with this profile",
    )


class InventoryConfig(BaseModel):
    """Top-level model for ``inventory.yaml``.

    An empty profile list is valid (no brownfield hardware).
    """

    model_config = ConfigDict(frozen=True)

    profiles: list[InventoryProfile] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
# 4. catalog.yaml  (HLD §3.4)
# ═══════════════════════════════════════════════════════════════════════


class CatalogProfile(BaseModel):
    """A greenfield hardware profile available for purchase."""

    model_config = ConfigDict(frozen=True)

    profile_name: str = Field(..., min_length=1, description="Profile identifier")
    cpu_topology: CpuTopology
    ram_gb: int = Field(..., gt=0, description="Total physical RAM in GB")
    cost_weight: float = Field(
        default=1.0,
        gt=0,
        description="Relative cost weight (used by Expander to pick cheapest profile)",
    )


class CatalogConfig(BaseModel):
    """Top-level model for ``catalog.yaml``.

    At least one catalog profile is required for greenfield expansion.
    """

    model_config = ConfigDict(frozen=True)

    profiles: list[CatalogProfile] = Field(
        ...,
        min_length=1,
        description="Available hardware profiles for expansion",
    )
