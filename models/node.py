"""Domain model for physical bare-metal OpenShift nodes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from models.vm import VM


class Node(BaseModel):
    """A physical bare-metal node in the OpenShift cluster.

    Tracks total and used resources across all three scheduling dimensions
    (CPU, Memory, Pods).  The ``*_used`` fields are **mutable** — they are
    updated by ``ClusterState.place()`` / ``ClusterState.unplace()`` to
    enable O(1) rollback during Lookahead simulation.

    Memory is stored in MB for unit-consistency with :class:`VM`.
    CPU capacity is in logical cores (post-overhead subtraction).

    Construction
    ------------
    Use the factory class methods instead of the raw constructor to
    enforce inventory/catalog invariants:

        Node.new_inventory(...)   # cost_weight=0.0, is_inventory=True
        Node.new_catalog(...)     # cost_weight>0,   is_inventory=False
    """

    model_config = ConfigDict(strict=True)

    # ── Identity ────────────────────────────────────────────────────────
    id: str = Field(
        ...,
        min_length=1,
        description="Unique node identifier (e.g. 'inv-r740-01', 'cat-r760-03')",
    )
    profile: str = Field(
        ...,
        min_length=1,
        description="Hardware profile name from inventory/catalog YAML",
    )

    # ── Capacity (set once at creation after normalization + safety) ────
    cpu_total: float = Field(
        ...,
        gt=0,
        description="Schedulable CPU capacity in logical cores (post-overhead, post-safety)",
    )
    memory_total: float = Field(
        ...,
        gt=0,
        description="Schedulable memory in MB (post-overhead, post-safety)",
    )
    pods_total: int = Field(
        ...,
        gt=0,
        description="Max pods from cluster_limits.max_pods_per_node",
    )

    # ── Usage (mutated during placement loop) ──────────────────────────
    cpu_used: float = Field(
        default=0.0,
        ge=0,
        description="CPU cores currently allocated to placed VMs",
    )
    memory_used: float = Field(
        default=0.0,
        ge=0,
        description="Memory in MB currently allocated to placed VMs",
    )
    pods_used: int = Field(
        default=0,
        ge=0,
        description="Number of pods (VMs) currently placed on this node",
    )

    # ── Metadata ───────────────────────────────────────────────────────
    cost_weight: float = Field(
        ...,
        ge=0,
        description="Cost weight (0.0 for inventory/brownfield, >0 for catalog/greenfield)",
    )
    is_inventory: bool = Field(
        ...,
        description="True = brownfield (existing hardware), False = greenfield (new purchase)",
    )

    # ── Factory class methods ──────────────────────────────────────────

    @classmethod
    def new_inventory(
        cls,
        *,
        id: str,
        profile: str,
        cpu_total: float,
        memory_total: float,
        pods_total: int,
    ) -> Node:
        """Create a brownfield inventory node (cost = 0).

        Enforces: ``cost_weight=0.0`` and ``is_inventory=True``.
        """
        return cls(
            id=id,
            profile=profile,
            cpu_total=cpu_total,
            memory_total=memory_total,
            pods_total=pods_total,
            cost_weight=0.0,
            is_inventory=True,
        )

    @classmethod
    def new_catalog(
        cls,
        *,
        id: str,
        profile: str,
        cpu_total: float,
        memory_total: float,
        pods_total: int,
        cost_weight: float,
    ) -> Node:
        """Create a greenfield catalog node (new purchase).

        Enforces: ``is_inventory=False`` and ``cost_weight > 0``.
        """
        if cost_weight <= 0:
            raise ValueError(
                f"Catalog nodes must have cost_weight > 0, got {cost_weight}"
            )
        return cls(
            id=id,
            profile=profile,
            cpu_total=cpu_total,
            memory_total=memory_total,
            pods_total=pods_total,
            cost_weight=cost_weight,
            is_inventory=False,
        )

    # ── Derived properties for the Scorer (HLD §6.1) ──────────────────

    @property
    def cpu_remaining(self) -> float:
        """Unused CPU cores available for placement."""
        return self.cpu_total - self.cpu_used

    @property
    def memory_remaining(self) -> float:
        """Unused memory in MB available for placement."""
        return self.memory_total - self.memory_used

    @property
    def pods_remaining(self) -> int:
        """Unused pod slots available for placement."""
        return self.pods_total - self.pods_used

    @property
    def cpu_util(self) -> float:
        """CPU utilization ratio ∈ [0.0, 1.0].

        Used by scorer for ``balance_score`` and ``spread_score``.
        """
        if self.cpu_total == 0.0:
            return 0.0
        return self.cpu_used / self.cpu_total

    @property
    def memory_util(self) -> float:
        """Memory utilization ratio ∈ [0.0, 1.0].

        Used by scorer for ``balance_score`` and ``spread_score``.
        """
        if self.memory_total == 0.0:
            return 0.0
        return self.memory_used / self.memory_total

    # ── Placement helpers ──────────────────────────────────────────────

    def fits(self, vm: VM) -> bool:
        """Return True if *vm* can be placed without exceeding any dimension.

        Checks all three scheduling constraints:
        ``cpu_used + vm.cpu ≤ cpu_total``
        ``memory_used + vm.memory_mb ≤ memory_total``
        ``pods_used + vm.pods ≤ pods_total``
        """
        return (
            self.cpu_used + vm.cpu <= self.cpu_total
            and self.memory_used + vm.memory_mb <= self.memory_total
            and self.pods_used + vm.pods <= self.pods_total
        )
