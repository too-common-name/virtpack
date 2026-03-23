"""Domain model for Virtual Machine workloads."""

from pydantic import BaseModel, ConfigDict, Field


class VM(BaseModel):
    """A normalized virtual machine workload unit.

    Represents a single VM after ETL filtering and resource normalization.
    Memory is stored in MB to match RVTools vInfo native units.
    CPU is stored as effective vCPU (post-overcommit normalization).

    Immutable: once created during the Transform phase, a VM's resource
    requests never change.
    """

    model_config = ConfigDict(strict=True, frozen=True)

    name: str = Field(
        ...,
        min_length=1,
        description="Unique VM identifier from RVTools vInfo 'VM' column",
    )
    cpu: float = Field(
        ...,
        gt=0,
        description="Effective vCPU request (post-overcommit: vm_cpu / cpu_ratio)",
    )
    memory_mb: float = Field(
        ...,
        gt=0,
        description="Memory request in MB (RVTools native unit)",
    )
    pods: int = Field(
        default=1,
        ge=1,
        description="Pod count consumed by this VM (always 1 for KubeVirt virt-launcher)",
    )
