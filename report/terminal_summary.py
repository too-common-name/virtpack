"""Rich terminal summary output (HLD §8.1, §8.2, §8.3).

Renders the cluster plan summary to the terminal using ``rich``.
All metric computation is performed by pure helper functions to keep
rendering and logic strictly decoupled.

Key metrics:

* **VMware Source Environment** — hosts, capacity, VM demand, overcommit.
* **Cluster Plan Summary** (§8.1) — node counts, utilization, bottleneck.
* **Node Utilization** — per-node CPU/MEM breakdown.
* **Engineering Metrics** (§8.2, §8.3) — CFI and node pressure.
* **Migration Comparison** — VMware vs OCP Virt side-by-side.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from algorithms.scorer import fragmentation_penalty

if TYPE_CHECKING:
    from core.cluster_state import ClusterState
    from core.ha_injector import HAResult
    from loaders.rvtools_parser import RawHost, RawVM
    from models.node import Node
    from models.vm import VM


# ═══════════════════════════════════════════════════════════════════════
# VMware source environment summary
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class VMwareSummary:
    """Pre-migration metrics from the RVTools VMware export.

    Constructed by :func:`compute_vmware_summary` from raw parsed data.
    """

    # Physical hosts
    host_count: int
    total_physical_cores: int  # Σ(sockets × cores_per_socket)
    total_logical_cpus: int  # with HT (threads)
    total_ram_gb: float  # Σ(memory_mb / 1024)

    # VM workload (post-filter: powered on, no templates/SRM)
    vm_count: int
    total_vcpu: int  # Σ(vm.cpu)
    total_vmem_gb: float  # Σ(vm.memory_mb / 1024)

    # Computed ratios
    cpu_overcommit: float  # total_vcpu / total_logical_cpus
    mem_ratio: float  # total_vmem_gb / total_ram_gb


# ═══════════════════════════════════════════════════════════════════════
# OCP Virt per-node detail
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class NodeDetail:
    """Per-node utilization detail for the node table."""

    node_id: str
    profile: str
    cpu_total: float
    cpu_used: float
    cpu_util: float
    memory_total_gb: float
    memory_used_gb: float
    memory_util: float
    vm_count: int
    is_inventory: bool


# ═══════════════════════════════════════════════════════════════════════
# OCP Virt plan summary
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PlanSummary:
    """Pre-computed metrics for the terminal report.

    Constructed by :func:`compute_summary` from the placement results.
    Passed to :func:`render_summary` for Rich output.
    """

    # Node counts
    total_nodes: int
    inventory_nodes: int
    catalog_nodes: int
    ha_nodes: int

    # VM counts
    total_vms: int
    placed_vms: int
    unplaced_vms: int

    # Cluster utilization (aggregated across all nodes)
    cluster_cpu_util: float  # ∈ [0, 1]
    cluster_memory_util: float  # ∈ [0, 1]

    # Cluster-wide absolute capacity
    total_cpu_capacity: float  # sum of cpu_total across all nodes
    total_memory_capacity_mb: float  # sum of memory_total across all nodes

    # Per-node peak
    peak_cpu_util: float  # max node cpu_util
    peak_memory_util: float  # max node memory_util

    # Bottleneck
    bottleneck: str  # "CPU" | "MEMORY" | "BALANCED"

    # Headroom (remaining capacity)
    headroom_cpu: float  # 1 - cluster_cpu_util
    headroom_memory: float  # 1 - cluster_memory_util

    # Engineering metrics (§8.2, §8.3)
    cfi: float  # Cluster Fragmentation Index
    pressure_p95: float  # Node Pressure 95th percentile
    pressure_max: float  # Node Pressure maximum

    # HA
    ha_fully_covered: bool
    ha_deficit_cpu: float
    ha_deficit_memory: float

    # Consolidation — nodes that can be powered off (HLD §1.1 Scenario A2)
    shutdown_candidates: int = 0

    # Per-node breakdown
    node_details: list[NodeDetail] = field(default_factory=list)

    # Unplaced VM names (for detail section)
    unplaced_vm_names: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
# Pure metric calculations
# ═══════════════════════════════════════════════════════════════════════


def _node_pressure(node: Node) -> float:
    """Pressure = max(cpu_util, memory_util) — §8.3."""
    return max(node.cpu_util, node.memory_util)


def _percentile(values: list[float], pct: float) -> float:
    """Compute the *pct*-th percentile (0–100) using nearest-rank method.

    Requires a non-empty sorted list.
    """
    if not values:
        return 0.0
    k = max(0, math.ceil(pct / 100.0 * len(values)) - 1)
    return values[k]


def _compute_cfi(nodes: list[Node]) -> float:
    """Cluster Fragmentation Index = average(stranded_penalty) — §8.2.

    Measures how much remaining capacity is dimensionally stranded
    on average across all nodes.  Lower = remaining capacity is
    better balanced across CPU and memory = good.
    """
    if not nodes:
        return 0.0
    return sum(fragmentation_penalty(n) for n in nodes) / len(nodes)


def _determine_bottleneck(cpu_util: float, mem_util: float) -> str:
    """Return the binding resource dimension."""
    diff = abs(cpu_util - mem_util)
    if diff < 0.02:  # Within 2% → balanced
        return "BALANCED"
    return "CPU" if cpu_util > mem_util else "MEMORY"


# ═══════════════════════════════════════════════════════════════════════
# VMware summary builder
# ═══════════════════════════════════════════════════════════════════════


def compute_vmware_summary(
    *,
    hosts: list[RawHost],
    raw_vms: list[RawVM],
) -> VMwareSummary | None:
    """Build a VMware environment summary from raw RVTools data.

    Returns ``None`` if no hosts are available (e.g. --no-auto-discovery).
    """
    if not hosts:
        return None

    total_physical_cores = sum(h.sockets * h.cores_per_socket for h in hosts)
    total_logical_cpus = sum(
        h.sockets * h.cores_per_socket * (2 if h.ht_active else 1) for h in hosts
    )
    total_ram_gb = sum(h.memory_mb for h in hosts) / 1024.0

    vm_count = len(raw_vms)
    total_vcpu = sum(v.cpu for v in raw_vms)
    total_vmem_gb = sum(v.memory_mb for v in raw_vms) / 1024.0

    cpu_overcommit = total_vcpu / total_logical_cpus if total_logical_cpus > 0 else 0.0
    mem_ratio = total_vmem_gb / total_ram_gb if total_ram_gb > 0 else 0.0

    return VMwareSummary(
        host_count=len(hosts),
        total_physical_cores=total_physical_cores,
        total_logical_cpus=total_logical_cpus,
        total_ram_gb=total_ram_gb,
        vm_count=vm_count,
        total_vcpu=total_vcpu,
        total_vmem_gb=total_vmem_gb,
        cpu_overcommit=cpu_overcommit,
        mem_ratio=mem_ratio,
    )


# ═══════════════════════════════════════════════════════════════════════
# OCP Virt plan summary builder
# ═══════════════════════════════════════════════════════════════════════


def compute_summary(
    *,
    state: ClusterState,
    vms: list[VM],
    unplaced: list[VM],
    ha_result: HAResult | None = None,
    unused_inventory: list[Node] | None = None,
) -> PlanSummary:
    """Compute all summary metrics from the final placement state.

    Parameters
    ----------
    state : ClusterState
        The final cluster state (post-placement, post-HA injection).
    vms : list[VM]
        All input VMs (placed + unplaced).
    unplaced : list[VM]
        VMs that could not be placed.
    ha_result : HAResult | None
        HA injection outcome.  ``None`` if HA was skipped.
    unused_inventory : list[Node] | None
        Inventory nodes that were NOT activated during consolidation.
        ``None`` or empty in spread mode.
    """
    nodes = state.nodes
    node_vm_map = state.node_vm_map
    ha_nodes_count = len(ha_result.nodes_added) if ha_result else 0

    # ── Node counts ───────────────────────────────────────────────
    total_nodes = len(nodes)
    inventory_count = len(state.inventory_nodes)
    catalog_count = len(state.catalog_nodes)

    # ── VM counts ─────────────────────────────────────────────────
    total_vms = len(vms)
    placed_vms = state.total_placed_vms
    unplaced_vms = len(unplaced)

    # ── Cluster-wide utilization ──────────────────────────────────
    total_cpu_cap = sum(n.cpu_total for n in nodes)
    total_cpu_used = sum(n.cpu_used for n in nodes)
    total_mem_cap = sum(n.memory_total for n in nodes)
    total_mem_used = sum(n.memory_used for n in nodes)

    cluster_cpu_util = total_cpu_used / total_cpu_cap if total_cpu_cap > 0 else 0.0
    cluster_mem_util = total_mem_used / total_mem_cap if total_mem_cap > 0 else 0.0

    # ── Per-node peaks ────────────────────────────────────────────
    active = state.active_nodes
    peak_cpu = max((n.cpu_util for n in active), default=0.0)
    peak_mem = max((n.memory_util for n in active), default=0.0)

    # ── Engineering metrics ───────────────────────────────────────
    cfi = _compute_cfi(nodes)
    pressures = sorted(_node_pressure(n) for n in active)
    p95 = _percentile(pressures, 95)
    p_max = pressures[-1] if pressures else 0.0

    # ── HA ────────────────────────────────────────────────────────
    ha_covered = ha_result.fully_covered if ha_result else True
    ha_def_cpu = ha_result.deficit_cpu if ha_result else 0.0
    ha_def_mem = ha_result.deficit_memory if ha_result else 0.0

    # ── Consolidation ─────────────────────────────────────────────
    shutdown = len(unused_inventory) if unused_inventory else 0

    # ── Per-node details ──────────────────────────────────────────
    details: list[NodeDetail] = []
    for n in nodes:
        vm_names = node_vm_map.get(n.id, [])
        details.append(
            NodeDetail(
                node_id=n.id,
                profile=n.profile,
                cpu_total=n.cpu_total,
                cpu_used=n.cpu_used,
                cpu_util=n.cpu_util,
                memory_total_gb=n.memory_total / 1024.0,
                memory_used_gb=n.memory_used / 1024.0,
                memory_util=n.memory_util,
                vm_count=len(vm_names),
                is_inventory=n.is_inventory,
            )
        )
    # Sort by VM count descending (busiest first)
    details.sort(key=lambda d: d.vm_count, reverse=True)

    return PlanSummary(
        total_nodes=total_nodes,
        inventory_nodes=inventory_count,
        catalog_nodes=catalog_count,
        ha_nodes=ha_nodes_count,
        total_vms=total_vms,
        placed_vms=placed_vms,
        unplaced_vms=unplaced_vms,
        cluster_cpu_util=cluster_cpu_util,
        cluster_memory_util=cluster_mem_util,
        total_cpu_capacity=total_cpu_cap,
        total_memory_capacity_mb=total_mem_cap,
        peak_cpu_util=peak_cpu,
        peak_memory_util=peak_mem,
        bottleneck=_determine_bottleneck(cluster_cpu_util, cluster_mem_util),
        headroom_cpu=1.0 - cluster_cpu_util,
        headroom_memory=1.0 - cluster_mem_util,
        cfi=cfi,
        pressure_p95=p95,
        pressure_max=p_max,
        ha_fully_covered=ha_covered,
        ha_deficit_cpu=ha_def_cpu,
        ha_deficit_memory=ha_def_mem,
        shutdown_candidates=shutdown,
        node_details=details,
        unplaced_vm_names=[vm.name for vm in unplaced],
    )


# ═══════════════════════════════════════════════════════════════════════
# Rich rendering
# ═══════════════════════════════════════════════════════════════════════


def render_vmware_summary(vmware: VMwareSummary) -> None:
    """Print the VMware source environment summary panel."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    console = Console()
    lines = Text()

    lines.append(f"Physical Hosts: {vmware.host_count}\n", style="bold")
    lines.append(
        f"Total Physical Cores: {vmware.total_physical_cores}"
        f"  ({vmware.total_logical_cpus} logical w/ HT)\n"
    )
    lines.append(f"Total RAM: {vmware.total_ram_gb:,.0f} GB\n\n")

    lines.append(f"VMs to Migrate: {vmware.vm_count}\n", style="bold")
    lines.append(f"Total vCPU Demand: {vmware.total_vcpu:,}\n")
    lines.append(f"Total Memory Demand: {vmware.total_vmem_gb:,.0f} GB\n\n")

    lines.append(f"VMware vCPU:pCPU Ratio: {vmware.cpu_overcommit:.1f}:1\n")
    lines.append(f"VMware vMEM:pMEM Ratio: {vmware.mem_ratio:.2f}:1")

    console.print(
        Panel(
            lines,
            title="[bold]VMware Source Environment[/bold]",
            border_style="yellow",
            width=55,
        )
    )


def render_node_table(details: list[NodeDetail]) -> None:
    """Print per-node utilization as a Rich table."""
    from rich.console import Console
    from rich.table import Table

    if not details:
        return

    console = Console()
    table = Table(
        title="Node Utilization",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Node", style="bold")
    table.add_column("Profile")
    table.add_column("VMs", justify="right")
    table.add_column("CPU Used", justify="right")
    table.add_column("CPU %", justify="right")
    table.add_column("MEM Used", justify="right")
    table.add_column("MEM %", justify="right")
    table.add_column("Type", justify="center")

    for d in details:
        cpu_style = "red" if d.cpu_util > 0.85 else ""
        mem_style = "red" if d.memory_util > 0.85 else ""
        node_type = "inv" if d.is_inventory else "cat"

        table.add_row(
            d.node_id,
            d.profile,
            str(d.vm_count),
            f"{d.cpu_used:.1f}/{d.cpu_total:.0f}",
            f"[{cpu_style}]{d.cpu_util:.0%}[/{cpu_style}]" if cpu_style else f"{d.cpu_util:.0%}",
            f"{d.memory_used_gb:.0f}/{d.memory_total_gb:.0f} GB",
            f"[{mem_style}]{d.memory_util:.0%}[/{mem_style}]"
            if mem_style
            else f"{d.memory_util:.0%}",
            node_type,
        )

    console.print(table)


def render_comparison(vmware: VMwareSummary, plan: PlanSummary) -> None:
    """Print the VMware → OCP Virt migration comparison table."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()

    table = Table(
        title="Migration Comparison: VMware → OCP Virt",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Metric", style="bold")
    table.add_column("VMware", justify="right")
    table.add_column("OCP Virt", justify="right")
    table.add_column("Delta", justify="right")

    # Active nodes
    ocp_nodes = plan.total_nodes
    node_delta = ocp_nodes - vmware.host_count
    node_delta_str = f"{node_delta:+d}" if node_delta != 0 else "—"
    table.add_row(
        "Active Nodes",
        str(vmware.host_count),
        str(ocp_nodes),
        node_delta_str,
    )

    # CPU capacity
    ocp_cpu = plan.total_cpu_capacity
    cpu_delta = ocp_cpu - vmware.total_logical_cpus
    cpu_pct = cpu_delta / vmware.total_logical_cpus * 100 if vmware.total_logical_cpus else 0
    table.add_row(
        "CPU Capacity (cores)",
        f"{vmware.total_logical_cpus:,} logical",
        f"{ocp_cpu:,.0f} usable",
        f"{cpu_delta:+,.0f} ({cpu_pct:+.0f}%)",
    )

    # Memory capacity
    ocp_mem_gb = plan.total_memory_capacity_mb / 1024.0
    mem_delta = ocp_mem_gb - vmware.total_ram_gb
    mem_pct = mem_delta / vmware.total_ram_gb * 100 if vmware.total_ram_gb else 0
    table.add_row(
        "Memory Capacity (GB)",
        f"{vmware.total_ram_gb:,.0f} raw",
        f"{ocp_mem_gb:,.0f} usable",
        f"{mem_delta:+,.0f} ({mem_pct:+.0f}%)",
    )

    # vCPU demand
    table.add_row(
        "vCPU Demand",
        f"{vmware.total_vcpu:,} vCPUs",
        "(normalized by overcommit)",
        "",
    )

    # Memory demand (same in both — 1:1)
    table.add_row(
        "Memory Demand (GB)",
        f"{vmware.total_vmem_gb:,.0f}",
        f"{vmware.total_vmem_gb:,.0f} (1:1)",
        "—",
    )

    console.print(table)

    # Explanatory note
    console.print(
        Panel(
            "[dim]OCP Virt usable capacity is lower because:[/dim]\n"
            "  • HT efficiency: 2 threads ≠ 2 cores (1.5x factor)\n"
            "  • System overheads: kubelet + OCP Virt operator reserved\n"
            "  • No memory overcommit: 1:1 reservation (VMware uses TPS/ballooning)\n"
            "  • Safety margins: utilization targets reserve headroom",
            title="[dim]ⓘ  Why capacity differs[/dim]",
            border_style="dim",
            width=62,
        )
    )


def render_summary(
    summary: PlanSummary,
    vmware: VMwareSummary | None = None,
) -> None:
    """Print the full report to the terminal using Rich.

    Renders (in order):
    1. VMware source environment (if available)
    2. OCP Virt cluster plan summary
    3. Node utilization table
    4. Engineering metrics
    5. Migration comparison (if VMware data available)
    6. HA warnings
    7. Unplaced VM details
    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    console = Console()

    # ── 1. VMware source environment ──────────────────────────────
    if vmware is not None:
        render_vmware_summary(vmware)
        console.print()

    # ── 2. Main summary panel ─────────────────────────────────────
    lines = Text()

    # Node breakdown
    existing = summary.inventory_nodes
    new = summary.catalog_nodes
    lines.append(f"Nodes Required: {summary.total_nodes}", style="bold")
    lines.append(f"  ({existing} existing + {new} new")
    if summary.ha_nodes > 0:
        lines.append(f", {summary.ha_nodes} HA spare")
    lines.append(")\n")

    # Consolidation: show nodes available for shutdown
    if summary.shutdown_candidates > 0:
        total_available = existing + summary.shutdown_candidates
        lines.append(
            f"Nodes Available for Shutdown: {summary.shutdown_candidates}"
            f"  (of {total_available} total inventory)",
            style="bold green",
        )
        lines.append("\n")

    # OCP subscriptions (each pair of sockets = 1 subscription)
    # For simplicity: total nodes = subscriptions (bare-metal 1:1)
    lines.append(f"Required OCP Subscriptions: {summary.total_nodes}\n")
    if summary.shutdown_candidates > 0:
        saved = summary.shutdown_candidates
        lines.append(f"Subscription Savings: {saved} node(s) can be powered off\n", style="green")
    lines.append("\n")

    # Utilization
    lines.append(f"Peak CPU Utilization:    {summary.peak_cpu_util:6.1%}\n")
    lines.append(f"Peak Memory Utilization: {summary.peak_memory_util:6.1%}\n\n")

    # Bottleneck
    color = "red" if summary.bottleneck != "BALANCED" else "green"
    lines.append("Cluster Bottleneck: ")
    lines.append(summary.bottleneck, style=f"bold {color}")
    lines.append("\n\n")

    # Headroom
    lines.append("Remaining Capacity (Headroom):\n")
    lines.append(f"  CPU:    {summary.headroom_cpu:6.1%}\n")
    lines.append(f"  Memory: {summary.headroom_memory:6.1%}\n\n")

    # Unplaced
    if summary.unplaced_vms > 0:
        lines.append(f"Unplaced VMs: {summary.unplaced_vms}", style="bold red")
    else:
        lines.append("Unplaced VMs: 0", style="bold green")

    console.print(
        Panel(lines, title="[bold]OCP Virt Cluster Plan[/bold]", border_style="blue", width=55)
    )

    # ── 3. Node utilization table ─────────────────────────────────
    if summary.node_details:
        render_node_table(summary.node_details)

    # ── 4. Engineering metrics table ──────────────────────────────
    eng_table = Table(title="Engineering Metrics", show_header=True, header_style="bold cyan")
    eng_table.add_column("Metric", style="bold")
    eng_table.add_column("Value", justify="right")

    eng_table.add_row("Cluster Fragmentation Index (CFI)", f"{summary.cfi:.4f}")
    eng_table.add_row("Node Pressure P95", f"{summary.pressure_p95:.2f}")
    eng_table.add_row("Node Pressure Max", f"{summary.pressure_max:.2f}")
    eng_table.add_row("Cluster CPU Utilization", f"{summary.cluster_cpu_util:.1%}")
    eng_table.add_row("Cluster Memory Utilization", f"{summary.cluster_memory_util:.1%}")

    console.print(eng_table)

    # ── 5. Migration comparison ───────────────────────────────────
    if vmware is not None:
        render_comparison(vmware, summary)

    # ── 6. HA status ──────────────────────────────────────────────
    if not summary.ha_fully_covered:
        console.print(
            Panel(
                f"[bold red]HA DEFICIT[/bold red]\n"
                f"  CPU:    {summary.ha_deficit_cpu:.1f} cores uncovered\n"
                f"  Memory: {summary.ha_deficit_memory:.0f} MB uncovered",
                title="[bold red]⚠ HA Warning[/bold red]",
                border_style="red",
                width=55,
            )
        )

    # ── 7. Unplaced detail ────────────────────────────────────────
    if summary.unplaced_vm_names:
        unplaced_text = "\n".join(f"  • {name}" for name in summary.unplaced_vm_names[:20])
        if len(summary.unplaced_vm_names) > 20:
            unplaced_text += f"\n  ... and {len(summary.unplaced_vm_names) - 20} more"
        console.print(
            Panel(
                unplaced_text,
                title=f"[bold red]Unplaced VMs ({summary.unplaced_vms})[/bold red]",
                border_style="red",
                width=55,
            )
        )
