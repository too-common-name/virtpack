"""Rich terminal summary output (HLD §8.1, §8.2, §8.3).

Renders the cluster plan summary to the terminal using ``rich``.
All metric computation is performed by pure helper functions to keep
rendering and logic strictly decoupled.

Key metrics:

* **Cluster Plan Summary** (§8.1) — node counts, utilization, bottleneck.
* **Cluster Fragmentation Index** (§8.2) — average frag penalty.
* **Node Pressure Index** (§8.3) — P95 and max pressure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from algorithms.scorer import fragmentation_penalty

if TYPE_CHECKING:
    from core.cluster_state import ClusterState
    from core.ha_injector import HAResult
    from models.node import Node
    from models.vm import VM


# ═══════════════════════════════════════════════════════════════════════
# Summary data container
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
    """Cluster Fragmentation Index = average(fragmentation_penalty) — §8.2."""
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
# Summary builder
# ═══════════════════════════════════════════════════════════════════════


def compute_summary(
    *,
    state: ClusterState,
    vms: list[VM],
    unplaced: list[VM],
    ha_result: HAResult | None = None,
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
    """
    nodes = state.nodes
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
        unplaced_vm_names=[vm.name for vm in unplaced],
    )


# ═══════════════════════════════════════════════════════════════════════
# Rich rendering
# ═══════════════════════════════════════════════════════════════════════


def render_summary(summary: PlanSummary) -> None:
    """Print the cluster plan summary to the terminal using Rich.

    Layout matches HLD §8.1.
    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    console = Console()

    # ── Main summary panel ────────────────────────────────────────
    lines = Text()

    # Node breakdown
    existing = summary.inventory_nodes
    new = summary.catalog_nodes
    lines.append(f"Nodes Required: {summary.total_nodes}", style="bold")
    lines.append(f"  ({existing} existing + {new} new")
    if summary.ha_nodes > 0:
        lines.append(f", {summary.ha_nodes} HA spare")
    lines.append(")\n")

    # OCP subscriptions (each pair of sockets = 1 subscription)
    # For simplicity: total nodes = subscriptions (bare-metal 1:1)
    lines.append(f"Required OCP Subscriptions: {summary.total_nodes}\n\n")

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
        Panel(lines, title="[bold]Cluster Plan Summary[/bold]", border_style="blue", width=50)
    )

    # ── Engineering metrics table ─────────────────────────────────
    eng_table = Table(title="Engineering Metrics", show_header=True, header_style="bold cyan")
    eng_table.add_column("Metric", style="bold")
    eng_table.add_column("Value", justify="right")

    eng_table.add_row("Cluster Fragmentation Index (CFI)", f"{summary.cfi:.4f}")
    eng_table.add_row("Node Pressure P95", f"{summary.pressure_p95:.2f}")
    eng_table.add_row("Node Pressure Max", f"{summary.pressure_max:.2f}")
    eng_table.add_row("Cluster CPU Utilization", f"{summary.cluster_cpu_util:.1%}")
    eng_table.add_row("Cluster Memory Utilization", f"{summary.cluster_memory_util:.1%}")

    console.print(eng_table)

    # ── HA status ─────────────────────────────────────────────────
    if not summary.ha_fully_covered:
        console.print(
            Panel(
                f"[bold red]HA DEFICIT[/bold red]\n"
                f"  CPU:    {summary.ha_deficit_cpu:.1f} cores uncovered\n"
                f"  Memory: {summary.ha_deficit_memory:.0f} MB uncovered",
                title="[bold red]⚠ HA Warning[/bold red]",
                border_style="red",
                width=50,
            )
        )

    # ── Unplaced detail ───────────────────────────────────────────
    if summary.unplaced_vm_names:
        unplaced_text = "\n".join(f"  • {name}" for name in summary.unplaced_vm_names[:20])
        if len(summary.unplaced_vm_names) > 20:
            unplaced_text += f"\n  ... and {len(summary.unplaced_vm_names) - 20} more"
        console.print(
            Panel(
                unplaced_text,
                title=f"[bold red]Unplaced VMs ({summary.unplaced_vms})[/bold red]",
                border_style="red",
                width=50,
            )
        )
