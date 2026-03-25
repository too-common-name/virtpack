"""``virtpack plan`` CLI entry point (LLD §4).

Orchestrates the full ETL → Normalize → Place → HA → Report pipeline.

Usage::

    virtpack plan \\
        --rvtools rvtools.xlsx \\
        --config config.yaml \\
        --catalog catalog.yaml \\
        --inventory inventory.yaml \\
        --output ./out \\
        --strategy spread \\
        --debug
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console

from models.config import PlacementStrategy

if TYPE_CHECKING:
    from models.config import PlanConfig
    from models.node import Node

# ── Application ───────────────────────────────────────────────────────
app = typer.Typer(
    name="virtpack",
    help="Deterministic OpenShift Virtualization capacity planner.",
    add_completion=False,
)

_console = Console(stderr=True)


# ═══════════════════════════════════════════════════════════════════════
# vHost auto-discovery bridge
# ═══════════════════════════════════════════════════════════════════════


def _build_autodiscovery_nodes(
    rvtools_path: Path,
    config: PlanConfig,
) -> tuple[list[Node], list[RawHost]]:
    """Build inventory nodes from the RVTools ``vHost`` sheet.

    Returns
    -------
    tuple[list[Node], list[RawHost]]
        (normalized inventory nodes, raw hosts for VMware summary).
        Both lists are empty if the sheet cannot be parsed (non-fatal).
    """
    from core.normalizer import normalize_node_capacity
    from loaders.rvtools_parser import RVToolsParseError, parse_vhost
    from models.config import CpuTopology
    from models.node import Node as _Node  # runtime import (used for construction)

    try:
        hosts = parse_vhost(rvtools_path)
    except RVToolsParseError as exc:
        _console.print(f"[yellow]⚠ vHost auto-discovery skipped:[/yellow] {exc}")
        return [], []

    if not hosts:
        return [], []

    nodes: list[Node] = []
    for host in hosts:
        topology = CpuTopology(
            sockets=host.sockets,
            cores_per_socket=host.cores_per_socket,
            threads_per_core=2 if host.ht_active else 1,
        )
        cpu_total, memory_total, pods_total = normalize_node_capacity(
            topology=topology,
            ram_gb=max(1, host.memory_mb // 1024),  # MB → GB (floor)
            config=config,
        )
        nodes.append(
            _Node.new_inventory(
                profile=f"autodiscovered-{host.sockets}s{host.cores_per_socket}c",
                index=0,
                cpu_total=cpu_total,
                memory_total=memory_total,
                pods_total=pods_total,
                id_override=host.name,
            )
        )
    return nodes, hosts


# ═══════════════════════════════════════════════════════════════════════
# Main command
# ═══════════════════════════════════════════════════════════════════════


@app.command()
def plan(
    rvtools: Annotated[
        Path,
        typer.Option("--rvtools", help="RVTools .xlsx export (vInfo + optional vHost)"),
    ],
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Global config.yaml (defaults used if omitted)"),
    ] = None,
    catalog_path: Annotated[
        Path | None,
        typer.Option("--catalog", help="Greenfield hardware catalog.yaml"),
    ] = None,
    inventory_path: Annotated[
        Path | None,
        typer.Option("--inventory", help="Brownfield inventory.yaml"),
    ] = None,
    output: Annotated[
        Path,
        typer.Option("--output", help="Output directory for placement_map.csv"),
    ] = Path("./out"),
    strategy: Annotated[
        PlacementStrategy,
        typer.Option(
            "--strategy",
            help=(
                "Placement strategy: 'spread' distributes VMs across all "
                "inventory nodes; 'consolidate' packs VMs tightly and "
                "reports nodes that can be powered off."
            ),
            case_sensitive=False,
        ),
    ] = PlacementStrategy.SPREAD,
    debug: Annotated[
        bool,
        typer.Option("--debug", help="Verbose placement logs"),
    ] = False,
    no_auto_discovery: Annotated[
        bool,
        typer.Option("--no-auto-discovery", help="Disable vHost inventory auto-parsing"),
    ] = False,
) -> None:
    """Run the capacity planning simulation.

    Parses workloads, normalizes resources, simulates placement using
    a K8s-like scheduler, injects HA spare capacity, and generates
    an auditable placement map.
    """
    from core.cluster_state import ClusterState
    from core.ha_injector import inject_ha_nodes
    from core.normalizer import build_inventory_nodes, normalize_vm
    from core.placement_engine import run_placement
    from loaders.rvtools_parser import RVToolsParseError, parse_vinfo
    from loaders.yaml_loader import (
        ConfigLoadError,
        load_catalog_config,
        load_inventory_config,
        load_plan_config,
    )
    from report.csv_exporter import export_placement_csv
    from report.terminal_summary import compute_summary, compute_vmware_summary, render_summary

    out = Console()

    # ══════════════════════════════════════════════════════════════════
    # 1. EXTRACT — Load configs + parse RVTools
    # ══════════════════════════════════════════════════════════════════
    try:
        plan_config = load_plan_config(config_path)
    except ConfigLoadError as exc:
        _console.print(f"[red]Error loading config:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    # CLI --strategy overrides config.yaml placement_strategy
    effective_strategy = strategy

    try:
        inventory_config = load_inventory_config(inventory_path)
    except ConfigLoadError as exc:
        _console.print(f"[red]Error loading inventory:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    try:
        catalog_config = load_catalog_config(catalog_path)
    except ConfigLoadError as exc:
        _console.print(f"[red]Error loading catalog:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    try:
        raw_vms = parse_vinfo(rvtools)
    except RVToolsParseError as exc:
        _console.print(f"[red]Error parsing RVTools:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if debug:
        out.print(f"[dim]Parsed {len(raw_vms)} VMs from vInfo[/dim]")
        out.print(f"[dim]Strategy: {effective_strategy.value}[/dim]")

    # ══════════════════════════════════════════════════════════════════
    # 2. TRANSFORM — Normalize VMs + Build Nodes
    # ══════════════════════════════════════════════════════════════════
    vms = [
        normalize_vm(
            name=rv.name,
            raw_cpu=float(rv.cpu),
            raw_memory_mb=float(rv.memory_mb),
            overcommit=plan_config.overcommit,
        )
        for rv in raw_vms
    ]

    if debug:
        out.print(
            f"[dim]Normalized {len(vms)} VMs (cpu_ratio={plan_config.overcommit.cpu_ratio})[/dim]"
        )

    # Build inventory nodes from YAML profiles
    inv_nodes = build_inventory_nodes(inventory_config, plan_config)

    # vHost auto-discovery (unless disabled)
    discovered_hosts: list[RawHost] = []
    if not no_auto_discovery:
        auto_nodes, discovered_hosts = _build_autodiscovery_nodes(rvtools, plan_config)
        inv_nodes.extend(auto_nodes)
        if debug and auto_nodes:
            out.print(f"[dim]Auto-discovered {len(auto_nodes)} hosts from vHost[/dim]")

    if debug:
        out.print(f"[dim]Total inventory nodes: {len(inv_nodes)}[/dim]")

    # ══════════════════════════════════════════════════════════════════
    # 3. PLACEMENT — Filter → Expand → Score → Bind
    # ══════════════════════════════════════════════════════════════════

    if effective_strategy == PlacementStrategy.CONSOLIDATE:
        # Consolidate: inventory nodes held in pool, pulled lazily
        state = ClusterState()
        inventory_pool = inv_nodes
        if debug:
            out.print(
                f"[dim]Consolidate mode: {len(inventory_pool)} inventory nodes in pool[/dim]"
            )
    else:
        # Spread (default): all inventory nodes added upfront
        state = ClusterState(inv_nodes)
        inventory_pool = None

    result = run_placement(
        vms=vms,
        state=state,
        config=plan_config,
        catalog=catalog_config,
        inventory_pool=inventory_pool,
    )

    catalog_node_count = len(state.catalog_nodes)
    if debug:
        out.print(
            f"[dim]Placement complete: "
            f"{state.total_placed_vms} placed, "
            f"{len(result.unplaced)} unplaced, "
            f"{catalog_node_count} catalog nodes added[/dim]"
        )
        if result.unused_inventory:
            out.print(
                f"[dim]Unused inventory nodes: {len(result.unused_inventory)} "
                f"(can be shut down)[/dim]"
            )

    # ══════════════════════════════════════════════════════════════════
    # 4. HA INJECTION
    # ══════════════════════════════════════════════════════════════════
    ha_result = inject_ha_nodes(
        state=state,
        config=plan_config,
        catalog=catalog_config,
    )

    if debug and ha_result.nodes_added:
        out.print(f"[dim]HA injection: {len(ha_result.nodes_added)} spare nodes added[/dim]")

    # ══════════════════════════════════════════════════════════════════
    # 5. REPORTING
    # ══════════════════════════════════════════════════════════════════

    # Ensure output directory exists
    output.mkdir(parents=True, exist_ok=True)

    # CSV placement map
    csv_path = output / "placement_map.csv"
    rows_written = export_placement_csv(state=state, vms=vms, path=csv_path)
    if debug:
        out.print(f"[dim]Wrote {rows_written} rows to {csv_path}[/dim]")

    # VMware source summary (available when vHost was auto-discovered)
    vmware_summary = compute_vmware_summary(hosts=discovered_hosts, raw_vms=raw_vms)

    # Terminal summary
    summary = compute_summary(
        state=state,
        vms=vms,
        unplaced=result.unplaced,
        ha_result=ha_result,
        unused_inventory=result.unused_inventory if result.unused_inventory else None,
    )
    render_summary(summary, vmware=vmware_summary)

    out.print(f"\n[bold green]✓[/bold green] Placement map saved to [bold]{csv_path}[/bold]")

    # Exit code: non-zero if VMs are unplaced
    if result.unplaced:
        raise typer.Exit(code=2)


# ═══════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app()
