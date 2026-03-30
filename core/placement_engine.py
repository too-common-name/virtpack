"""Placement engine — the main simulation loop (HLD §6.2).

Orchestrates the full filter → expand → score → bind pipeline
with Lookahead k=2.

Pure orchestration — all scoring and expansion are delegated to
``algorithms.scorer`` and ``algorithms.expander``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from algorithms.expander import expand
from algorithms.scorer import score_node
from models.config import PlacementStrategy

if TYPE_CHECKING:
    from core.cluster_state import ClusterState
    from models.config import AlgorithmWeights, CatalogConfig, PlanConfig
    from models.node import Node
    from models.vm import VM


# ═══════════════════════════════════════════════════════════════════════
# Result container
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class PlacementResult:
    """Outcome of a full placement run."""

    state: ClusterState
    unplaced: list[VM] = field(default_factory=list)
    unused_inventory: list[Node] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
# Engine
# ═══════════════════════════════════════════════════════════════════════

# Lookahead penalty applied when the *next* VM cannot fit on the node
# after the current VM is tentatively placed.  Must be large enough to
# dominate the score range (~[-1, 1]) so the node is strongly disfavoured.
_LOOKAHEAD_PENALTY: float = -10.0

# Weight applied to the Lookahead component of the total score.
_LOOKAHEAD_WEIGHT: float = 0.5


def _pull_from_pool(
    vm: VM,
    pool: list[Node],
    *,
    remaining_cpu: float = 0.0,
    remaining_mem: float = 0.0,
) -> Node | None:
    """Pick the best inventory node from *pool* that can fit *vm*.

    When *remaining_cpu* / *remaining_mem* are provided (total demand of
    all VMs still to be placed), the heuristic picks the **smallest**
    eligible node whose total capacity covers that remaining demand.
    If no single node covers it, fall back to the **largest** eligible
    node to maximise co-location and minimise cascading pulls.

    When remaining demand is zero (or not provided), fall back to the
    largest node — the safe default.

    The chosen node is **removed** from *pool* and returned (or ``None``
    if no node in the pool fits).
    """
    eligible = [n for n in pool if n.fits(vm)]
    if not eligible:
        return None

    if remaining_cpu > 0 or remaining_mem > 0:
        covers = [
            n for n in eligible if n.cpu_total >= remaining_cpu and n.memory_total >= remaining_mem
        ]
        if covers:
            best = min(covers, key=lambda n: (n.memory_total, n.cpu_total))
        else:
            best = max(eligible, key=lambda n: (n.memory_total, n.cpu_total))
    else:
        best = max(eligible, key=lambda n: (n.memory_total, n.cpu_total))

    pool.remove(best)
    return best


def run_placement(
    *,
    vms: list[VM],
    state: ClusterState,
    config: PlanConfig,
    catalog: CatalogConfig | None = None,
    inventory_pool: list[Node] | None = None,
    strategy: PlacementStrategy = PlacementStrategy.SPREAD,
) -> PlacementResult:
    """Execute the full placement simulation.

    Algorithm (HLD §6.2)::

        Sort VMs by memory desc
        For each VM:
            0. Monster VM check
            1. FILTER  — get candidate nodes
            2. EXPAND  — pull inventory from pool OR create catalog node
            3. SCORE   — weighted score + Lookahead k=2
            4. BIND    — place VM on best node

    Parameters
    ----------
    vms : list[VM]
        Normalized VMs (post-overcommit). Will be sorted internally.
    state : ClusterState
        Pre-populated with inventory nodes (spread mode) or empty
        (consolidate mode).
    config : PlanConfig
        Global configuration (weights, limits, overheads, safety).
    catalog : CatalogConfig | None
        Available catalog profiles for expansion.  ``None`` means
        **inventory-only mode** — VMs that don't fit on existing
        nodes are added to the unplaced list without expansion.
    inventory_pool : list[Node] | None
        Reserved inventory nodes for **consolidate mode** (HLD §1.1).
        When no active node fits a VM, the engine pulls a node from
        this pool (free, cost=0) before attempting catalog expansion.
        ``None`` or empty list means no pool (spread mode).
    strategy : PlacementStrategy
        Scoring strategy.  ``SPREAD`` uses LeastAllocated (spread VMs
        evenly); ``CONSOLIDATE`` uses MostAllocated (pack VMs tightly).

    Returns
    -------
    PlacementResult
        Contains the final ``ClusterState``, unplaced VMs, and any
        unused inventory nodes remaining in the pool.
    """
    weights = config.algorithm_weights
    sorted_vms = sorted(vms, key=lambda v: v.memory_mb, reverse=True)
    unplaced: list[VM] = []
    catalog_counter = 0
    pool: list[Node] = list(inventory_pool) if inventory_pool else []

    for idx, vm in enumerate(sorted_vms):
        # ── 1. FILTER ────────────────────────────────────────────
        candidates = state.get_candidate_nodes(vm)

        # ── 2. EXPAND ────────────────────────────────────────────
        if not candidates:
            # 2a. Try inventory pool first (free nodes, consolidate mode)
            remaining = sorted_vms[idx:]
            rem_cpu = sum(v.cpu for v in remaining)
            rem_mem = sum(v.memory_mb for v in remaining)
            pool_node = _pull_from_pool(
                vm,
                pool,
                remaining_cpu=rem_cpu,
                remaining_mem=rem_mem,
            )
            if pool_node is not None:
                state.add_node(pool_node)
                candidates = [pool_node]
            else:
                # 2b. Try catalog expansion
                catalog_counter += 1
                new_node = expand(vm, catalog, config, catalog_counter)
                if new_node is None:
                    # Monster VM — no catalog profile large enough
                    unplaced.append(vm)
                    catalog_counter -= 1  # rollback counter
                    continue
                state.add_node(new_node)
                candidates = [new_node]

        # ── 3. SCORE + LOOKAHEAD ─────────────────────────────────
        next_vm = sorted_vms[idx + 1] if (idx + 1) < len(sorted_vms) else None
        best_node = _score_candidates(
            candidates=candidates,
            vm=vm,
            next_vm=next_vm,
            state=state,
            weights=weights,
            strategy=strategy,
        )

        # ── 4. BIND ─────────────────────────────────────────────
        state.place(vm, best_node)

    return PlacementResult(
        state=state,
        unplaced=unplaced,
        unused_inventory=pool,
    )


def _score_candidates(
    *,
    candidates: list[Node],
    vm: VM,
    next_vm: VM | None,
    state: ClusterState,
    weights: AlgorithmWeights,
    strategy: PlacementStrategy = PlacementStrategy.SPREAD,
) -> Node:
    """Score candidates on **projected** state and return the best node.

    Unlike pre-placement scoring, projected scoring evaluates the node
    *after* tentatively placing the current VM.  This mirrors the K8s
    scheduler approach and allows stranded nodes to "attract" VMs that
    reduce their dimensional imbalance.

    The ``strategy`` parameter controls whether the allocation component
    uses LeastAllocated (spread) or MostAllocated (consolidate).

    For each candidate::

        place(vm, node)
        base_score = score_node(node, strategy=strategy)

        if next_vm exists:
            if node fits next_vm:
                place(next_vm, node)
                lookahead_score = score_node(node, strategy=strategy)
                unplace(next_vm, node)
            else:
                lookahead_score = _LOOKAHEAD_PENALTY

        unplace(vm, node)
        total = base_score + 0.5 × lookahead_score
    """
    best_node = candidates[0]
    best_total = float("-inf")

    for node in candidates:
        # ── Projected base score (with VM placed) ──────────────
        state.place(vm, node)
        base = score_node(node, weights, strategy=strategy)

        # ── Lookahead (with both VM *and* next VM placed) ──────
        lookahead = 0.0
        if next_vm is not None:
            if node.fits(next_vm):
                state.place(next_vm, node)
                lookahead = score_node(node, weights, strategy=strategy)
                state.unplace(next_vm, node)
            else:
                lookahead = _LOOKAHEAD_PENALTY

        # ── Rollback current VM ────────────────────────────────
        state.unplace(vm, node)

        total = base + _LOOKAHEAD_WEIGHT * lookahead
        if total > best_total:
            best_total = total
            best_node = node

    return best_node
