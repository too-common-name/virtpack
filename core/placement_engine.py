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


# ═══════════════════════════════════════════════════════════════════════
# Engine
# ═══════════════════════════════════════════════════════════════════════

# Lookahead penalty applied when the *next* VM cannot fit on the node
# after the current VM is tentatively placed.  Must be large enough to
# dominate the score range (~[-1, 1]) so the node is strongly disfavoured.
_LOOKAHEAD_PENALTY: float = -10.0

# Weight applied to the Lookahead component of the total score.
_LOOKAHEAD_WEIGHT: float = 0.5


def run_placement(
    *,
    vms: list[VM],
    state: ClusterState,
    config: PlanConfig,
    catalog: CatalogConfig,
) -> PlacementResult:
    """Execute the full placement simulation.

    Algorithm (HLD §6.2)::

        Sort VMs by memory desc
        For each VM:
            0. Monster VM check
            1. FILTER  — get candidate nodes
            2. EXPAND  — create catalog node if no candidates
            3. SCORE   — weighted score + Lookahead k=2
            4. BIND    — place VM on best node

    Parameters
    ----------
    vms : list[VM]
        Normalized VMs (post-overcommit). Will be sorted internally.
    state : ClusterState
        Pre-populated with inventory nodes (and any prior catalog nodes).
    config : PlanConfig
        Global configuration (weights, limits, overheads, safety).
    catalog : CatalogConfig
        Available catalog profiles for expansion.

    Returns
    -------
    PlacementResult
        Contains the final ``ClusterState`` and the list of unplaced VMs.
    """
    weights = config.algorithm_weights
    sorted_vms = sorted(vms, key=lambda v: v.memory_mb, reverse=True)
    unplaced: list[VM] = []
    catalog_counter = 0

    for idx, vm in enumerate(sorted_vms):
        # ── 0. HARD CONSTRAINT (Monster VM check) ────────────────
        # Already handled by expand returning None below.
        # If no catalog profile can fit the VM, it's unplaced.

        # ── 1. FILTER ────────────────────────────────────────────
        candidates = state.get_candidate_nodes(vm)

        # ── 2. EXPAND ────────────────────────────────────────────
        if not candidates:
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
        )

        # ── 4. BIND ─────────────────────────────────────────────
        state.place(vm, best_node)

    return PlacementResult(state=state, unplaced=unplaced)


def _score_candidates(
    *,
    candidates: list[Node],
    vm: VM,
    next_vm: VM | None,
    state: ClusterState,
    weights: AlgorithmWeights,
) -> Node:
    """Score candidates with optional Lookahead and return the best node.

    For each candidate:
        base_score = score_node(node)

        if next_vm exists:
            simulate place(vm, node)
            if node fits next_vm → lookahead_score = score_node(node)
            else                 → lookahead_score = _LOOKAHEAD_PENALTY
            undo place

        total = base_score + 0.5 × lookahead_score
    """
    best_node = candidates[0]
    best_total = float("-inf")

    for node in candidates:
        base = score_node(node, weights)

        lookahead = 0.0
        if next_vm is not None:
            # Tentatively place current VM
            state.place(vm, node)
            lookahead = score_node(node, weights) if node.fits(next_vm) else _LOOKAHEAD_PENALTY
            # Rollback
            state.unplace(vm, node)

        total = base + _LOOKAHEAD_WEIGHT * lookahead
        if total > best_total:
            best_total = total
            best_node = node

    return best_node
