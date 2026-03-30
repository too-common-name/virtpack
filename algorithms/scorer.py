"""Weighted node scoring function for capacity planning (HLD §6.1).

All component scores are ∈ [0, 1].  The final weighted score is:

    score(node) = α·balance + β·alloc + γ·pod_headroom − δ·stranded_penalty

where *alloc* is strategy-dependent:

* **spread mode** → ``spread_score`` (LeastAllocated, favors empty nodes)
* **consolidate mode** → ``pack_score``  (MostAllocated, favors full nodes)

Each term provides a unique, non-redundant signal:

* **balance** — CPU/memory utilization proportionality
* **spread / pack** — allocation preference (mode-dependent)
* **pod_headroom** — pod slot availability
* **stranded_penalty** — dimensional imbalance of *remaining* capacity

Pure functions — no side effects, no hidden state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from models.config import PlacementStrategy

if TYPE_CHECKING:
    from models.config import AlgorithmWeights
    from models.node import Node


# ═══════════════════════════════════════════════════════════════════════
# Component Scores  (each ∈ [0, 1])
# ═══════════════════════════════════════════════════════════════════════


def balance_score(node: Node) -> float:
    """CPU/Memory balance — ``1 - abs(cpu_util - mem_util)``.

    Encourages balanced nodes to prevent exhausting one resource
    while the other sits idle (mimics ``NodeResourcesBalancedAllocation``).
    """
    return 1.0 - abs(node.cpu_util - node.memory_util)


def spread_score(node: Node) -> float:
    """Spread / LeastAllocated — ``((1 - cpu_util) + (1 - mem_util)) / 2``.

    Favors nodes with the most free resources, mimicking K8s
    ``LeastAllocated`` scheduling strategy.  Used in **spread** mode.
    """
    return ((1.0 - node.cpu_util) + (1.0 - node.memory_util)) / 2.0


def pack_score(node: Node) -> float:
    """Pack / MostAllocated — ``(cpu_util + mem_util) / 2``.

    Favors nodes with the least free resources, encouraging tight
    bin-packing.  Used in **consolidate** mode to preserve large
    contiguous free blocks on other nodes for future big VMs.

    Mathematically: ``pack_score = 1 - spread_score``.
    """
    return (node.cpu_util + node.memory_util) / 2.0


def pod_headroom_score(node: Node) -> float:
    """Pod headroom — ``1 - (pods_used / pods_total)``.

    Discourages pod exhaustion to preserve IP space and
    scheduling limits.
    """
    if node.pods_total == 0:
        return 0.0
    return 1.0 - (node.pods_used / node.pods_total)


def fragmentation_penalty(node: Node) -> float:
    """Stranded capacity penalty — ``(cpu_remaining% − memory_remaining%)²``.

    Penalizes nodes where remaining CPU and memory are disproportionate.
    When one dimension has ample remaining capacity but the other is
    nearly exhausted, the excess dimension is **stranded** — it cannot
    be consumed by future VMs.

    This is the genuine fragmentation signal for offline capacity
    planning: unlike online scheduling where pods arrive randomly,
    all items are known upfront, so the penalty focuses on
    *dimensional imbalance* of remaining capacity.

    ======== ========= ========= ========= =========================
    Scenario CPU rem%  Mem rem%  Penalty   Interpretation
    ======== ========= ========= ========= =========================
    Empty    100%      100%      0.00      No stranding
    Balanced 50%       50%       0.00      No stranding
    CPU-bound 10%      70%       0.36      Memory stranded
    Mem-bound 60%       5%       0.30      CPU stranded
    Full      0%        0%       0.00      Nothing remaining
    ======== ========= ========= ========= =========================

    Note: this is a *penalty* (subtracted in the weighted score).
    """
    if node.cpu_total == 0.0 or node.memory_total == 0.0:
        return 0.0
    cpu_rem = node.cpu_remaining / node.cpu_total
    mem_rem = node.memory_remaining / node.memory_total
    diff = cpu_rem - mem_rem
    return diff * diff


# ═══════════════════════════════════════════════════════════════════════
# Weighted Score
# ═══════════════════════════════════════════════════════════════════════


def score_node(
    node: Node,
    weights: AlgorithmWeights,
    *,
    strategy: PlacementStrategy = PlacementStrategy.SPREAD,
) -> float:
    """Compute the weighted score for a node.

    ::

        score = α·balance + β·alloc + γ·pod_headroom − δ·stranded_penalty

    where *alloc* is ``spread_score`` (LeastAllocated) in spread mode
    or ``pack_score`` (MostAllocated) in consolidate mode.

    Higher is better.  Range depends on weights but typically ∈ [-1, 1].
    """
    alloc = pack_score(node) if strategy == PlacementStrategy.CONSOLIDATE else spread_score(node)
    return (
        weights.alpha_balance * balance_score(node)
        + weights.beta_spread * alloc
        + weights.gamma_pod_headroom * pod_headroom_score(node)
        - weights.delta_frag_penalty * fragmentation_penalty(node)
    )
