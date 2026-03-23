"""K8s-like node scoring function (HLD §6.1).

All component scores are ∈ [0, 1].  The final weighted score is:

    score(node) = α·balance + β·spread + γ·pod_headroom − δ·frag_penalty

Pure functions — no side effects, no hidden state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    ``LeastAllocated`` scheduling strategy.
    """
    return ((1.0 - node.cpu_util) + (1.0 - node.memory_util)) / 2.0


def pod_headroom_score(node: Node) -> float:
    """Pod headroom — ``1 - (pods_used / pods_total)``.

    Discourages pod exhaustion to preserve IP space and
    scheduling limits.
    """
    if node.pods_total == 0:
        return 0.0
    return 1.0 - (node.pods_used / node.pods_total)


def fragmentation_penalty(node: Node) -> float:
    """Memory fragmentation — ``(memory_remaining / memory_total)²``.

    Penalty increases when nodes contain small remaining memory
    segments that are unlikely to host future VMs.

    Note: this is a *penalty* (subtracted in the weighted score).
    A fully empty node has penalty 1.0 (no fragment yet — but the
    cost of leaving it nearly empty is high).  A full node has
    penalty ~0 (no wasted fragment).
    """
    if node.memory_total == 0.0:
        return 0.0
    ratio = node.memory_remaining / node.memory_total
    return ratio * ratio


# ═══════════════════════════════════════════════════════════════════════
# Weighted Score
# ═══════════════════════════════════════════════════════════════════════


def score_node(node: Node, weights: AlgorithmWeights) -> float:
    """Compute the weighted K8s-like score for a node.

    ::

        score = α·balance + β·spread + γ·pod_headroom − δ·frag_penalty

    Higher is better.  Range depends on weights but typically ∈ [-1, 1].
    """
    return (
        weights.alpha_balance * balance_score(node)
        + weights.beta_spread * spread_score(node)
        + weights.gamma_pod_headroom * pod_headroom_score(node)
        - weights.delta_frag_penalty * fragmentation_penalty(node)
    )
