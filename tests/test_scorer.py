"""Tests for algorithms.scorer — weighted node scoring (HLD §6.1)."""

from __future__ import annotations

import pytest

from algorithms.scorer import (
    balance_score,
    fragmentation_penalty,
    pack_score,
    pod_headroom_score,
    score_node,
    spread_score,
)
from models.config import AlgorithmWeights, PlacementStrategy
from models.node import Node

# ── Helpers ──────────────────────────────────────────────────────────────


def _node(
    cpu_total: float = 100.0,
    memory_total: float = 100_000.0,
    pods_total: int = 250,
    cpu_used: float = 0.0,
    memory_used: float = 0.0,
    pods_used: int = 0,
) -> Node:
    return Node.new_inventory(
        profile="test",
        index=1,
        cpu_total=cpu_total,
        memory_total=memory_total,
        pods_total=pods_total,
    ).model_copy(update={"cpu_used": cpu_used, "memory_used": memory_used, "pods_used": pods_used})


# ═══════════════════════════════════════════════════════════════════════
# balance_score
# ═══════════════════════════════════════════════════════════════════════


class TestBalanceScore:
    """balance = 1 - abs(cpu_util - mem_util)."""

    def test_perfect_balance(self) -> None:
        """Same CPU and memory utilization → score 1.0."""
        n = _node(cpu_used=50.0, memory_used=50_000.0)  # both 50%
        assert balance_score(n) == pytest.approx(1.0)

    def test_full_imbalance(self) -> None:
        """CPU at 100%, memory at 0% → score 0.0."""
        n = _node(cpu_used=100.0, memory_used=0.0)
        assert balance_score(n) == pytest.approx(0.0)

    def test_empty_node(self) -> None:
        """Empty node → both utils at 0 → score 1.0."""
        n = _node()
        assert balance_score(n) == pytest.approx(1.0)

    def test_partial_imbalance(self) -> None:
        """CPU 80%, memory 60% → score = 1 - 0.2 = 0.8."""
        n = _node(cpu_used=80.0, memory_used=60_000.0)
        assert balance_score(n) == pytest.approx(0.8)


# ═══════════════════════════════════════════════════════════════════════
# spread_score
# ═══════════════════════════════════════════════════════════════════════


class TestSpreadScore:
    """spread = ((1 - cpu_util) + (1 - mem_util)) / 2."""

    def test_empty_node(self) -> None:
        """Empty node → both free → score 1.0."""
        n = _node()
        assert spread_score(n) == pytest.approx(1.0)

    def test_full_node(self) -> None:
        """Fully used → score 0.0."""
        n = _node(cpu_used=100.0, memory_used=100_000.0)
        assert spread_score(n) == pytest.approx(0.0)

    def test_half_used(self) -> None:
        """Half used on both → score 0.5."""
        n = _node(cpu_used=50.0, memory_used=50_000.0)
        assert spread_score(n) == pytest.approx(0.5)


# ═══════════════════════════════════════════════════════════════════════
# pack_score
# ═══════════════════════════════════════════════════════════════════════


class TestPackScore:
    """pack = (cpu_util + mem_util) / 2  (MostAllocated, complement of spread)."""

    def test_empty_node(self) -> None:
        """Empty node → both idle → score 0.0."""
        n = _node()
        assert pack_score(n) == pytest.approx(0.0)

    def test_full_node(self) -> None:
        """Fully used → score 1.0."""
        n = _node(cpu_used=100.0, memory_used=100_000.0)
        assert pack_score(n) == pytest.approx(1.0)

    def test_half_used(self) -> None:
        """Half used on both → score 0.5."""
        n = _node(cpu_used=50.0, memory_used=50_000.0)
        assert pack_score(n) == pytest.approx(0.5)

    def test_complement_of_spread(self) -> None:
        """pack_score == 1 - spread_score for any node."""
        n = _node(cpu_used=30.0, memory_used=70_000.0)
        assert pack_score(n) == pytest.approx(1.0 - spread_score(n))


# ═══════════════════════════════════════════════════════════════════════
# pod_headroom_score
# ═══════════════════════════════════════════════════════════════════════


class TestPodHeadroomScore:
    """pod_headroom = 1 - (pods_used / pods_total)."""

    def test_no_pods(self) -> None:
        n = _node()
        assert pod_headroom_score(n) == pytest.approx(1.0)

    def test_all_pods_used(self) -> None:
        n = _node(pods_used=250)
        assert pod_headroom_score(n) == pytest.approx(0.0)

    def test_half_pods(self) -> None:
        n = _node(pods_used=125)
        assert pod_headroom_score(n) == pytest.approx(0.5)

    def test_zero_pods_total_rejected_by_validator(self) -> None:
        """pods_total=0 is rejected by Pydantic — the zero-guard in
        the scorer is only a safety net."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            _node(pods_total=0)


# ═══════════════════════════════════════════════════════════════════════
# fragmentation_penalty (stranded capacity)
# ═══════════════════════════════════════════════════════════════════════


class TestFragmentationPenalty:
    """Stranded capacity = (cpu_remaining% − memory_remaining%)²."""

    def test_empty_node(self) -> None:
        """Empty → both at 100% remaining → diff=0 → penalty=0."""
        n = _node()
        assert fragmentation_penalty(n) == pytest.approx(0.0)

    def test_full_node(self) -> None:
        """Full → both at 0% remaining → diff=0 → penalty=0."""
        n = _node(cpu_used=100.0, memory_used=100_000.0)
        assert fragmentation_penalty(n) == pytest.approx(0.0)

    def test_balanced_half_used(self) -> None:
        """50% CPU and 50% memory used → remaining (50%, 50%) → penalty=0."""
        n = _node(cpu_used=50.0, memory_used=50_000.0)
        assert fragmentation_penalty(n) == pytest.approx(0.0)

    def test_cpu_bound(self) -> None:
        """CPU 90% used, memory 30% used → remaining (10%, 70%).

        diff = 0.10 − 0.70 = −0.60 → penalty = 0.36
        """
        n = _node(cpu_used=90.0, memory_used=30_000.0)
        assert fragmentation_penalty(n) == pytest.approx(0.36)

    def test_memory_bound(self) -> None:
        """CPU 40% used, memory 95% used → remaining (60%, 5%).

        diff = 0.60 − 0.05 = 0.55 → penalty = 0.3025
        """
        n = _node(cpu_used=40.0, memory_used=95_000.0)
        assert fragmentation_penalty(n) == pytest.approx(0.3025)

    def test_slight_imbalance(self) -> None:
        """CPU 50% used, memory 60% used → remaining (50%, 40%).

        diff = 0.50 − 0.40 = 0.10 → penalty = 0.01
        """
        n = _node(cpu_used=50.0, memory_used=60_000.0)
        assert fragmentation_penalty(n) == pytest.approx(0.01)

    def test_zero_total_guard(self) -> None:
        """Zero-capacity node → 0.0 (safety net)."""
        n = _node(cpu_total=100.0, memory_total=100_000.0)
        # Manually override totals to test the guard
        n_zero = n.model_copy(update={"cpu_total": 0.0})
        assert fragmentation_penalty(n_zero) == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════════
# score_node (weighted combination)
# ═══════════════════════════════════════════════════════════════════════


class TestScoreNode:
    """Weighted combination: α·balance + β·spread + γ·pod_headroom − δ·stranded."""

    def test_with_default_weights(self) -> None:
        """Smoke test with default AlgorithmWeights.

        Node at 50% CPU, 50% memory, 50 pods:
        - balance = 1.0 (perfectly balanced)
        - spread = 0.5
        - pod_headroom = 0.8
        - stranded = (0.5 − 0.5)² = 0.0 (balanced remaining)
        """
        weights = AlgorithmWeights()
        n = _node(cpu_used=50.0, memory_used=50_000.0, pods_used=50)
        score = score_node(n, weights)
        expected = (
            0.3 * 1.0  # balance
            + 0.3 * 0.5  # spread
            + 0.1 * 0.8  # pod_headroom
            - 0.3 * 0.0  # stranded penalty (balanced remaining)
        )
        assert score == pytest.approx(expected)

    def test_uniform_weights(self) -> None:
        """All weights equal (0.25 each) on an empty node."""
        weights = AlgorithmWeights(
            alpha_balance=0.25,
            beta_alloc=0.25,
            gamma_pod_headroom=0.25,
            delta_frag_penalty=0.25,
        )
        n = _node()  # empty node
        # balance=1.0, spread=1.0, pod_headroom=1.0, stranded=0.0
        expected = 0.25 * 1.0 + 0.25 * 1.0 + 0.25 * 1.0 - 0.25 * 0.0
        assert score_node(n, weights) == pytest.approx(expected)

    def test_pure_balance_weight(self) -> None:
        """Only balance weight active — verify isolation."""
        weights = AlgorithmWeights(
            alpha_balance=1.0,
            beta_alloc=0.0,
            gamma_pod_headroom=0.0,
            delta_frag_penalty=0.0,
        )
        n = _node(cpu_used=50.0, memory_used=50_000.0)
        assert score_node(n, weights) == pytest.approx(1.0)

    def test_fuller_node_scores_lower_spread(self) -> None:
        """A fuller node scores lower on spread than a less-used one."""
        weights = AlgorithmWeights(
            alpha_balance=0.0,
            beta_alloc=1.0,
            gamma_pod_headroom=0.0,
            delta_frag_penalty=0.0,
        )
        n_light = _node(cpu_used=20.0, memory_used=20_000.0)
        n_heavy = _node(cpu_used=80.0, memory_used=80_000.0)
        assert score_node(n_light, weights) > score_node(n_heavy, weights)

    def test_stranded_penalty_penalizes_imbalanced_node(self) -> None:
        """A node with lopsided remaining capacity should score lower
        when delta_frag_penalty is active."""
        weights = AlgorithmWeights(
            alpha_balance=0.0,
            beta_alloc=0.0,
            gamma_pod_headroom=0.0,
            delta_frag_penalty=1.0,
        )
        # Balanced remaining: 50% CPU, 50% memory
        balanced = _node(cpu_used=50.0, memory_used=50_000.0)
        # Lopsided remaining: 10% CPU, 70% memory → stranded penalty = 0.36
        lopsided = _node(cpu_used=90.0, memory_used=30_000.0)

        assert score_node(balanced, weights) > score_node(lopsided, weights)

    def test_fuller_node_scores_higher_pack(self) -> None:
        """In consolidate mode, a fuller node scores higher on the alloc component."""
        weights = AlgorithmWeights(
            alpha_balance=0.0,
            beta_alloc=1.0,
            gamma_pod_headroom=0.0,
            delta_frag_penalty=0.0,
        )
        n_light = _node(cpu_used=20.0, memory_used=20_000.0)
        n_heavy = _node(cpu_used=80.0, memory_used=80_000.0)
        assert score_node(n_heavy, weights, strategy=PlacementStrategy.CONSOLIDATE) > score_node(
            n_light, weights, strategy=PlacementStrategy.CONSOLIDATE
        )

    def test_consolidate_inverts_spread_preference(self) -> None:
        """spread favors light node; consolidate favors heavy node."""
        weights = AlgorithmWeights(
            alpha_balance=0.0,
            beta_alloc=1.0,
            gamma_pod_headroom=0.0,
            delta_frag_penalty=0.0,
        )
        n_light = _node(cpu_used=20.0, memory_used=20_000.0)
        n_heavy = _node(cpu_used=80.0, memory_used=80_000.0)

        # Spread: light wins
        assert score_node(n_light, weights, strategy=PlacementStrategy.SPREAD) > score_node(
            n_heavy, weights, strategy=PlacementStrategy.SPREAD
        )
        # Consolidate: heavy wins
        assert score_node(n_heavy, weights, strategy=PlacementStrategy.CONSOLIDATE) > score_node(
            n_light, weights, strategy=PlacementStrategy.CONSOLIDATE
        )

    def test_default_strategy_is_spread(self) -> None:
        """score_node without strategy kwarg uses spread (backward compat)."""
        weights = AlgorithmWeights()
        n = _node(cpu_used=50.0, memory_used=50_000.0, pods_used=50)
        assert score_node(n, weights) == score_node(n, weights, strategy=PlacementStrategy.SPREAD)

    def test_score_is_deterministic(self) -> None:
        """Same inputs always produce the same score."""
        weights = AlgorithmWeights()
        n = _node(cpu_used=33.0, memory_used=44_000.0, pods_used=100)
        scores = [score_node(n, weights) for _ in range(10)]
        assert all(s == scores[0] for s in scores)
