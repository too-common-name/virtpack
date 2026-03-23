"""Tests for algorithms.scorer — K8s-like node scoring (HLD §6.1)."""

from __future__ import annotations

import pytest

from algorithms.scorer import (
    balance_score,
    fragmentation_penalty,
    pod_headroom_score,
    score_node,
    spread_score,
)
from models.config import AlgorithmWeights
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
# fragmentation_penalty
# ═══════════════════════════════════════════════════════════════════════


class TestFragmentationPenalty:
    """frag = (memory_remaining / memory_total)²."""

    def test_empty_node(self) -> None:
        """Empty → remaining=total → penalty = 1.0."""
        n = _node()
        assert fragmentation_penalty(n) == pytest.approx(1.0)

    def test_full_node(self) -> None:
        """Full → remaining=0 → penalty = 0.0."""
        n = _node(memory_used=100_000.0)
        assert fragmentation_penalty(n) == pytest.approx(0.0)

    def test_half_used(self) -> None:
        """50% used → remaining=50% → penalty = 0.25."""
        n = _node(memory_used=50_000.0)
        assert fragmentation_penalty(n) == pytest.approx(0.25)

    def test_90_percent_used(self) -> None:
        """90% used → 10% remaining → penalty = 0.01."""
        n = _node(memory_used=90_000.0)
        assert fragmentation_penalty(n) == pytest.approx(0.01)


# ═══════════════════════════════════════════════════════════════════════
# score_node (weighted combination)
# ═══════════════════════════════════════════════════════════════════════


class TestScoreNode:
    """Weighted combination: α·balance + β·spread + γ·pod_headroom − δ·frag."""

    def test_with_default_weights(self) -> None:
        """Smoke test with default AlgorithmWeights."""
        weights = AlgorithmWeights()
        n = _node(cpu_used=50.0, memory_used=50_000.0, pods_used=50)
        score = score_node(n, weights)
        # balance=1.0, spread=0.5, pod_headroom=0.8, frag=0.25
        expected = (
            0.3 * 1.0  # balance
            + 0.3 * 0.5  # spread
            + 0.1 * 0.8  # pod_headroom
            - 0.3 * 0.25  # fragmentation
        )
        assert score == pytest.approx(expected)

    def test_uniform_weights(self) -> None:
        """All weights equal (0.25 each)."""
        weights = AlgorithmWeights(
            alpha_balance=0.25,
            beta_spread=0.25,
            gamma_pod_headroom=0.25,
            delta_frag_penalty=0.25,
        )
        n = _node()  # empty node
        # balance=1.0, spread=1.0, pod_headroom=1.0, frag=1.0
        expected = 0.25 * 1.0 + 0.25 * 1.0 + 0.25 * 1.0 - 0.25 * 1.0
        assert score_node(n, weights) == pytest.approx(expected)

    def test_pure_balance_weight(self) -> None:
        """Only balance weight active — verify isolation."""
        weights = AlgorithmWeights(
            alpha_balance=1.0,
            beta_spread=0.0,
            gamma_pod_headroom=0.0,
            delta_frag_penalty=0.0,
        )
        n = _node(cpu_used=50.0, memory_used=50_000.0)
        assert score_node(n, weights) == pytest.approx(1.0)

    def test_fuller_node_scores_lower_spread(self) -> None:
        """A fuller node scores lower on spread than a less-used one."""
        weights = AlgorithmWeights(
            alpha_balance=0.0,
            beta_spread=1.0,
            gamma_pod_headroom=0.0,
            delta_frag_penalty=0.0,
        )
        n_light = _node(cpu_used=20.0, memory_used=20_000.0)
        n_heavy = _node(cpu_used=80.0, memory_used=80_000.0)
        assert score_node(n_light, weights) > score_node(n_heavy, weights)

    def test_score_is_deterministic(self) -> None:
        """Same inputs always produce the same score."""
        weights = AlgorithmWeights()
        n = _node(cpu_used=33.0, memory_used=44_000.0, pods_used=100)
        scores = [score_node(n, weights) for _ in range(10)]
        assert all(s == scores[0] for s in scores)
