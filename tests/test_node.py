"""Unit tests for models.node.Node."""

import pytest
from pydantic import ValidationError

from models.node import Node
from models.vm import VM


# ── Helpers ────────────────────────────────────────────────────────────

def _make_node(
    cpu_total: float = 64.0,
    memory_total: float = 524288.0,  # 512 GB in MB
    pods_total: int = 500,
    cpu_used: float = 0.0,
    memory_used: float = 0.0,
    pods_used: int = 0,
) -> Node:
    """Shortcut: create an inventory node with customizable usage."""
    return Node(
        id="test-node-01",
        profile="r740-test",
        cpu_total=cpu_total,
        memory_total=memory_total,
        pods_total=pods_total,
        cpu_used=cpu_used,
        memory_used=memory_used,
        pods_used=pods_used,
        cost_weight=0.0,
        is_inventory=True,
    )


def _make_vm(
    cpu: float = 1.0,
    memory_mb: float = 4096.0,
    pods: int = 1,
) -> VM:
    return VM(name="vm-test", cpu=cpu, memory_mb=memory_mb, pods=pods)


# ── fits() ─────────────────────────────────────────────────────────────


class TestNodeFits:
    """Hard-constraint filter: fits() across all 3 dimensions."""

    def test_fits_success(self) -> None:
        node = _make_node()
        vm = _make_vm(cpu=2.0, memory_mb=8192.0)
        assert node.fits(vm) is True

    def test_fits_exact_boundary(self) -> None:
        """VM that fills the node exactly should still fit."""
        node = _make_node(cpu_total=4.0, memory_total=8192.0, pods_total=1)
        vm = _make_vm(cpu=4.0, memory_mb=8192.0, pods=1)
        assert node.fits(vm) is True

    def test_fails_cpu_exceeded(self) -> None:
        node = _make_node(cpu_total=4.0, cpu_used=3.5)
        vm = _make_vm(cpu=1.0)  # 3.5 + 1.0 = 4.5 > 4.0
        assert node.fits(vm) is False

    def test_fails_memory_exceeded(self) -> None:
        node = _make_node(memory_total=8192.0, memory_used=5000.0)
        vm = _make_vm(memory_mb=4000.0)  # 5000 + 4000 = 9000 > 8192
        assert node.fits(vm) is False

    def test_fails_pods_exceeded(self) -> None:
        node = _make_node(pods_total=10, pods_used=10)
        vm = _make_vm()  # pods=1, 10 + 1 = 11 > 10
        assert node.fits(vm) is False

    def test_fails_only_cpu_while_others_fit(self) -> None:
        """A single dimension failure is enough to reject."""
        node = _make_node(
            cpu_total=2.0, cpu_used=1.5,
            memory_total=524288.0, memory_used=0.0,
            pods_total=500, pods_used=0,
        )
        vm = _make_vm(cpu=1.0, memory_mb=1024.0)  # CPU: 1.5+1.0 > 2.0
        assert node.fits(vm) is False


# ── Derived properties ─────────────────────────────────────────────────


class TestNodeDerivedProperties:
    """Scorer-facing properties computed from usage/total."""

    def test_utilization_empty_node(self) -> None:
        node = _make_node()
        assert node.cpu_util == pytest.approx(0.0)
        assert node.memory_util == pytest.approx(0.0)

    def test_utilization_half_loaded(self) -> None:
        node = _make_node(
            cpu_total=64.0, cpu_used=32.0,
            memory_total=524288.0, memory_used=262144.0,
        )
        assert node.cpu_util == pytest.approx(0.5)
        assert node.memory_util == pytest.approx(0.5)

    def test_remaining_capacity(self) -> None:
        node = _make_node(
            cpu_total=64.0, cpu_used=10.0,
            memory_total=524288.0, memory_used=100000.0,
            pods_total=500, pods_used=42,
        )
        assert node.cpu_remaining == pytest.approx(54.0)
        assert node.memory_remaining == pytest.approx(424288.0)
        assert node.pods_remaining == 458

    def test_utilization_fully_loaded(self) -> None:
        node = _make_node(
            cpu_total=64.0, cpu_used=64.0,
            memory_total=524288.0, memory_used=524288.0,
            pods_total=500, pods_used=500,
        )
        assert node.cpu_util == pytest.approx(1.0)
        assert node.memory_util == pytest.approx(1.0)
        assert node.pods_remaining == 0


# ── Factory methods ────────────────────────────────────────────────────


class TestNodeFactories:
    """new_inventory() and new_catalog() enforce metadata invariants."""

    def test_new_inventory_sets_cost_zero(self) -> None:
        node = Node.new_inventory(
            id="inv-01", profile="r740",
            cpu_total=48.0, memory_total=400000.0, pods_total=500,
        )
        assert node.cost_weight == 0.0
        assert node.is_inventory is True

    def test_new_catalog_sets_is_inventory_false(self) -> None:
        node = Node.new_catalog(
            id="cat-01", profile="r760",
            cpu_total=60.0, memory_total=800000.0, pods_total=500,
            cost_weight=1.0,
        )
        assert node.is_inventory is False
        assert node.cost_weight == 1.0

    def test_new_catalog_rejects_zero_cost(self) -> None:
        with pytest.raises(ValueError, match="cost_weight > 0"):
            Node.new_catalog(
                id="bad", profile="bad",
                cpu_total=1.0, memory_total=1.0, pods_total=1,
                cost_weight=0.0,
            )

    def test_new_catalog_rejects_negative_cost(self) -> None:
        with pytest.raises(ValueError, match="cost_weight > 0"):
            Node.new_catalog(
                id="bad", profile="bad",
                cpu_total=1.0, memory_total=1.0, pods_total=1,
                cost_weight=-1.0,
            )


# ── Mutability (place / unplace support) ──────────────────────────────


class TestNodeMutability:
    """Node must be mutable (not frozen) for O(1) place/unplace."""

    def test_can_mutate_cpu_used(self) -> None:
        node = _make_node()
        node.cpu_used = 10.0
        assert node.cpu_used == 10.0

    def test_can_mutate_memory_used(self) -> None:
        node = _make_node()
        node.memory_used = 50000.0
        assert node.memory_used == 50000.0

    def test_can_mutate_pods_used(self) -> None:
        node = _make_node()
        node.pods_used = 42
        assert node.pods_used == 42

    def test_simulate_place_unplace_roundtrip(self) -> None:
        """Simulates the Lookahead rollback pattern from HLD §6.2."""
        node = _make_node(cpu_total=64.0, memory_total=524288.0, pods_total=500)
        vm = _make_vm(cpu=4.0, memory_mb=16384.0)

        # Snapshot
        orig_cpu = node.cpu_used
        orig_mem = node.memory_used
        orig_pods = node.pods_used

        # Simulate place
        node.cpu_used += vm.cpu
        node.memory_used += vm.memory_mb
        node.pods_used += vm.pods

        assert node.cpu_used == pytest.approx(4.0)

        # Undo (unplace)
        node.cpu_used -= vm.cpu
        node.memory_used -= vm.memory_mb
        node.pods_used -= vm.pods

        assert node.cpu_used == pytest.approx(orig_cpu)
        assert node.memory_used == pytest.approx(orig_mem)
        assert node.pods_used == orig_pods


# ── Validation ─────────────────────────────────────────────────────────


class TestNodeValidation:
    """Pydantic field constraints on construction."""

    def test_rejects_empty_id(self) -> None:
        with pytest.raises(ValidationError):
            Node(
                id="", profile="p", cpu_total=1.0, memory_total=1.0,
                pods_total=1, cost_weight=0.0, is_inventory=True,
            )

    def test_rejects_zero_cpu_total(self) -> None:
        with pytest.raises(ValidationError):
            Node(
                id="n", profile="p", cpu_total=0.0, memory_total=1.0,
                pods_total=1, cost_weight=0.0, is_inventory=True,
            )

    def test_rejects_negative_memory_total(self) -> None:
        with pytest.raises(ValidationError):
            Node(
                id="n", profile="p", cpu_total=1.0, memory_total=-1.0,
                pods_total=1, cost_weight=0.0, is_inventory=True,
            )
