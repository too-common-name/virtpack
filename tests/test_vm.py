"""Unit tests for models.vm.VM."""

import pytest
from pydantic import ValidationError

from models.vm import VM


# ── Construction ────────────────────────────────────────────────────────


class TestVMConstruction:
    """Valid construction and default values."""

    def test_valid_construction(self) -> None:
        vm = VM(name="db01", cpu=2.0, memory_mb=32768.0)

        assert vm.name == "db01"
        assert vm.cpu == 2.0
        assert vm.memory_mb == 32768.0

    def test_default_pods_is_one(self) -> None:
        vm = VM(name="web01", cpu=0.5, memory_mb=1024.0)
        assert vm.pods == 1

    def test_explicit_pods(self) -> None:
        vm = VM(name="web01", cpu=0.5, memory_mb=1024.0, pods=3)
        assert vm.pods == 3


# ── Frozen (immutability) ──────────────────────────────────────────────


class TestVMFrozen:
    """VM must be immutable after creation."""

    def test_cannot_mutate_cpu(self) -> None:
        vm = VM(name="db01", cpu=2.0, memory_mb=4096.0)
        with pytest.raises(ValidationError):
            vm.cpu = 4.0  # type: ignore[misc]

    def test_cannot_mutate_name(self) -> None:
        vm = VM(name="db01", cpu=2.0, memory_mb=4096.0)
        with pytest.raises(ValidationError):
            vm.name = "db02"  # type: ignore[misc]

    def test_cannot_mutate_memory(self) -> None:
        vm = VM(name="db01", cpu=2.0, memory_mb=4096.0)
        with pytest.raises(ValidationError):
            vm.memory_mb = 8192.0  # type: ignore[misc]


# ── Strict typing ─────────────────────────────────────────────────────


class TestVMStrictTyping:
    """strict=True must reject type coercion."""

    def test_rejects_string_for_cpu(self) -> None:
        with pytest.raises(ValidationError):
            VM(name="bad", cpu="four", memory_mb=1024.0)  # type: ignore[arg-type]

    def test_allows_int_for_float_fields(self) -> None:
        """Pydantic V2 strict mode permits int → float (int is a subtype)."""
        vm = VM(name="ok", cpu=4, memory_mb=1024)  # type: ignore[arg-type]
        assert vm.cpu == 4.0
        assert isinstance(vm.cpu, (int, float))

    def test_rejects_float_for_pods(self) -> None:
        with pytest.raises(ValidationError):
            VM(name="bad", cpu=1.0, memory_mb=1024.0, pods=1.5)  # type: ignore[arg-type]

    def test_rejects_string_for_memory(self) -> None:
        with pytest.raises(ValidationError):
            VM(name="bad", cpu=1.0, memory_mb="lots")  # type: ignore[arg-type]


# ── Field validation boundaries ────────────────────────────────────────


class TestVMValidation:
    """Boundary checks on field constraints."""

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(ValidationError):
            VM(name="", cpu=1.0, memory_mb=1024.0)

    def test_rejects_zero_cpu(self) -> None:
        with pytest.raises(ValidationError):
            VM(name="bad", cpu=0.0, memory_mb=1024.0)

    def test_rejects_negative_cpu(self) -> None:
        with pytest.raises(ValidationError):
            VM(name="bad", cpu=-1.0, memory_mb=1024.0)

    def test_rejects_zero_memory(self) -> None:
        with pytest.raises(ValidationError):
            VM(name="bad", cpu=1.0, memory_mb=0.0)

    def test_rejects_negative_memory(self) -> None:
        with pytest.raises(ValidationError):
            VM(name="bad", cpu=1.0, memory_mb=-512.0)

    def test_rejects_zero_pods(self) -> None:
        with pytest.raises(ValidationError):
            VM(name="bad", cpu=1.0, memory_mb=1024.0, pods=0)
