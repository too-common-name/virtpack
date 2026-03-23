"""Tests for algorithms.expander — catalog node expansion."""

from __future__ import annotations

import pytest

from algorithms.expander import _profile_fits_vm, expand, select_profile
from models.config import (
    CatalogConfig,
    CatalogProfile,
    CpuTopology,
    PlanConfig,
)
from models.vm import VM

# ── Helpers ──────────────────────────────────────────────────────────────


def _make_topology(sockets: int = 2, cores: int = 32, threads: int = 2) -> CpuTopology:
    return CpuTopology(sockets=sockets, cores_per_socket=cores, threads_per_core=threads)


def _make_profile(
    name: str = "standard",
    sockets: int = 2,
    cores: int = 32,
    threads: int = 2,
    ram_gb: int = 512,
    cost: float = 1.0,
) -> CatalogProfile:
    return CatalogProfile(
        profile_name=name,
        cpu_topology=_make_topology(sockets, cores, threads),
        ram_gb=ram_gb,
        cost_weight=cost,
    )


def _default_config() -> PlanConfig:
    return PlanConfig()


def _small_vm() -> VM:
    """A VM that easily fits on any standard profile."""
    return VM(name="small-vm", cpu=2.0, memory_mb=4096.0)


def _large_vm() -> VM:
    """A VM large enough to test boundary conditions."""
    return VM(name="large-vm", cpu=80.0, memory_mb=400_000.0)


def _monster_vm() -> VM:
    """A VM that won't fit on any reasonable profile."""
    return VM(name="monster-vm", cpu=500.0, memory_mb=10_000_000.0)


# ═══════════════════════════════════════════════════════════════════════
# _profile_fits_vm
# ═══════════════════════════════════════════════════════════════════════


class TestProfileFitsVm:
    """Test the internal fit check against an empty node."""

    def test_small_vm_fits_standard_profile(self) -> None:
        assert _profile_fits_vm(_make_profile(), _small_vm(), _default_config())

    def test_monster_vm_does_not_fit(self) -> None:
        assert not _profile_fits_vm(_make_profile(), _monster_vm(), _default_config())

    def test_respects_normalization(self) -> None:
        """A profile with tiny RAM should fail for a moderate VM."""
        tiny = _make_profile(ram_gb=8)
        vm = VM(name="mid-vm", cpu=1.0, memory_mb=10_000.0)
        assert not _profile_fits_vm(tiny, vm, _default_config())


# ═══════════════════════════════════════════════════════════════════════
# select_profile
# ═══════════════════════════════════════════════════════════════════════


class TestSelectProfile:
    """Test cheapest-eligible profile selection."""

    def test_picks_cheapest_eligible(self) -> None:
        cheap = _make_profile(name="cheap", cost=0.5)
        expensive = _make_profile(name="expensive", cost=2.0)
        catalog = CatalogConfig(profiles=[expensive, cheap])

        result = select_profile(_small_vm(), catalog, _default_config())
        assert result is not None
        assert result.profile_name == "cheap"

    def test_returns_none_for_monster(self) -> None:
        catalog = CatalogConfig(profiles=[_make_profile()])
        result = select_profile(_monster_vm(), catalog, _default_config())
        assert result is None

    def test_skips_too_small_profiles(self) -> None:
        """Only the large-enough profile should be selected."""
        small = _make_profile(name="small", sockets=1, cores=4, threads=1, ram_gb=16, cost=0.1)
        large = _make_profile(name="large", cost=1.0)
        catalog = CatalogConfig(profiles=[small, large])

        vm = VM(name="needs-big", cpu=20.0, memory_mb=200_000.0)
        result = select_profile(vm, catalog, _default_config())
        assert result is not None
        assert result.profile_name == "large"

    def test_single_profile_catalog(self) -> None:
        catalog = CatalogConfig(profiles=[_make_profile(name="only-one")])
        result = select_profile(_small_vm(), catalog, _default_config())
        assert result is not None
        assert result.profile_name == "only-one"


# ═══════════════════════════════════════════════════════════════════════
# expand (full pipeline)
# ═══════════════════════════════════════════════════════════════════════


class TestExpand:
    """Test the complete expand step: select → build → return Node."""

    def test_creates_catalog_node(self) -> None:
        catalog = CatalogConfig(profiles=[_make_profile(name="r760-new", cost=1.5)])
        node = expand(_small_vm(), catalog, _default_config(), next_index=1)
        assert node is not None
        assert node.id == "r760-new-01"
        assert node.is_inventory is False
        assert node.cost_weight == 1.5
        assert node.cpu_total > 0
        assert node.memory_total > 0

    def test_returns_none_for_monster(self) -> None:
        catalog = CatalogConfig(profiles=[_make_profile()])
        node = expand(_monster_vm(), catalog, _default_config(), next_index=1)
        assert node is None

    def test_index_appears_in_id(self) -> None:
        catalog = CatalogConfig(profiles=[_make_profile(name="gen3")])
        node = expand(_small_vm(), catalog, _default_config(), next_index=5)
        assert node is not None
        assert node.id == "gen3-05"

    def test_node_is_normalized(self) -> None:
        """The created node should have overhead-subtracted capacity,
        not raw capacity."""
        profile = _make_profile(name="test-norm", ram_gb=512)
        catalog = CatalogConfig(profiles=[profile])
        node = expand(_small_vm(), catalog, _default_config(), next_index=1)
        assert node is not None
        # Raw logical CPUs = 2×32×2 = 128
        # Effective (post-HT, overheads, safety) should be less
        assert node.cpu_total < 128.0
        # Raw memory = 512 GB = 524_288 MB
        # Post-overhead should be less
        assert node.memory_total < 524_288.0

    def test_picks_cheapest_when_multiple_fit(self) -> None:
        cheap = _make_profile(name="cheap", cost=0.8)
        pricey = _make_profile(name="pricey", cost=3.0)
        catalog = CatalogConfig(profiles=[pricey, cheap])
        node = expand(_small_vm(), catalog, _default_config(), next_index=1)
        assert node is not None
        assert node.profile == "cheap"
        assert node.cost_weight == pytest.approx(0.8)
