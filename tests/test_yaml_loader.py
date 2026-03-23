"""Tests for io.yaml_loader — YAML config file loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from loaders.yaml_loader import (
    ConfigLoadError,
    load_catalog_config,
    load_inventory_config,
    load_plan_config,
)

# ── Helpers ──────────────────────────────────────────────────────────────


def _write(tmp_path: Path, name: str, content: str) -> Path:
    """Write *content* to a file under *tmp_path* and return the path."""
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


# ═══════════════════════════════════════════════════════════════════════
# load_plan_config
# ═══════════════════════════════════════════════════════════════════════


class TestLoadPlanConfig:
    """PlanConfig: all sections have defaults, so empty/missing is valid."""

    def test_none_path_returns_defaults(self) -> None:
        cfg = load_plan_config(None)
        assert cfg.overcommit.cpu_ratio == 8.0
        assert cfg.safety_margins.utilization_targets.cpu == 85.0

    def test_empty_file_returns_defaults(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "config.yaml", "")
        cfg = load_plan_config(p)
        assert cfg.cluster_limits.max_pods_per_node == 250

    def test_yaml_null_returns_defaults(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "config.yaml", "---\n")
        cfg = load_plan_config(p)
        assert cfg.algorithm_weights.alpha_balance == pytest.approx(0.3)

    def test_partial_override(self, tmp_path: Path) -> None:
        content = """\
overcommit:
  cpu_ratio: 4.0
"""
        p = _write(tmp_path, "config.yaml", content)
        cfg = load_plan_config(p)
        assert cfg.overcommit.cpu_ratio == 4.0
        # Other sections still have defaults
        assert cfg.cluster_limits.max_pods_per_node == 250

    def test_full_config(self, tmp_path: Path) -> None:
        content = """\
cluster_limits:
  max_pods_per_node: 200
overcommit:
  cpu_ratio: 4.0
  memory_ratio: 1.5
virt_overheads:
  ht_efficiency_factor: 1.8
  ocp_virt_core: 3.0
  ocp_virt_memory_mb: 500
  eviction_hard_mb: 200
safety_margins:
  utilization_targets:
    cpu: 90
    memory: 75
  ha_failures_to_tolerate: 2
algorithm_weights:
  alpha_balance: 0.25
  beta_spread: 0.25
  gamma_pod_headroom: 0.25
  delta_frag_penalty: 0.25
"""
        p = _write(tmp_path, "config.yaml", content)
        cfg = load_plan_config(p)
        assert cfg.cluster_limits.max_pods_per_node == 200
        assert cfg.overcommit.memory_ratio == 1.5
        assert cfg.virt_overheads.ht_efficiency_factor == 1.8
        assert cfg.safety_margins.ha_failures_to_tolerate == 2
        assert cfg.algorithm_weights.gamma_pod_headroom == pytest.approx(0.25)

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigLoadError, match="file not found"):
            load_plan_config(tmp_path / "missing.yaml")

    def test_invalid_yaml_syntax(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "bad.yaml", "{ unclosed: [bracket")
        with pytest.raises(ConfigLoadError, match="invalid YAML"):
            load_plan_config(p)

    def test_non_mapping_top_level(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "list.yaml", "- item1\n- item2\n")
        with pytest.raises(ConfigLoadError, match="expected a YAML mapping"):
            load_plan_config(p)

    def test_validation_error_bad_weight_sum(self, tmp_path: Path) -> None:
        content = """\
algorithm_weights:
  alpha_balance: 0.9
  beta_spread: 0.9
  gamma_pod_headroom: 0.9
  delta_frag_penalty: 0.9
"""
        p = _write(tmp_path, "config.yaml", content)
        with pytest.raises(ConfigLoadError, match="validation failed"):
            load_plan_config(p)


# ═══════════════════════════════════════════════════════════════════════
# load_inventory_config
# ═══════════════════════════════════════════════════════════════════════


class TestLoadInventoryConfig:
    """InventoryConfig: empty profiles is valid (pure greenfield)."""

    def test_none_path_returns_empty(self) -> None:
        inv = load_inventory_config(None)
        assert inv.profiles == []

    def test_empty_file_returns_empty(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "inventory.yaml", "")
        inv = load_inventory_config(p)
        assert inv.profiles == []

    def test_single_profile(self, tmp_path: Path) -> None:
        content = """\
profiles:
  - profile_name: r740-existing
    cpu_topology:
      sockets: 2
      cores_per_socket: 24
      threads_per_core: 1
    ram_gb: 512
    quantity: 12
"""
        p = _write(tmp_path, "inventory.yaml", content)
        inv = load_inventory_config(p)
        assert len(inv.profiles) == 1
        profile = inv.profiles[0]
        assert profile.profile_name == "r740-existing"
        assert profile.cpu_topology.sockets == 2
        assert profile.cpu_topology.cores_per_socket == 24
        assert profile.cpu_topology.threads_per_core == 1
        assert profile.ram_gb == 512
        assert profile.quantity == 12

    def test_multiple_profiles(self, tmp_path: Path) -> None:
        content = """\
profiles:
  - profile_name: small
    cpu_topology: { sockets: 1, cores_per_socket: 8 }
    ram_gb: 64
    quantity: 4
  - profile_name: large
    cpu_topology: { sockets: 2, cores_per_socket: 32, threads_per_core: 2 }
    ram_gb: 1024
    quantity: 2
"""
        p = _write(tmp_path, "inventory.yaml", content)
        inv = load_inventory_config(p)
        assert len(inv.profiles) == 2
        assert inv.profiles[0].profile_name == "small"
        assert inv.profiles[1].profile_name == "large"
        assert inv.profiles[1].cpu_topology.threads_per_core == 2

    def test_default_quantity_is_one(self, tmp_path: Path) -> None:
        content = """\
profiles:
  - profile_name: test
    cpu_topology: { sockets: 1, cores_per_socket: 8 }
    ram_gb: 64
"""
        p = _write(tmp_path, "inventory.yaml", content)
        inv = load_inventory_config(p)
        assert inv.profiles[0].quantity == 1

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigLoadError, match="file not found"):
            load_inventory_config(tmp_path / "missing.yaml")

    def test_validation_error(self, tmp_path: Path) -> None:
        content = """\
profiles:
  - profile_name: bad
    cpu_topology: { sockets: 0, cores_per_socket: 8 }
    ram_gb: 64
"""
        p = _write(tmp_path, "inventory.yaml", content)
        with pytest.raises(ConfigLoadError, match="validation failed"):
            load_inventory_config(p)


# ═══════════════════════════════════════════════════════════════════════
# load_catalog_config
# ═══════════════════════════════════════════════════════════════════════


class TestLoadCatalogConfig:
    """CatalogConfig: at least one profile if file provided; None for inventory-only."""

    def test_none_path_returns_none(self) -> None:
        """Inventory-only mode: no catalog file → None."""
        assert load_catalog_config(None) is None

    def test_single_profile(self, tmp_path: Path) -> None:
        content = """\
profiles:
  - profile_name: r760-new
    cpu_topology:
      sockets: 2
      cores_per_socket: 32
      threads_per_core: 2
    ram_gb: 1024
    cost_weight: 1.0
"""
        p = _write(tmp_path, "catalog.yaml", content)
        cat = load_catalog_config(p)
        assert cat is not None
        assert len(cat.profiles) == 1
        assert cat.profiles[0].profile_name == "r760-new"
        assert cat.profiles[0].cost_weight == 1.0

    def test_multiple_profiles(self, tmp_path: Path) -> None:
        content = """\
profiles:
  - profile_name: standard
    cpu_topology: { sockets: 2, cores_per_socket: 32, threads_per_core: 2 }
    ram_gb: 512
    cost_weight: 1.0
  - profile_name: premium
    cpu_topology: { sockets: 2, cores_per_socket: 64, threads_per_core: 2 }
    ram_gb: 2048
    cost_weight: 3.5
"""
        p = _write(tmp_path, "catalog.yaml", content)
        cat = load_catalog_config(p)
        assert cat is not None
        assert len(cat.profiles) == 2
        assert cat.profiles[1].cost_weight == 3.5

    def test_default_cost_weight(self, tmp_path: Path) -> None:
        content = """\
profiles:
  - profile_name: test
    cpu_topology: { sockets: 1, cores_per_socket: 16, threads_per_core: 2 }
    ram_gb: 256
"""
        p = _write(tmp_path, "catalog.yaml", content)
        cat = load_catalog_config(p)
        assert cat is not None
        assert cat.profiles[0].cost_weight == 1.0

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigLoadError, match="file not found"):
            load_catalog_config(tmp_path / "missing.yaml")

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "catalog.yaml", "")
        with pytest.raises(ConfigLoadError, match="must not be empty"):
            load_catalog_config(p)

    def test_empty_profiles_list_raises(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "catalog.yaml", "profiles: []\n")
        with pytest.raises(ConfigLoadError, match="validation failed"):
            load_catalog_config(p)

    def test_invalid_yaml(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "catalog.yaml", ": invalid: : yaml:")
        with pytest.raises(ConfigLoadError, match="invalid YAML"):
            load_catalog_config(p)

    def test_validation_error_missing_profile_name(self, tmp_path: Path) -> None:
        content = """\
profiles:
  - cpu_topology: { sockets: 1, cores_per_socket: 8 }
    ram_gb: 64
"""
        p = _write(tmp_path, "catalog.yaml", content)
        with pytest.raises(ConfigLoadError, match="validation failed"):
            load_catalog_config(p)


# ═══════════════════════════════════════════════════════════════════════
# ConfigLoadError
# ═══════════════════════════════════════════════════════════════════════


class TestConfigLoadError:
    def test_message_includes_path_and_reason(self) -> None:
        err = ConfigLoadError(Path("/tmp/foo.yaml"), "file not found")
        assert "/tmp/foo.yaml" in str(err)
        assert "file not found" in str(err)

    def test_attributes(self) -> None:
        p = Path("/etc/config.yaml")
        err = ConfigLoadError(p, "bad format")
        assert err.path == p
        assert err.reason == "bad format"
