"""Tests for cli/main.py — the ``virtpack plan`` CLI entry point (LLD §4).

Uses ``typer.testing.CliRunner`` for isolated, non-interactive invocations.
Each test creates its own YAML config + RVTools fixtures in a ``tmp_path``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from typer.testing import CliRunner

from cli.main import app

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()

# ═══════════════════════════════════════════════════════════════════════
# Fixture helpers
# ═══════════════════════════════════════════════════════════════════════


def _write_config_yaml(path: Path) -> Path:
    """Write a minimal ``config.yaml`` with defaults."""
    cfg = path / "config.yaml"
    cfg.write_text(
        """\
cluster_limits:
  max_pods_per_node: 250
overcommit:
  cpu_ratio: 1.0
  memory_ratio: 1.0
safety_margins:
  utilization_targets:
    cpu: 100
    memory: 100
  ha_failures_to_tolerate: 0
algorithm_weights:
  alpha_balance: 0.3
  beta_spread: 0.3
  gamma_pod_headroom: 0.1
  delta_frag_penalty: 0.3
"""
    )
    return cfg


def _write_inventory_yaml(path: Path) -> Path:
    """Write an inventory with 1 node that has generous capacity."""
    inv = path / "inventory.yaml"
    inv.write_text(
        """\
profiles:
  - profile_name: test-node
    cpu_topology:
      sockets: 2
      cores_per_socket: 16
    ram_gb: 256
    quantity: 2
"""
    )
    return inv


def _write_catalog_yaml(path: Path) -> Path:
    """Write a catalog with 1 profile."""
    cat = path / "catalog.yaml"
    cat.write_text(
        """\
profiles:
  - profile_name: cat-node
    cpu_topology:
      sockets: 2
      cores_per_socket: 16
    ram_gb: 256
    cost_weight: 1.0
"""
    )
    return cat


def _write_rvtools_xlsx(path: Path, *, vm_count: int = 3) -> Path:
    """Write a synthetic RVTools Excel file with ``vInfo`` and ``vHost`` sheets."""
    import pandas as pd

    xlsx_path = path / "rvtools.xlsx"

    # vInfo
    vinfo_data = {
        "VM": [f"vm-{i:03d}" for i in range(1, vm_count + 1)],
        "CPUs": [2] * vm_count,
        "Memory": [4096] * vm_count,
        "Powerstate": ["poweredOn"] * vm_count,
        "Template": ["false"] * vm_count,
    }
    df_vinfo = pd.DataFrame(vinfo_data)

    # vHost (minimal: 1 host)
    vhost_data = {
        "Host": ["esxi-01"],
        "# CPU": [2],
        "Cores per CPU": [8],
        "HT Active": ["True"],
        "# Memory": [131072],
    }
    df_vhost = pd.DataFrame(vhost_data)

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df_vinfo.to_excel(writer, sheet_name="vInfo", index=False)
        df_vhost.to_excel(writer, sheet_name="vHost", index=False)

    return xlsx_path


# ═══════════════════════════════════════════════════════════════════════
# CLI smoke tests
# ═══════════════════════════════════════════════════════════════════════


class TestPlanCommand:
    """Tests for the ``virtpack plan`` command."""

    def test_smoke_all_placed(self, tmp_path: Path) -> None:
        """All VMs fit on inventory → exit code 0, CSV written."""
        cfg = _write_config_yaml(tmp_path)
        inv = _write_inventory_yaml(tmp_path)
        xlsx = _write_rvtools_xlsx(tmp_path, vm_count=3)
        out_dir = tmp_path / "out"

        result = runner.invoke(
            app,
            [
                "--rvtools",
                str(xlsx),
                "--config",
                str(cfg),
                "--inventory",
                str(inv),
                "--output",
                str(out_dir),
                "--no-auto-discovery",
            ],
        )

        assert result.exit_code == 0, result.output
        assert (out_dir / "placement_map.csv").exists()
        # CSV should have 3 rows (1 header + 3 data)
        lines = (out_dir / "placement_map.csv").read_text().strip().splitlines()
        assert len(lines) == 4  # header + 3 VMs

    def test_unplaced_vms_exit_code_2(self, tmp_path: Path) -> None:
        """VMs that can't fit → exit code 2."""
        # Config with aggressive overheads that leave little room
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            """\
cluster_limits:
  max_pods_per_node: 1
overcommit:
  cpu_ratio: 1.0
  memory_ratio: 1.0
safety_margins:
  utilization_targets:
    cpu: 100
    memory: 100
  ha_failures_to_tolerate: 0
algorithm_weights:
  alpha_balance: 0.3
  beta_spread: 0.3
  gamma_pod_headroom: 0.1
  delta_frag_penalty: 0.3
"""
        )
        inv = _write_inventory_yaml(tmp_path)
        xlsx = _write_rvtools_xlsx(tmp_path, vm_count=5)
        out_dir = tmp_path / "out"

        result = runner.invoke(
            app,
            [
                "--rvtools",
                str(xlsx),
                "--config",
                str(cfg),
                "--inventory",
                str(inv),
                "--output",
                str(out_dir),
                "--no-auto-discovery",
            ],
        )

        # max_pods_per_node=1, 2 nodes, 5 VMs → 3 unplaced → exit code 2
        assert result.exit_code == 2

    def test_debug_flag(self, tmp_path: Path) -> None:
        """--debug prints extra diagnostic messages."""
        cfg = _write_config_yaml(tmp_path)
        inv = _write_inventory_yaml(tmp_path)
        xlsx = _write_rvtools_xlsx(tmp_path, vm_count=2)
        out_dir = tmp_path / "out"

        result = runner.invoke(
            app,
            [
                "--rvtools",
                str(xlsx),
                "--config",
                str(cfg),
                "--inventory",
                str(inv),
                "--output",
                str(out_dir),
                "--no-auto-discovery",
                "--debug",
            ],
        )

        assert result.exit_code == 0, result.output
        assert "Parsed" in result.output
        assert "Normalized" in result.output

    def test_auto_discovery_enabled(self, tmp_path: Path) -> None:
        """Without --no-auto-discovery, vHost hosts are added to inventory."""
        cfg = _write_config_yaml(tmp_path)
        xlsx = _write_rvtools_xlsx(tmp_path, vm_count=2)
        out_dir = tmp_path / "out"

        result = runner.invoke(
            app,
            [
                "--rvtools",
                str(xlsx),
                "--config",
                str(cfg),
                "--output",
                str(out_dir),
                "--debug",
            ],
        )

        assert result.exit_code == 0, result.output
        # Should see auto-discovered host in debug output
        assert "Auto-discovered" in result.output

    def test_inventory_only_no_catalog(self, tmp_path: Path) -> None:
        """No --catalog → inventory-only mode, unplaced VMs if no room."""
        cfg = _write_config_yaml(tmp_path)
        inv = _write_inventory_yaml(tmp_path)
        xlsx = _write_rvtools_xlsx(tmp_path, vm_count=3)
        out_dir = tmp_path / "out"

        result = runner.invoke(
            app,
            [
                "--rvtools",
                str(xlsx),
                "--config",
                str(cfg),
                "--inventory",
                str(inv),
                "--output",
                str(out_dir),
                "--no-auto-discovery",
            ],
        )

        # VMs should fit on 2 inventory nodes
        assert result.exit_code == 0, result.output

    def test_with_catalog_expansion(self, tmp_path: Path) -> None:
        """Catalog nodes are added when inventory is insufficient."""
        # Tiny inventory that can't fit much
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            """\
cluster_limits:
  max_pods_per_node: 250
overcommit:
  cpu_ratio: 1.0
  memory_ratio: 1.0
safety_margins:
  utilization_targets:
    cpu: 100
    memory: 100
  ha_failures_to_tolerate: 0
algorithm_weights:
  alpha_balance: 0.3
  beta_spread: 0.3
  gamma_pod_headroom: 0.1
  delta_frag_penalty: 0.3
"""
        )
        inv = tmp_path / "inventory.yaml"
        inv.write_text(
            """\
profiles:
  - profile_name: tiny
    cpu_topology:
      sockets: 1
      cores_per_socket: 4
    ram_gb: 16
    quantity: 1
"""
        )
        cat = _write_catalog_yaml(tmp_path)
        xlsx = _write_rvtools_xlsx(tmp_path, vm_count=5)
        out_dir = tmp_path / "out"

        result = runner.invoke(
            app,
            [
                "--rvtools",
                str(xlsx),
                "--config",
                str(cfg),
                "--inventory",
                str(inv),
                "--catalog",
                str(cat),
                "--output",
                str(out_dir),
                "--no-auto-discovery",
            ],
        )

        assert result.exit_code == 0, result.output
        # CSV should contain all 5 VMs
        csv_path = out_dir / "placement_map.csv"
        lines = csv_path.read_text().strip().splitlines()
        assert len(lines) == 6  # header + 5 VMs

    def test_missing_rvtools_file(self, tmp_path: Path) -> None:
        """Missing RVTools file → exit code 1 with error message."""
        cfg = _write_config_yaml(tmp_path)
        out_dir = tmp_path / "out"

        result = runner.invoke(
            app,
            [
                "--rvtools",
                str(tmp_path / "nonexistent.xlsx"),
                "--config",
                str(cfg),
                "--output",
                str(out_dir),
                "--no-auto-discovery",
            ],
        )

        assert result.exit_code == 1

    def test_invalid_config_yaml(self, tmp_path: Path) -> None:
        """Invalid config YAML → exit code 1."""
        bad_cfg = tmp_path / "config.yaml"
        bad_cfg.write_text("cluster_limits:\n  max_pods_per_node: -1\n")
        xlsx = _write_rvtools_xlsx(tmp_path, vm_count=1)
        out_dir = tmp_path / "out"

        result = runner.invoke(
            app,
            [
                "--rvtools",
                str(xlsx),
                "--config",
                str(bad_cfg),
                "--output",
                str(out_dir),
                "--no-auto-discovery",
            ],
        )

        assert result.exit_code == 1

    def test_defaults_without_config(self, tmp_path: Path) -> None:
        """No --config → defaults are used, should still work."""
        xlsx = _write_rvtools_xlsx(tmp_path, vm_count=2)
        out_dir = tmp_path / "out"

        result = runner.invoke(
            app,
            [
                "--rvtools",
                str(xlsx),
                "--output",
                str(out_dir),
                "--no-auto-discovery",
            ],
        )

        # With default overcommit (8:1), 2 small VMs should fit somewhere
        # even without inventory (they'll be unplaced since no nodes)
        # Actually — no inventory AND no catalog = all unplaced = exit 2
        assert result.exit_code == 2

    def test_output_dir_created(self, tmp_path: Path) -> None:
        """Output directory is auto-created if it doesn't exist."""
        cfg = _write_config_yaml(tmp_path)
        inv = _write_inventory_yaml(tmp_path)
        xlsx = _write_rvtools_xlsx(tmp_path, vm_count=1)
        nested_out = tmp_path / "deep" / "nested" / "out"

        result = runner.invoke(
            app,
            [
                "--rvtools",
                str(xlsx),
                "--config",
                str(cfg),
                "--inventory",
                str(inv),
                "--output",
                str(nested_out),
                "--no-auto-discovery",
            ],
        )

        assert result.exit_code == 0, result.output
        assert nested_out.exists()
        assert (nested_out / "placement_map.csv").exists()
