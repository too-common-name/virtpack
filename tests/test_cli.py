"""Tests for cli/main.py — the ``virtpack`` CLI entry point (LLD §4).

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
# init command tests
# ═══════════════════════════════════════════════════════════════════════


class TestInitCommand:
    """Tests for the ``virtpack init`` command."""

    def test_generates_all_stubs(self, tmp_path: Path) -> None:
        """Init creates config.yaml, inventory.yaml, and catalog.yaml."""
        result = runner.invoke(app, ["init", "--output-dir", str(tmp_path)])
        assert result.exit_code == 0, result.output

        assert (tmp_path / "config.yaml").exists()
        assert (tmp_path / "inventory.yaml").exists()
        assert (tmp_path / "catalog.yaml").exists()

    def test_stubs_are_valid_yaml(self, tmp_path: Path) -> None:
        """Generated stubs parse and validate against Pydantic models."""
        from loaders.yaml_loader import (
            load_catalog_config,
            load_inventory_config,
            load_plan_config,
        )

        runner.invoke(app, ["init", "--output-dir", str(tmp_path)])

        plan = load_plan_config(tmp_path / "config.yaml")
        assert plan.overcommit.cpu_ratio == 8.0

        inv = load_inventory_config(tmp_path / "inventory.yaml")
        assert len(inv.profiles) == 1

        cat = load_catalog_config(tmp_path / "catalog.yaml")
        assert cat is not None
        assert len(cat.profiles) == 2

    def test_skips_existing_files(self, tmp_path: Path) -> None:
        """Existing files are not overwritten without --force."""
        (tmp_path / "config.yaml").write_text("# my custom config\n")

        result = runner.invoke(app, ["init", "--output-dir", str(tmp_path)])
        assert result.exit_code == 0

        # config.yaml should NOT be overwritten
        assert (tmp_path / "config.yaml").read_text() == "# my custom config\n"
        # But others should be created
        assert (tmp_path / "inventory.yaml").exists()
        assert "already exists" in result.output

    def test_force_overwrites(self, tmp_path: Path) -> None:
        """--force overwrites existing files."""
        (tmp_path / "config.yaml").write_text("# old\n")

        result = runner.invoke(app, ["init", "--output-dir", str(tmp_path), "--force"])
        assert result.exit_code == 0

        content = (tmp_path / "config.yaml").read_text()
        assert "cluster_limits" in content  # overwritten with stub

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        """Output directory is auto-created if it doesn't exist."""
        nested = tmp_path / "deep" / "nested"

        result = runner.invoke(app, ["init", "--output-dir", str(nested)])
        assert result.exit_code == 0
        assert (nested / "config.yaml").exists()


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
                "plan",
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
                "plan",
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
                "plan",
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
                "plan",
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
                "plan",
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
                "plan",
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
                "plan",
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
                "plan",
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
                "plan",
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
                "plan",
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


# ═══════════════════════════════════════════════════════════════════════
# Consolidation strategy tests
# ═══════════════════════════════════════════════════════════════════════


class TestConsolidateStrategy:
    """Tests for ``--strategy consolidate`` CLI flag."""

    def test_consolidate_places_all_vms(self, tmp_path: Path) -> None:
        """Consolidate mode places all VMs on fewer nodes."""
        cfg = _write_config_yaml(tmp_path)
        inv = _write_inventory_yaml(tmp_path)  # 2 generous nodes
        xlsx = _write_rvtools_xlsx(tmp_path, vm_count=3)
        out_dir = tmp_path / "out"

        result = runner.invoke(
            app,
            [
                "plan",
                "--rvtools",
                str(xlsx),
                "--config",
                str(cfg),
                "--inventory",
                str(inv),
                "--output",
                str(out_dir),
                "--no-auto-discovery",
                "--strategy",
                "consolidate",
            ],
        )

        assert result.exit_code == 0, result.output
        assert (out_dir / "placement_map.csv").exists()
        # All 3 VMs should be placed
        csv_lines = (out_dir / "placement_map.csv").read_text().strip().splitlines()
        assert len(csv_lines) == 4  # header + 3

    def test_consolidate_uses_fewer_nodes(self, tmp_path: Path) -> None:
        """Consolidate should use fewer nodes than spread for the same input."""
        cfg = _write_config_yaml(tmp_path)
        inv = _write_inventory_yaml(tmp_path)  # 2 generous nodes
        xlsx = _write_rvtools_xlsx(tmp_path, vm_count=2)

        # Spread
        out_spread = tmp_path / "out_spread"
        result_spread = runner.invoke(
            app,
            [
                "plan",
                "--rvtools",
                str(xlsx),
                "--config",
                str(cfg),
                "--inventory",
                str(inv),
                "--output",
                str(out_spread),
                "--no-auto-discovery",
                "--strategy",
                "spread",
            ],
        )
        assert result_spread.exit_code == 0, result_spread.output

        # Consolidate
        out_cons = tmp_path / "out_cons"
        result_cons = runner.invoke(
            app,
            [
                "plan",
                "--rvtools",
                str(xlsx),
                "--config",
                str(cfg),
                "--inventory",
                str(inv),
                "--output",
                str(out_cons),
                "--no-auto-discovery",
                "--strategy",
                "consolidate",
            ],
        )
        assert result_cons.exit_code == 0, result_cons.output

        # Count distinct nodes used in each placement map
        def _count_nodes(csv_path: Path) -> int:
            import csv as _csv

            with csv_path.open() as f:
                reader = _csv.DictReader(f)
                return len({row["Target_Node"] for row in reader})

        spread_nodes = _count_nodes(out_spread / "placement_map.csv")
        cons_nodes = _count_nodes(out_cons / "placement_map.csv")

        # Consolidate should use fewer or equal nodes
        assert cons_nodes <= spread_nodes

    def test_consolidate_reports_shutdown(self, tmp_path: Path) -> None:
        """Consolidate mode should mention 'Shutdown' in output."""
        cfg = _write_config_yaml(tmp_path)
        inv = _write_inventory_yaml(tmp_path)  # 2 generous nodes
        xlsx = _write_rvtools_xlsx(tmp_path, vm_count=1)  # 1 tiny VM
        out_dir = tmp_path / "out"

        result = runner.invoke(
            app,
            [
                "plan",
                "--rvtools",
                str(xlsx),
                "--config",
                str(cfg),
                "--inventory",
                str(inv),
                "--output",
                str(out_dir),
                "--no-auto-discovery",
                "--strategy",
                "consolidate",
            ],
        )

        assert result.exit_code == 0, result.output
        # With 2 inventory nodes and 1 small VM, 1 node should be unused
        assert "Shutdown" in result.output

    def test_consolidate_debug_shows_strategy(self, tmp_path: Path) -> None:
        """Debug mode should print the strategy name."""
        cfg = _write_config_yaml(tmp_path)
        inv = _write_inventory_yaml(tmp_path)
        xlsx = _write_rvtools_xlsx(tmp_path, vm_count=1)
        out_dir = tmp_path / "out"

        result = runner.invoke(
            app,
            [
                "plan",
                "--rvtools",
                str(xlsx),
                "--config",
                str(cfg),
                "--inventory",
                str(inv),
                "--output",
                str(out_dir),
                "--no-auto-discovery",
                "--strategy",
                "consolidate",
                "--debug",
            ],
        )

        assert result.exit_code == 0, result.output
        assert "consolidate" in result.output.lower()
