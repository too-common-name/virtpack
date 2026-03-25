# virtpack

**Deterministic OpenShift Virtualization capacity planner CLI.**

`virtpack` simulates placement of Virtual Machines onto bare-metal OpenShift nodes using a multidimensional bin-packing heuristic. It parses real workloads from RVTools exports, normalizes resources with Red Hat overhead math, and generates auditable placement reports.

## Use Cases

### Scenario A1 — Pure Brownfield / Spread (default)

> *"I have 15 VMware hosts. Can I reuse them ALL for OpenShift Virtualization?"*

Provide only the RVTools export — `virtpack` auto-discovers physical hosts from the `vHost` sheet and distributes VMs across all existing hardware.

```bash
virtpack plan --rvtools rvtools.xlsx
```

### Scenario A2 — Pure Brownfield / Consolidate

> *"I have 15 VMware hosts, but I want to minimize OCP subscriptions. Which nodes can I shut down?"*

Same input, but with `--strategy consolidate`. The engine packs VMs tightly onto the fewest inventory nodes and reports which ones can be powered off.

```bash
virtpack plan --rvtools rvtools.xlsx --strategy consolidate
```

### Scenario B — Pure Greenfield

> *"I'm buying new hardware. What's the minimum set of nodes?"*

Provide a hardware catalog with available server profiles. The engine creates new nodes on-demand, only purchasing hardware when existing capacity is exhausted.

```bash
virtpack plan --rvtools rvtools.xlsx --catalog catalog.yaml
```

### Scenario C — Hybrid

> *"I have some old servers and I may need to buy more."*

Provide both inventory and catalog. Existing hardware is filled first (free), and catalog expansion happens only when inventory is exhausted.

```bash
virtpack plan --rvtools rvtools.xlsx --inventory inventory.yaml --catalog catalog.yaml
```

## Quick Start

### Installation

Requires Python 3.12+.

```bash
# Clone and install in development mode
git clone <repo-url> && cd virtpack
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
```

### Generate Config Stubs

Use `virtpack init` to generate annotated YAML templates with sensible defaults:

```bash
# Generate config.yaml, inventory.yaml, catalog.yaml in the current directory
virtpack init

# Generate into a specific directory
virtpack init --output-dir config/

# Overwrite existing files
virtpack init --output-dir config/ --force
```

Edit the generated files for your environment, then run the planner.

### Running

```bash
# Brownfield spread (default): auto-discover hosts, distribute across all
virtpack plan --rvtools export.xlsx

# Brownfield consolidate: pack tightly, report shutdown candidates
virtpack plan --rvtools export.xlsx --strategy consolidate

# With a config override
virtpack plan --rvtools export.xlsx --config config.yaml

# Greenfield: with hardware catalog
virtpack plan --rvtools export.xlsx --catalog catalog.yaml

# Hybrid: existing inventory + catalog for expansion
virtpack plan --rvtools export.xlsx --inventory inventory.yaml --catalog catalog.yaml

# Disable vHost auto-discovery (use only YAML inventory)
virtpack plan --rvtools export.xlsx --inventory inventory.yaml --no-auto-discovery

# Debug mode: verbose placement decisions
virtpack plan --rvtools export.xlsx --debug

# Custom output directory (default: ./out)
virtpack plan --rvtools export.xlsx --output ./results
```

### CLI Reference

| Command | Description |
|---------|-------------|
| `virtpack init` | Generate stub YAML config files |
| `virtpack plan` | Run the capacity planning simulation |

**`virtpack plan` options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--rvtools PATH` | RVTools `.xlsx` export (vInfo + optional vHost) | *required* |
| `--config PATH` | Global `config.yaml` | built-in defaults |
| `--catalog PATH` | Greenfield hardware `catalog.yaml` | none (brownfield only) |
| `--inventory PATH` | Brownfield `inventory.yaml` | none (auto-discovery) |
| `--output PATH` | Output directory for `placement_map.csv` | `./out` |
| `--strategy` | `spread` or `consolidate` | `spread` |
| `--no-auto-discovery` | Disable `vHost` inventory auto-parsing | `false` |
| `--debug` | Verbose placement logs | `false` |

**`virtpack init` options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--output-dir, -o PATH` | Directory to write stub files into | `.` (current dir) |
| `--force, -f` | Overwrite existing files | `false` |

### Output

| File | Description |
|------|-------------|
| `placement_map.csv` | VM-to-Node mapping (audit trail) |
| Terminal summary | Rich formatted report (see below) |

**Terminal report sections:**

1. **VMware Source Environment** — Physical hosts, cores, RAM, VM demand, overcommit ratios (requires `vHost` sheet)
2. **OCP Virt Cluster Plan** — Node counts, peak utilization, bottleneck, headroom, shutdown candidates
3. **Node Utilization** — Per-node CPU/memory breakdown table
4. **Engineering Metrics** — CFI, node pressure P95/max, cluster utilization
5. **Migration Comparison** — VMware vs OCP Virt side-by-side (capacity, nodes, delta)
6. **HA Status** — Whether HA spare capacity is satisfied, nodes reclaimed, or deficit
7. **Unplaced VMs** — List of VMs that could not be placed (if any)

## Configuration

All configuration files are optional. Sensible defaults are built in.
Use `virtpack init` to generate annotated stubs as a starting point.

### config.yaml (Global)

```yaml
cluster_limits:
  max_pods_per_node: 250

overcommit:
  cpu_ratio: 8.0        # vCPU overcommit (8 vCPU → 1 physical core)
  memory_ratio: 1.0     # no memory overcommit (1:1 reservation)

virt_overheads:
  ht_efficiency_factor: 1.5   # HT ≠ full cores (Red Hat sizing guide)
  ocp_virt_core: 2.0          # CPU cores reserved for OCP Virt stack
  ocp_virt_memory_mb: 360     # Memory reserved for OCP Virt stack (MB)
  eviction_hard_mb: 100       # kubelet hard eviction threshold (MB)

safety_margins:
  utilization_targets:
    cpu: 85              # percent — max planned CPU utilization
    memory: 80           # percent — max planned memory utilization
  ha_failures_to_tolerate: 1

algorithm_weights:
  # Weights must sum to 1.0
  alpha_balance: 0.3     # CPU/memory proportionality
  beta_spread: 0.3       # LeastAllocated (distribute VMs)
  gamma_pod_headroom: 0.1  # pod slot availability
  delta_frag_penalty: 0.3  # stranded capacity penalty

placement_strategy: spread  # spread | consolidate
```

> **Understanding `cpu_ratio`:** A `cpu_ratio` of 8.0 means each vCPU consumes 1/8 of a physical core. Note that this ratio operates on the **usable** node capacity (after subtracting kubelet, OCP Virt overheads, and safety margins), not on raw logical CPUs. A VMware "2:1 overcommit" does not equal a virtpack "2:1 overcommit" — virtpack is more conservative because it accounts for real OpenShift overhead that VMware silently absorbs.

> **Understanding safety margins:** These exist to bridge the gap between offline capacity planning and real-time scheduling. `virtpack` is an offline simulator that predicts placement ahead of time, but in production, DaemonSets, monitoring agents, kubelet memory drift, and workload bursts consume resources not modeled here. The safety margin reserves headroom so the real K8s scheduler has room to operate.

### inventory.yaml (Brownfield)

```yaml
profiles:
  - profile_name: r740-existing
    cpu_topology: { sockets: 2, cores_per_socket: 24, threads_per_core: 1 }
    ram_gb: 512
    quantity: 12
```

> **Tip:** For lift-and-shift migrations, you often don't need this file. `virtpack` auto-discovers physical hosts from the RVTools `vHost` sheet and injects them as cost=0 inventory nodes automatically. Use `--no-auto-discovery` to disable this and rely only on the YAML file.

### catalog.yaml (Greenfield)

```yaml
profiles:
  - profile_name: r760-new
    cpu_topology: { sockets: 2, cores_per_socket: 32, threads_per_core: 2 }
    ram_gb: 1024
    cost_weight: 1.0
```

> **Tip:** This file is optional. Omit it for pure brownfield scenarios. The Expander only creates catalog nodes when no existing inventory node can fit a VM.

## Algorithm Overview

The placement engine implements a **First-Fit-Decreasing (FFD)** heuristic for multidimensional vector bin packing (Speitkamp & Bichler, 2010):

1. **Sort** VMs by memory (descending)
2. **For each VM:**
   - **Filter** candidate nodes (hard constraints: CPU, memory, pods)
   - **Expand** — if no node fits: pull from inventory pool (consolidate) or create the cheapest catalog node (greenfield)
   - **Score** candidates on **projected state** (after tentatively placing the VM) + Lookahead (k=2)
   - **Bind** VM to the highest-scoring node

### Scoring Function

```
score(node) = α·balance + β·spread + γ·pod_headroom − δ·stranded_penalty
```

All scoring is performed on the **projected** state — the node is evaluated *after* tentatively placing the VM, not before. This enables stranded nodes to "attract" complementary VMs that reduce their dimensional imbalance.

| Term | Signal | Formula |
|------|--------|---------|
| **balance** | CPU/memory proportionality | `1 − abs(cpu_util − mem_util)` |
| **spread** | LeastAllocated (use all nodes) | `((1−cpu_util) + (1−mem_util)) / 2` |
| **pod_headroom** | Pod slot availability | `1 − (pods_used / max_pods)` |
| **stranded_penalty** | Dimensional imbalance of remaining capacity | `(cpu_rem% − mem_rem%)²` |

Each term provides a **unique, non-redundant signal**. The weights are configurable to match the customer's operational philosophy — see `docs/hld.md` §3.5 for tuning guidance.

### HA Injection

After placement, the engine ensures the cluster can tolerate N simultaneous node failures (`ha_failures_to_tolerate`). If there's a deficit:

1. **Reclaim** unused inventory nodes from the shutdown pool (free, already owned)
2. **Expand** with catalog nodes if still short (requires `catalog.yaml`)
3. **Report** any remaining deficit if neither source is available

In consolidate mode, this means some "shutdown candidate" nodes may be reclaimed for HA spare capacity — the final shutdown count is adjusted accordingly.

## Development

```bash
# Lint
ruff check .

# Type check
mypy

# Tests with coverage
pytest --cov --cov-report=term-missing
```

## Architecture

```
virtpack/
├── cli/main.py              # Typer CLI entry point (plan + init)
├── core/
│   ├── cluster_state.py     # Mutable node state (O(1) place/unplace)
│   ├── placement_engine.py  # FFD + Lookahead simulation loop
│   ├── normalizer.py        # VM/Node resource normalization
│   └── ha_injector.py       # HA spare capacity injection (reclaim + expand)
├── models/
│   ├── vm.py, node.py       # Pydantic V2 domain models
│   └── config.py            # Configuration models (frozen)
├── algorithms/
│   ├── scorer.py            # Weighted scoring function
│   └── expander.py          # Catalog node creation
├── loaders/
│   ├── rvtools_parser.py    # RVTools Excel parsing (vInfo + vHost)
│   └── yaml_loader.py       # YAML config loading
└── report/
    ├── csv_exporter.py      # placement_map.csv
    └── terminal_summary.py  # Rich terminal output
```

## References

1. Speitkamp, B. & Bichler, M. (2010). *"A Mathematical Programming Approach for Server Consolidation Problems in Virtualized Data Centers."* IEEE Transactions on Services Computing. [PDF](https://pub.dss.in.tum.de/bichler-research/2006_bichler_capacity_planning.pdf)
2. Red Hat (2024). *OpenShift Virtualization Cluster Sizing Guide.* [Link](https://access.redhat.com/sites/default/files/attachments/openshift_virtualization_cluster_sizing_guide.pdf)

## License

TBD
