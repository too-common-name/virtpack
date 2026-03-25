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

### Running

```bash
# Brownfield spread: auto-discover hosts, distribute across all
virtpack plan --rvtools export.xlsx --output ./results

# Brownfield consolidate: pack tightly, report shutdown candidates
virtpack plan --rvtools export.xlsx --strategy consolidate --output ./results

# With a config override
virtpack plan --rvtools export.xlsx --config config.yaml --output ./results

# Greenfield: with hardware catalog
virtpack plan --rvtools export.xlsx --catalog catalog.yaml --output ./results

# Debug mode: verbose placement decisions
virtpack plan --rvtools export.xlsx --debug
```

### Output

| File | Description |
|------|-------------|
| `placement_map.csv` | VM-to-Node mapping (audit trail) |
| Terminal summary | Node counts, utilization, bottleneck, engineering metrics |

## Configuration

### config.yaml (Global)

```yaml
cluster_limits:
  max_pods_per_node: 250

overcommit:
  cpu_ratio: 8.0        # vCPU overcommit
  memory_ratio: 1.0     # no memory overcommit

virt_overheads:
  ht_efficiency_factor: 1.5   # HT ≠ full cores (Red Hat sizing guide)
  ocp_virt_core: 2.0
  ocp_virt_memory_mb: 360
  eviction_hard_mb: 100

safety_margins:
  utilization_targets:
    cpu: 85              # percent
    memory: 80
  ha_failures_to_tolerate: 1

algorithm_weights:
  alpha_balance: 0.3     # CPU/memory proportionality
  beta_spread: 0.3       # LeastAllocated (distribute VMs)
  gamma_pod_headroom: 0.1  # pod slot availability
  delta_frag_penalty: 0.3  # stranded capacity penalty
```

### inventory.yaml (Brownfield)

```yaml
profiles:
  - profile_name: r740-existing
    cpu_topology: { sockets: 2, cores_per_socket: 24, threads_per_core: 1 }
    ram_gb: 512
    quantity: 12
```

### catalog.yaml (Greenfield)

```yaml
profiles:
  - profile_name: r760-new
    cpu_topology: { sockets: 2, cores_per_socket: 32, threads_per_core: 2 }
    ram_gb: 1024
    cost_weight: 1.0
```

## Algorithm Overview

The placement engine implements a **First-Fit-Decreasing (FFD)** heuristic for multidimensional vector bin packing (Speitkamp & Bichler, 2010):

1. **Sort** VMs by memory (descending)
2. **For each VM:**
   - **Filter** candidate nodes (hard constraints: CPU, memory, pods)
   - **Expand** — if no node fits and a catalog exists, create the cheapest suitable node
   - **Score** candidates using a weighted function + Lookahead (k=2)
   - **Bind** VM to the highest-scoring node

### Scoring Function

```
score(node) = α·balance + β·spread + γ·pod_headroom − δ·stranded_penalty
```

| Term | Signal | Formula |
|------|--------|---------|
| **balance** | CPU/memory proportionality | `1 − abs(cpu_util − mem_util)` |
| **spread** | LeastAllocated (use all nodes) | `((1−cpu_util) + (1−mem_util)) / 2` |
| **pod_headroom** | Pod slot availability | `1 − (pods_used / max_pods)` |
| **stranded_penalty** | Dimensional imbalance of remaining capacity | `(cpu_rem% − mem_rem%)²` |

Each term provides a **unique, non-redundant signal**. The weights are configurable to match the customer's operational philosophy — see `docs/hld.md` §3.5 for tuning guidance.

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
├── cli/main.py              # Typer CLI entry point
├── core/
│   ├── cluster_state.py     # Mutable node state (O(1) place/unplace)
│   ├── placement_engine.py  # FFD + Lookahead simulation loop
│   ├── normalizer.py        # VM/Node resource normalization
│   └── ha_injector.py       # HA spare capacity injection
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
