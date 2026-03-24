# virtpack: OpenShift Virtualization Capacity Planner CLI - LLD

## 1. Code Architecture
```bash
virtpack/
    ├── cli/
    │   └── main.py
    ├── core/
    │   ├── cluster_state.py
    │   ├── placement_engine.py
    │   ├── normalizer.py
    │   └── ha_injector.py
    ├── models/
    │   ├── vm.py
    │   ├── node.py
    │   └── config.py
    ├── algorithms/
    │   ├── scorer.py
    │   └── expander.py
    ├── loaders/
    │   ├── rvtools_parser.py
    │   └── yaml_loader.py
    └── report/
        ├── csv_exporter.py
        └── terminal_summary.py
```

> **Note:** The CLI package is named `cli/` (not `cmd/`) and the I/O package is named `loaders/` (not `io/`) to avoid shadowing Python built-in modules.
------------------------------------------------------------------------
## 2. Key Classes & Interfaces

### 2.1 Domain Models
Using Pydantic V2 `BaseModel` with strict typing.

#### VM (Pydantic, `strict=True`, `frozen=True`)
```python
class VM(BaseModel):
    name: str               # RVTools vInfo "VM" column
    cpu: float              # Effective vCPU (post-overcommit: vm_cpu / cpu_ratio)
    memory_mb: float        # Memory in MB (RVTools native unit)
    pods: int = 1           # Always 1 for KubeVirt virt-launcher
```

#### Node (Pydantic, `strict=True`, mutable)
```python
class Node(BaseModel):
    id: str
    profile: str
    
    # Capacity (set once after normalization + safety margins)
    cpu_total: float        # Schedulable CPU (logical cores, post-overhead)
    memory_total: float     # Schedulable memory in MB (post-overhead)
    pods_total: int         # Max pods from cluster_limits.max_pods_per_node
    
    # Usage (mutated by place/unplace during placement loop)
    cpu_used: float = 0.0
    memory_used: float = 0.0
    pods_used: int = 0
    
    # Metadata
    cost_weight: float      # 0.0 for inventory, >0 for catalog
    is_inventory: bool      # True = brownfield, False = greenfield
    
    # Derived properties (used by Scorer, HLD §6.1)
    @property cpu_util -> float       # cpu_used / cpu_total
    @property memory_util -> float    # memory_used / memory_total
    @property cpu_remaining -> float
    @property memory_remaining -> float
    @property pods_remaining -> int
    
    # Placement filter
    def fits(vm: VM) -> bool          # True if all 3 dimensions fit
    
    # Factory methods (enforce metadata invariants)
    @classmethod new_inventory(...)   # cost_weight=0.0, is_inventory=True
    @classmethod new_catalog(...)     # cost_weight>0,   is_inventory=False
```

#### Config Models (Pydantic, `frozen=True`)
```python
# config.yaml → PlanConfig
#   ├── ClusterLimits          (max_pods_per_node)
#   ├── OvercommitConfig       (cpu_ratio, memory_ratio)
#   ├── VirtOverheads          (ht_efficiency_factor, ocp_virt_core,
#   │                           ocp_virt_memory_mb, eviction_hard_mb)
#   ├── SafetyMargins
#   │   ├── UtilizationTargets (cpu, memory)
#   │   └── ha_failures_to_tolerate
#   └── AlgorithmWeights       (α, β, γ, δ — must sum to 1.0)

# Hardware topology (shared by inventory & catalog)
class CpuTopology(BaseModel):
    sockets: int
    cores_per_socket: int
    threads_per_core: int = 1
    @computed_field physical_cores -> int   # sockets × cores_per_socket
    @computed_field logical_cpus -> int     # physical_cores × threads_per_core

# inventory.yaml → InventoryConfig
#   └── profiles: list[InventoryProfile]   (may be empty)
#       └── InventoryProfile: profile_name, cpu_topology, ram_gb, quantity

# catalog.yaml → CatalogConfig
#   └── profiles: list[CatalogProfile]     (min_length=1)
#       └── CatalogProfile: profile_name, cpu_topology, ram_gb, cost_weight
```

#### ClusterState
The `unplace` method is critical for enabling $O(1)$ rollback during Lookahead heuristic simulation.

```
class ClusterState:
    nodes: List[Node]

    def place(self, vm: VM, node: Node) -> None:
        pass
            
    def unplace(self, vm: VM, node: Node) -> None:
        pass
            
    def get_candidate_nodes(self, vm: VM) -> List[Node]:
        pass
```

![Class Diagram](http://www.plantuml.com/plantuml/proxy?src=https://raw.githubusercontent.com/too-common-name/virtpack/main/doc/uml/class.puml)

### 2.2 The Engine & Algorithms
```
class PlacementEngine:

    def run(vms, state, config):
        for vm in sorted_vms:
            handle_vm(vm)

    def handle_vm(vm):
        candidates = filter(vm)
        if not candidates:
            node = expand()
        best = score(candidates)
        bind(vm, best)

class Scorer:
    # Component scores — each ∈ [0, 1]
    def balance_score(node) -> float        # 1 - abs(cpu_util - mem_util)
    def spread_score(node) -> float         # ((1-cpu_util)+(1-mem_util))/2
    def pod_headroom_score(node) -> float   # 1 - (pods_used / pods_total)
    def fragmentation_penalty(node) -> float  # (cpu_rem% - mem_rem%)²

    # Weighted combination
    def score_node(node, weights) -> float  # α·balance + β·spread + γ·pod − δ·frag

class Expander:

    def select_profile(catalog)     # cheapest profile that fits the VM

    def create_node(profile)        # normalize + build catalog node

```

![Sequence Diagram](http://www.plantuml.com/plantuml/proxy?src=https://raw.githubusercontent.com/too-common-name/virtpack/main/doc/uml/sequence_placement.puml)
------------------------------------------------------------------------

## 3. Exhaustive Testing Strategy

To ensure enterprise-grade reliability, the codebase must pass a rigorous multi-tiered `pytest` CI/CD pipeline.

### 3.1 Unit Tests (Math & Transformation)
* **MCO Overheads:** Hardcode 5 physical node sizes (e.g., 64GB, 256GB, 1TB). Assert that the `normalizer.py` outputs the exact usable capacity down to the Megabyte as specified by the Red Hat `kubelet-auto-sizing.yaml` math.
* **ETL Filtering:** Feed a mocked RVTools DataFrame containing 10 `poweredOn` VMs, 5 `poweredOff`, 3 `SRM Placeholders`, and 2 `Templates`. Assert the parser returns exactly 10 VMs.
* **Score Vectors:** Inject mocked Node states into the scoring function. Assert that a node with lopsided remaining CPU/memory generates a high `stranded_penalty` (fragmentation_penalty).

### 3.2 Integration Tests (The Placement Engine)
* **Inventory Priority:** Provide 5 inventory nodes and a catalog. Pass 5 tiny VMs. Assert that the `placement_engine` outputs 0 catalog nodes, proving it respects the `Cost = 0` inventory boundaries.
* **The "Monster VM" Exception:** Pass a VM that requests 2TB of RAM into a catalog where the maximum node is 1TB. Assert the system gracefully skips the VM, adds it to an `Unplaced` list, issues a warning, and finishes the rest of the loop without crashing.

### 3.3 E2E and Performance Load Testing
* **The CLI Smoke Test:** Execute the Typer CLI entry point using mock files. Assert the exit code is `0` and the `.csv` file is successfully written to disk.
* **Algorithmic Scaling:** Generate a synthetic workload of 5,000 heterogeneous VMs. The placement engine must complete the entire Filter/Score/Lookahead/Bind simulation in `< 5.0 seconds` to ensure the tool remains highly responsive for field engineers.

------------------------------------------------------------------------

## 4. CLI UX DESIGN

### Command

```bash
virtpack plan \
  --rvtools rvtools.xlsx \
  --config config.yaml \
  --catalog catalog.yaml \
  --inventory inventory.yaml \
  --output ./out \
  --debug
```

### Flags
| Flag                 | Description                                      |
|----------------------|--------------------------------------------------|
| --rvtools           | Input Excel (vInfo and vHost sheets)            |
| --config            | Global limits, weights, and safety margins      |
| --catalog           | Greenfield hardware catalog                     |
| --inventory         | Existing brownfield nodes                       |
| --output            | Directory for placement_map.csv                 |
| --strategy          | `spread` (default) or `consolidate` (HLD §1.1) |
| --debug             | Verbose placement logs & Fit Failure Reasons    |
| --no-auto-discovery | Disable vHost inventory auto-parsing            |


### CLI Output (Example)
```yaml
=========================================
Cluster Plan Summary
=========================================
Nodes Required: 14 (12 existing + 2 new)
Required OCP Subscriptions: 7

Peak CPU Utilization: 68%
Peak Memory Utilization: 78%

Cluster Bottleneck: MEMORY

Remaining Capacity:
  CPU: 32%
  Memory: 12%

Unplaced VMs: 2
=========================================
```

### Debug Mode Example
```yaml
[DEBUG] VM db01:
  node-01 rejected: insufficient memory
  node-02 rejected: pod limit exceeded

  node-03 score: 0.82
    balance: 0.91
    spread: 0.70
    stranded: 0.02

  selected: node-03
```