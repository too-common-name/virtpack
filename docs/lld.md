# virtpack: OpenShift Virtualization Capacity Planner CLI - LLD

## 1. Code Architecture
```bash
virtpack/
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
    └── io/
        ├── rvtools_parser.py
        └── yaml_loader.py
```
------------------------------------------------------------------------
## 2. Key Classes & Interfaces

### 2.1 Domain Models
Using `pydantic` or Python `dataclasses` for strict typing.

```
    class VM:
        name: str
        cpu: float
        memory: float
        pods: int = 1

    class Node:
        id: str
        profile: str
        
        # Resource tracking
        cpu_total: float
        cpu_used: float
        memory_total: float
        memory_used: float
        pods_total: int
        pods_used: int
        
        # Placement metadata
        cost_weight: float
        is_inventory: bool
```
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

    def score(node, vm) -> float

    def balance(node)
    def spread(node)
    def fragmentation(node)

class Expander:

    def select_profile(catalog)

    def create_node(profile)

```

![Sequence Diagram](http://www.plantuml.com/plantuml/proxy?src=https://raw.githubusercontent.com/too-common-name/virtpack/main/doc/uml/sequence_placement.puml)
------------------------------------------------------------------------

## 3. Exhaustive Testing Strategy

To ensure enterprise-grade reliability, the codebase must pass a rigorous multi-tiered `pytest` CI/CD pipeline.

### 3.1 Unit Tests (Math & Transformation)
* **MCO Overheads:** Hardcode 5 physical node sizes (e.g., 64GB, 256GB, 1TB). Assert that the `normalizer.py` outputs the exact usable capacity down to the Megabyte as specified by the Red Hat `kubelet-auto-sizing.yaml` math.
* **ETL Filtering:** Feed a mocked RVTools DataFrame containing 10 `poweredOn` VMs, 5 `poweredOff`, 3 `SRM Placeholders`, and 2 `Templates`. Assert the parser returns exactly 10 VMs.
* **Score Vectors:** Inject mocked Node states into the scoring function. Assert that a node with highly lopsided memory utilization generates a massive `fragmentation_penalty`.

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
    fragmentation: 0.10

  selected: node-03
```