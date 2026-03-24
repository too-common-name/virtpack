# virtpack: OpenShift Virtualization Capacity Planner CLI - HLD

## 1. Purpose & Scope

This tool provides **capacity planning for OpenShift Virtualization clusters** by simulating placement of Virtual Machines onto physical bare-metal nodes.

Traditional sizing approaches rely on aggregated formulas:

    Total RAM / Node RAM
    Total vCPU / Node CPU

This ignores resource fragmentation and workload distribution, leading to clusters that appear valid on paper but fail during scheduling.

`virtpack` solves this problem by:
- parsing real workloads from RVTools exports
- normalizing resource usage with Red Hat overhead math
- applying configurable safety margins
- simulating placement using a **multidimensional bin-packing heuristic**
- generating auditable placement reports

### 1.1 Target Use Cases

The tool is designed to address four distinct scenarios, controlled by the `--strategy` flag (default: `spread`):

#### Scenario A1 — Pure Brownfield / Spread (default)
> *"I have 15 VMware hosts. Can I reuse them ALL for OpenShift Virtualization?"*

- **Input:** RVTools export + no catalog · `--strategy spread`
- **Nodes:** Existing hardware only (inventory, cost = 0)
- **Goal:** Prove all VMs fit on the existing infrastructure; show balanced utilization
- **Key concern:** Empty nodes = wasted OCP subscriptions; the engine must use ALL available nodes
- **Strategy:** All inventory nodes are added to `ClusterState` upfront. The scorer distributes VMs evenly across the full fleet using the `spread_score` signal.

#### Scenario A2 — Pure Brownfield / Consolidate
> *"I have 15 VMware hosts, but I want to minimize OCP subscriptions. Which nodes can I shut down?"*

- **Input:** RVTools export + no catalog · `--strategy consolidate`
- **Nodes:** Existing hardware only (inventory, cost = 0), pulled lazily from a pool
- **Goal:** Pack VMs onto the fewest inventory nodes; identify nodes that can be powered off
- **Key concern:** OCP subscriptions are per-node — unused nodes should be shut down
- **Strategy:** Inventory nodes are held in a **pool** (not added to `ClusterState` upfront). When no active node can fit the next VM, the engine pulls the largest available node from the pool. This mirrors the Expander's lazy expansion pattern but at zero cost. The summary reports "Nodes Available for Shutdown" and subscription savings.

#### Scenario B — Pure Greenfield
> *"I'm buying new hardware. What's the minimum set?"*

- **Input:** RVTools export + hardware catalog
- **Nodes:** New purchases only (catalog, cost > 0)
- **Goal:** Minimize the number (and cost) of new nodes purchased
- **Key concern:** Each additional node is an additional subscription and hardware cost
- **Strategy:** The Expander creates nodes on-demand; a new node is only added when no existing node can fit the next VM. Consolidation is **structural**, built into the expansion logic — the scorer does not need a MostAllocated signal.

#### Scenario C — Hybrid
> *"I have some old servers, but I may need to buy more."*

- **Input:** RVTools export + inventory + catalog
- **Nodes:** Existing hardware filled first, then catalog expansion on demand
- **Goal:** Maximize utilization of existing hardware; minimize new purchases
- **Strategy:** Spread across inventory (free), expand with catalog only when inventory is exhausted

### 1.2 Offline Capacity Planning vs Live Scheduling

`virtpack` is an **offline sizing tool**, not a runtime scheduler. This distinction drives fundamental design decisions:

| | K8s Scheduler | virtpack |
|---|---|---|
| **Input** | One pod at a time (online) | All VMs at once (offline) |
| **Goal** | Place this pod NOW | Prove a valid placement EXISTS |
| **Fragmentation** | Real concern (pods come and go) | Static allocation (no churn) |
| **Node scaling** | Dynamic (autoscaler) | Fixed set or known catalog |
| **Output** | Runtime binding | Mathematical audit trail (§8.5) |

The placement map CSV is generated solely as a **proof of feasibility**. The real Kubernetes scheduler will drift from this placement on Day 2 (see §9).

### 1.3 Theoretical Foundation

The placement problem is an instance of the **Multidimensional Vector Bin Packing Problem (VBPP)**:

> Each VM is a *d*-dimensional item vector `(cpu, memory, pods)` and each node is a bin with capacity vector `(CPU, MEM, PODS)`. The objective is to assign all items to the minimum number of bins such that no capacity constraint is violated.

This problem is **NP-hard** (Garey & Johnson, 1979). `virtpack` uses a **First-Fit-Decreasing (FFD)** heuristic — sorting items by their largest dimension and greedily assigning each to the best-scoring bin — which provides well-studied approximation guarantees for VBPP (Speitkamp & Bichler, 2010).

------------------------------------------------------------------------

## 2. High-Level Architecture Flow

The tool operates on a standard ETL (Extract, Transform, Load) and simulation pipeline:

1. Extract
   - Parse RVTools (vInfo, optional vHost)
   - Load YAML configs (config, inventory, catalog)
   - Filter invalid VMs:
     - poweredOff
     - SRM placeholders
     - templates

2. Transform
   - Normalize VM resources (CPU overcommit)
   - Normalize Node capacity (subtract system overheads)
   - Apply pod limits and reserved pods

3. Safety Margins
   - Apply utilization targets to derive effective schedulable capacity

4. Sort
   - Sort VMs by memory (descending) — FFD heuristic

5. Placement Loop
   For each VM:
     a. Filter candidate nodes
     b. Expand (create new node if needed)
     c. Score nodes (weighted scoring + lookahead k=2)
     d. Bind VM to best node

6. HA Injection
   - Add spare capacity to tolerate N node failures

7. Reporting
   - CLI summary
   - placement_map.csv
   - engineering metrics

------------------------------------------------------------------------

## 3. Input Specifications

The system requires four decoupled input sources.

### 3.1 Workload (RVTools)
Parsed from the `vInfo` sheet.
* **Required columns:** `VM`, `CPUs`, `Memory`, `Powerstate`, `SRM Placeholder`, `Template`
* **Filtering Rules:** Ingest only if `Powerstate == poweredOn` AND `SRM Placeholder == false` AND `Template == false`.

### 3.2 Global Configuration (config.yaml)
Contains cluster-wide limits, overcommit ratios, Red Hat baseline overheads, and the tunable algorithm weights.

    cluster_limits:
      max_pods_per_node: 250

    overcommit:
      cpu_ratio: 8.0
      memory_ratio: 1.0

    virt_overheads:
      ht_efficiency_factor: 1.5   # HT threads ≠ full cores (Red Hat sizing guide)
      ocp_virt_core: 2.0
      ocp_virt_memory_mb: 360
      eviction_hard_mb: 100

    safety_margins:
      utilization_targets:
        cpu: 85
        memory: 80
      ha_failures_to_tolerate: 1
      
    algorithm_weights:
      alpha_balance: 0.3        # Favors nodes with proportional CPU/RAM usage
      beta_spread: 0.3          # Favors the emptiest nodes (LeastAllocated)
      gamma_pod_headroom: 0.1   # Favors nodes with available pod IP space
      delta_frag_penalty: 0.3   # Penalizes dimensional imbalance in remaining capacity

### 3.3 Inventory (inventory.yaml)
Represents **existing brownfield hardware**. 
Cost = 0. The algorithm strictly attempts to fill these before expanding.

    profiles:
      - profile_name: r740-existing
        cpu_topology: { sockets: 2, cores_per_socket: 24, threads_per_core: 1 }
        ram_gb: 512
        quantity: 12

#### 3.3.1 RVTools Host Auto-Discovery (Host Recycling)
For "Lift and Shift" migrations where the customer is reusing their existing hypervisor hardware, `virtpack` features an auto-discovery mechanism.
1. The parser reads the `vHost` sheet from the `rvtools.xlsx` file.
2. It extracts the physical host specifications: `Host` (Name), `Num CPU` (Cores), `HT Active` (Hyperthreading), and `Memory` (GB).
3. It automatically injects these servers into Phase 1 of the placement algorithm as `Cost = 0` inventory nodes, eliminating the need for the user to manually write an `inventory.yaml` file.

### 3.4 Catalog (catalog.yaml)
"Greenfield" hardware profiles allowed for purchase.

    profiles:
      - profile_name: r760-new
        cpu_topology: { sockets: 2, cores_per_socket: 32, threads_per_core: 2 }
        ram_gb: 1024
        cost_weight: 1.0

### 3.5 Tuning the Algorithm Weights

The `algorithm_weights` are hyperparameters representing the operational philosophy of the cluster. Each term in the scoring function (§6.1) provides a unique, non-redundant signal:

| Weight | Term | Signal |
|--------|------|--------|
| α `alpha_balance` | Balance | CPU/memory utilization should be proportional |
| β `beta_spread` | Spread (LeastAllocated) | Distribute VMs across available nodes |
| γ `gamma_pod_headroom` | Pod headroom | Don't exhaust pod/IP slots |
| δ `delta_frag_penalty` | Stranded capacity | Don't strand remaining capacity in one dimension |

**Tuning Profiles:**

* **Spread-Optimized (Brownfield Default):** High β, moderate δ. Spreads VMs across all paid-for nodes while avoiding stranded capacity. This is the recommended default for brownfield migrations.
* **Balance-Optimized:** High α, high δ. Keeps nodes tightly balanced across CPU/memory and minimizes wasted capacity from dimensional imbalance.
* **Pod-Sensitive:** Increase γ when running many small VMs that risk hitting the max-pods-per-node limit.

> **Note on consolidation:** For greenfield scenarios (B) and brownfield consolidation (A2), consolidation is handled **structurally** — the Expander and the inventory pool add new nodes only when no active node can fit the next VM. The scorer does not need a MostAllocated signal because node introduction is inherently lazy.

------------------------------------------------------------------------

## 4. Resource Normalization

Before placement, resources must be normalized for both the VMs (applying overcommit) and the physical nodes (subtracting system overheads).

#### VM Normalization
VM CPU requests are normalized using the configured CPU overcommit ratio:

    effective_cpu = vm_cpu / cpu_ratio

#### Node Normalization
True usable node capacity is calculated in two stages: first the effective CPU count is derived (accounting for hyperthreading efficiency), then system overheads are subtracted.

**Stage 1 — Effective CPU (Hyperthreading Adjustment)**

Hyperthreaded cores do not deliver 2× the throughput of a physical core. Per the Red Hat OpenShift Virtualization Cluster Sizing Guide, a configurable efficiency factor (default 1.5×) is applied:

    if threads_per_core > 1:
        effective_cpu = physical_cores × ht_efficiency_factor
    else:
        effective_cpu = physical_cores

Where `physical_cores = sockets × cores_per_socket`.

| `ht_efficiency_factor` | Meaning                          | Example (2s × 32c × 2t) |
|------------------------|----------------------------------|--------------------------|
| 2.0                    | Count every thread as full core  | 128 effective CPUs       |
| 1.5 (default)          | Red Hat sizing guide recommended | 96 effective CPUs        |
| 1.0                    | Ignore HT (physical cores only)  | 64 effective CPUs        |

**Stage 2 — Overhead Subtraction**

    usable_cpu =
        effective_cpu
      − kubelet_reserved_cpu
      − ocp_virt_cpu

    usable_memory =
        node_memory
      − kubelet_reserved_memory
      − eviction_hard
      − ocp_virt_memory
  
Overheads originate from:
- [OpenShift Virtualization Cluster Sizing Guide](https://access.redhat.com/sites/default/files/attachments/openshift_virtualization_cluster_sizing_guide.pdf)
- [Kubelet Auto Sizing Rules (Machine Config Operator)](https://github.com/openshift/machine-config-operator/blob/release-4.17/templates/common/_base/files/kubelet-auto-sizing.yaml)
 
### 4.1 Formal Resource Model

The placement problem can be modeled as a multidimensional vector bin packing problem (VBPP).

Let:
```
VM_i = (cpu_i, mem_i, pods_i)
Node_j = (CPU_j, MEM_j, PODS_j)

Constraints:
    Σ cpu_i ≤ CPU_j  
    Σ mem_i ≤ MEM_j  
    Σ pods_i ≤ PODS_j  
    for all VMs assigned to Node_j.
```

The objective is to place all VMs using the minimum number of nodes while ensuring balanced distribution. This problem is NP-hard (it generalizes the classical bin packing problem), therefore `virtpack` relies on FFD-style heuristics rather than exact optimization.

------------------------------------------------------------------------

## 5. Safety Margins

To guarantee K8s scheduling headroom, the engine shrinks usable capacity before placement begins:

    target_capacity = usable_capacity × target_utilization

If RAM target is 80%, a 400GB node will reject VMs once 320GB is consumed.

------------------------------------------------------------------------

## 6. The Placement Engine

The placement engine simulates VM-to-node assignment using a weighted scoring heuristic inspired by Kubernetes scheduling, adapted for offline capacity planning.

#### 6.1 The Scoring Model

Each candidate node receives a weighted score. Higher is better.

    score(node) = 
        (α * balance_score) 
      + (β * spread_score) 
      + (γ * pod_headroom)
      - (δ * stranded_penalty)

Each component ∈ [0, 1]:

* **CPU/Memory Balance (α):** `1 - abs(cpu_util - mem_util)`
  *Encourages balanced nodes to prevent exhausting one resource while the other sits idle. Mimics K8s `NodeResourcesBalancedAllocation`.*

* **Spread Score (β):** `((1 - cpu_util) + (1 - mem_util)) / 2`
  *Favors nodes with the most free resources. Mimics K8s `LeastAllocated`. This is the primary signal for distributing VMs across available hardware.*

* **Pod Headroom (γ):** `1 - (pods_used / max_pods)`
  *Discourages pod exhaustion to preserve IP space and scheduling limits.*

* **Stranded Capacity Penalty (δ):** `(cpu_remaining% - memory_remaining%)²`
  *Penalizes nodes where remaining CPU and memory are disproportionate. When one dimension has ample remaining capacity but the other is nearly exhausted, the excess dimension is "stranded" — it cannot be consumed by future VMs. This is the genuine fragmentation signal for capacity planning: unlike online scheduling where pods arrive and leave randomly, in offline VBPP all items are known upfront, so the penalty focuses on dimensional imbalance rather than memory gaps.*

  | Scenario | CPU rem% | Mem rem% | Penalty | Interpretation |
  |----------|----------|----------|---------|----------------|
  | Empty node | 100% | 100% | 0.00 | No stranding |
  | Balanced 50% | 50% | 50% | 0.00 | No stranding |
  | CPU-bound | 10% | 70% | 0.36 | 70% memory stranded |
  | Memory-bound | 60% | 5% | 0.30 | 60% CPU stranded |
  | Fully used | 0% | 0% | 0.00 | Nothing remaining |


#### 6.2 The Placement Algorithm (With Lookahead k=2)

    Sort VMs by memory desc          // FFD heuristic

    For index, vm in enumerate(vms):
        // 0. HARD CONSTRAINT (Monster VM Check)
        if vm_memory > max_catalog_node_memory:
            add to unplaced_list
            continue

        // 1. FILTER: Find candidate_nodes that fit CPU/RAM/PODs + Safety Margins
        
        // 2. EXPAND: If candidate_nodes is empty -> create_catalog_node()
            
        // 3. SCORE: Projected State with Lookahead
        //    Scoring evaluates the **projected** state (after tentatively
        //    placing the current VM), not the node's current state.
        //    This mirrors K8s scoring (NodeResourcesBalancedAllocation,
        //    LeastAllocated) and enables stranded nodes to "attract" VMs
        //    whose resource profile reduces dimensional imbalance.
        best_node = None
        best_total_score = -infinity  // Higher score is better
        
        for node in candidate_nodes:
            simulate_place(node, vm)
            base_score = calculate_score(node)  // scored AFTER placing vm
            
            // Lookahead: score with both vm AND next_vm placed
            lookahead_score = 0
            if (index + 1) < len(vms):
                next_vm = vms[index + 1]
                
                if node_fits(node, next_vm):
                    simulate_place(node, next_vm)
                    lookahead_score = calculate_score(node)
                    undo_simulate_place(node, next_vm)
                else:
                    lookahead_score = massive_negative_penalty
            
            undo_simulate_place(node, vm)
            
            total_score = base_score + (0.5 * lookahead_score)
            
            if total_score > best_total_score:
                best_total_score = total_score
                best_node = node
                
        // 4. BIND: place(vm, best_node)

> **Design rationale — projected scoring:** Pre-placement scoring evaluates a node's *current* state, so a stranded node always receives the same penalty regardless of which VM is being considered. Projected scoring evaluates the node *after* the VM, so a CPU-heavy VM makes a CPU-stranded node more attractive (it reduces the dimensional gap), while a memory-heavy VM makes it less attractive (it widens the gap). This enables automatic rebalancing of stranded nodes without requiring manual weight tuning.

#### 6.3 Unplaced VM Handling
If a VM cannot fit on any inventory or catalog profile, the VM is added to the **Unplaced list**. The planner continues processing the remaining workload while emitting a clear warning to the user. 
* **Fit Failure Reason:** When running in `--debug` mode, the tool explicitly logs why nodes were bypassed (e.g., `Node node-04 rejected: insufficient memory`). This provides total transparency into the placement engine's decisions.

------------------------------------------------------------------------

## 7. HA Node Injection (Expansion)

After placement, the system injects `N` failures to tolerate. 
Because CPU and memory are independent dimensions, worst-case failure must be evaluated separately:

    largest_memory_node = argmax(node.memory_used)
    largest_cpu_node = argmax(node.cpu_used)

Then compute:
-  required_spare_cpu
-  required_spare_memory

The cluster must maintain spare capacity: 
- `≥ largest_memory_node.memory_used`
- `≥ largest_cpu_node.cpu_used`

This guarantees recovery of the heaviest resource consumers. Other nodes may also host workloads, but worst-case recovery must always be guaranteed. The engine selects the most cost-efficient catalog nodes capable of satisfying the required spare capacity.

------------------------------------------------------------------------

## 8. Reporting & Outputs

The CLI provides comprehensive outputs for executives and engineers.

#### 8.1 CLI Summary (Terminal)
The terminal output prioritizes human readability and immediate operational insight:

    =========================================
    Cluster Plan Summary
    =========================================
    Nodes Required: 14 (12 existing + 2 new)
    Required OCP Bare Metal Subscriptions: 7

    Peak CPU Utilization: 68%
    Peak Memory Utilization: 78%
    Unplaced VMs: 2
    
    Cluster Bottleneck: MEMORY
    
    Cluster Remaining Capacity (Headroom):
      CPU: 32%
      Memory: 12%
    =========================================

#### 8.2 Cluster Fragmentation Index (CFI)

The CFI quantifies how much remaining cluster capacity is dimensionally stranded — i.e., remaining CPU and memory are disproportionate, making the excess in one dimension unusable.

    CFI = average(stranded_penalty(nodes))

where `stranded_penalty(node) = (cpu_remaining% − memory_remaining%)²`.

*Example:*
* Cluster A: CFI = 0.02 — remaining capacity is well-balanced across dimensions
* Cluster B: CFI = 0.25 — significant stranded capacity; consider different node profiles

Lower CFI = remaining capacity is more evenly distributed across dimensions = better.

#### 8.3 Node Pressure Index
Node pressure measures the highest resource utilization across nodes.

    pressure(node) = max(cpu_util, memory_util)

Reported metrics:
* Node Pressure P95
* Node Pressure Max

*Output Example:*
    Node Pressure P95: 0.81 | Node Pressure Max: 0.93

This tells engineers whether some specific nodes are dangerously full despite aggregate cluster averages.

#### 8.5 Simulated Placement Audit Log (`placement_map.csv`)
    VM_Name,vCPU,RAM,Target_Node,Node_Profile
    db01,8,32768,node-01,r760-new

*Architectural Note:* The real Kubernetes scheduler will drift from this exact placement map on Day 2 due to dynamic pod creation. This CSV is generated solely as a **Mathematical Audit Trail**. It exists to prove to architects and stakeholders that a valid, geometrically sound placement solution exists for the workload on the recommended hardware.

------------------------------------------------------------------------

## 9. Day-2 Operations & Drift Mitigation

Because `virtpack` performs "Offline Bin Packing" (evaluating all 1,000 VMs at once to find the mathematically perfect fit) and Kubernetes performs "Online Bin Packing" (placing VMs one by one as they boot in a random order), the real cluster will naturally drift from the simulated `placement_map.csv`. 

To address this operational reality, `virtpack` sizing pairs with standard Day-2 OpenShift operations:

1. **The Safety Buffer:** The `utilization_targets` (e.g., 85%) explicitly reserve 15% of cluster capacity to absorb the sub-optimal fragmentation caused by the random boot order of VMs.
2. **The OpenShift Descheduler:** Over time, as VMs scale, reboot, or migrate, the cluster becomes fragmented. Customers should deploy the OpenShift Descheduler Operator, which periodically evaluates the cluster and evicts VMs from sub-optimal nodes to consolidate space.
3. **Live Migration Defragmentation:** Because OpenShift Virtualization supports Live Migration, the Descheduler (or platform admins) can non-disruptively move VMs between nodes, constantly realigning the Day-2 cluster state back toward the Day-0 mathematical ideal calculated by `virtpack`.

------------------------------------------------------------------------

## 10. References

1. Speitkamp, B. & Bichler, M. (2010). *"A Mathematical Programming Approach for Server Consolidation Problems in Virtualized Data Centers."* IEEE Transactions on Services Computing, 3(4), 266–278. [PDF](https://pub.dss.in.tum.de/bichler-research/2006_bichler_capacity_planning.pdf)
2. Garey, M. R. & Johnson, D. S. (1979). *Computers and Intractability: A Guide to the Theory of NP-Completeness.* W. H. Freeman.
3. Red Hat (2024). *OpenShift Virtualization Cluster Sizing Guide.* [Link](https://access.redhat.com/sites/default/files/attachments/openshift_virtualization_cluster_sizing_guide.pdf)
4. OpenShift Machine Config Operator — kubelet-auto-sizing.yaml. [Source](https://github.com/openshift/machine-config-operator/blob/release-4.17/templates/common/_base/files/kubelet-auto-sizing.yaml)

------------------------------------------------------------------------

## 11. Future Improvements

Future improvements may extend scheduling realism. Possible enhancements include:

* **NUMA-aware packing:** Prevent large VMs from spanning multiple sockets.
* **Affinity and anti-affinity rules:** Prevent specific workloads from sharing nodes.
