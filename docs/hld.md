# virtpack: OpenShift Virtualization Capacity Planner CLI - HLD

## 1. Purpose & Scope

This tool provides **capacity planning for OpenShift Virtualization clusters** by simulating placement of Virtual Machines onto physical bare-metal nodes.

Traditional sizing approaches rely on aggregated formulas:

    Total RAM / Node RAM
    Total vCPU / Node CPU

This ignores resource fragmentation and workload distribution, leading to clusters that appear valid on paper but fail during scheduling.

`virtpack` solves this problem by:
- parsing real workloads
- normalizing resource usage
- applying safety margins
- simulating placement using a bin-packing heuristic
- generating auditable placement reports

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
   - Sort VMs by memory (descending)

5. Placement Loop
   For each VM:
     a. Filter candidate nodes
     b. Expand (create new node if needed)
     c. Score nodes (K8s-like scoring + lookahead k=2)
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
      beta_spread: 0.3          # Favors the emptiest nodes
      gamma_pod_headroom: 0.1   # Favors nodes with available pod IP space
      delta_frag_penalty: 0.3   # Punishes unusable memory gaps

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
The `algorithm_weights` are hyperparameters representing the operational philosophy of the cluster:
* **Cost-Optimized (Aggressive Bin-packing):** Increase `delta_frag_penalty` and decrease `beta_spread`. The algorithm will tightly pack nodes and ignore spreading rules to save money.
* **Resilience-Optimized (Default K8s):** Keep `alpha`, `beta`, and `delta` relatively equal. This mimics the default `kube-scheduler`, providing an accurate Day-2 representation.

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

The placement problem can be modeled as a multidimensional bin packing problem.

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

The objective is to minimize the number of nodes used while minimizing resource fragmentation. 
This problem is NP-hard, therefore `virtpack` relies on heuristic algorithms rather than exact optimization.

------------------------------------------------------------------------

## 5. Safety Margins

To guarantee K8s scheduling headroom, the engine shrinks usable capacity before placement begins:

    target_capacity = usable_capacity × target_utilization

If RAM target is 80%, a 400GB node will reject VMs once 320GB is consumed.

------------------------------------------------------------------------

## 6. The Placement Engine

The placement engine approximates Kubernetes scheduling behavior.
Real Kubernetes scheduling uses two phases:
1. **Filtering:** Nodes failing hard constraints are removed.
2. **Scoring:** Nodes are ranked based on `NodeResourcesBalancedAllocation`, `LeastAllocated`, and `PodTopologySpread`.

`virtpack` approximates these behaviors through a weighted scoring function combined with Lookahead.

#### 6.1 The Scoring Model (K8s Simulation)
We replicate a simplified score phase using the weights from `config.yaml`:

    score(node) = 
        (α * balance_score) 
      + (β * spread_score) 
      + (γ * pod_headroom)
      - (δ * fragmentation_penalty)

Where the metrics are mathematically calculated and explained as follows:
* **CPU/Memory Balance:** `1 - abs(cpu_util - mem_util)`
  *Encourages balanced nodes to prevent exhausting one resource while the other sits idle.*
* **Spread Score:** `((1 - cpu_util) + (1 - mem_util)) / 2`
  *Favors nodes with the most free resources, mimicking K8s `LeastAllocated`.*
* **Pod Headroom:** `1 - (pods_used / max_pods)`
  *Discourages pod exhaustion to preserve IP space and scheduling limits.*
* **Memory Fragmentation:** `(memory_remaining / node_memory)^2`
  *Fragmentation penalty increases when nodes contain small remaining memory segments that are unlikely to host future VMs.*

#### 6.2 The Placement Algorithm (With Lookahead k=2)

    Sort VMs by memory desc

    For index, vm in enumerate(vms):
        // 0. HARD CONSTRAINT (Monster VM Check)
        if vm_memory > max_catalog_node_memory:
            add to unplaced_list
            continue

        // 1. FILTER: Find candidate_nodes that fit CPU/RAM/PODs + Safety Margins
        
        // 2. EXPAND: If candidate_nodes is empty -> create_catalog_node()
            
        // 3. SCORE: With Lookahead
        best_node = None
        best_total_score = -infinity  // Higher score is better
        
        for node in candidate_nodes:        
            base_score = calculate_k8s_score(node)
            
            // Lookahead check for next VM
            lookahead_score = 0
            if (index + 1) < len(vms):
                next_vm = vms[index + 1]
                simulate_place(node, vm)
                
                if node_fits(node, next_vm):
                    lookahead_score = calculate_k8s_score(node)
                else:
                    lookahead_score = massive_negative_penalty
                
                undo_simulate_place(node, vm)
            
            total_score = base_score + (0.5 * lookahead_score)
            
            if total_score > best_total_score:
                best_total_score = total_score
                best_node = node
                
        // 4. BIND: place(vm, best_node)

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
    CFI = average(fragmentation_penalty(nodes))

*Example:*
* Cluster A: CFI = 0.12
* Cluster B: CFI = 0.42

Lower values = better packing.

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

## 10. Future Improvements

Future improvements may extend scheduling realism. Possible enhancements include:

* **NUMA-aware packing:** Prevent large VMs from spanning multiple sockets.
* **Affinity and anti-affinity rules:** Prevent specific workloads from sharing nodes.