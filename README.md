# OCP Virt Capacity Planner

## Input
* **RVTools file**: a file `.xlsx` or `.csv` that contains `vInfo` sheet with:
    - `VM`
    - `CPUs`
    - `Memory`
    - `Powerstate`
* **config.yaml**: config information about OCP Virt cluster
* **inventory.yaml**: the servers the customer already owns and wants to reuse.
* **catalog.yaml**: the hardware the algorithm is allowed to "buy" if the inventory fills up.


