# DVRP-GNN-RL: Dynamic Vehicle Routing with Graph Neural Networks and Reinforcement Learning

A learning-based approach to the **Dynamic Capacitated Vehicle Routing Problem (DVRP)**, where customer requests arrive throughout the day and routing decisions must be made without knowledge of future requests.

This project combines a **GATv2 graph encoder** with a pointer-based policy trained using **REINFORCE**. The model operates in a simulated environment, observes only revealed requests, and selects the next routing action as new information becomes available.

**Project:** End-of-year engineering project at INSEA, Rabat  
**Author:** Mohamed Kachchar  
**Supervisor:** Houda Lahrch  
**Report:** `rapport_dynamique.tex` (French)

---

## Table of Contents

- [Problem Definition](#problem-definition)
- [Scope and Assumptions](#scope-and-assumptions)
- [Methodology](#methodology)
- [Handling Hidden Information](#handling-hidden-information)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Reproducing Experiments](#reproducing-experiments)
- [Experimental Setup](#experimental-setup)
- [Results](#results)
- [Validation and Tests](#validation-and-tests)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [References](#references)

## Problem Definition

In the classical Vehicle Routing Problem (VRP), all customer requests are known before the vehicle begins its route. In a dynamic setting, some requests become available only after the planning horizon has started.

This project studies a single-vehicle routing problem with multiple trips, limited vehicle capacity, and customer requests arriving throughout an eight-hour planning horizon.

The **degree of dynamism (DoD)** controls the proportion of dynamically arriving customers. Each dynamic request has a release time \(r_i\), before which it is unavailable to the routing policy.

The objective is to learn a routing policy that makes efficient decisions using only the information available at each decision point.

## Scope and Assumptions

The current implementation focuses on the following problem setting:

| Feature | Current implementation |
|---|---|
| Vehicles | One vehicle |
| Trips | Multiple trips allowed |
| Vehicle capacity | Limited |
| Customer arrivals | Dynamic, generated using a Poisson process |
| Planning horizon | 480 minutes |
| Degree of dynamism | Configurable from 0 to 1 |
| Waiting | Waiting at the depot is supported |
| Locations | Two-dimensional Euclidean coordinates |
| Time windows | Not included in the current problem formulation |
| Real road networks | Not included |

The simulator can access the complete generated instance, but the policy receives only the currently available information. This separation is central to the problem formulation.

## Methodology

The solution combines graph representation learning with policy-gradient reinforcement learning.

### 1. Graph representation

The routing environment is represented as a graph in which nodes correspond to the depot and customers. Node features describe their locations and relevant operational attributes, while edge attributes encode distances.

The graph encoder uses **Graph Attention Network v2 (GATv2)** layers to produce node embeddings that capture relationships between locations.

### 2. Pointer-based policy

A pointer decoder uses the graph embeddings and the current routing state to select the next node. Its context includes the mean graph embedding, the current node embedding, remaining capacity, elapsed time, and the fraction of visible customers already served.

A feasibility mask prevents the policy from selecting invalid actions.

### 3. Reinforcement learning

The policy is trained using **REINFORCE**, a Monte Carlo policy-gradient algorithm. Training uses a frozen greedy policy as a baseline to reduce the variance of policy-gradient estimates.

The baseline is updated only when the current policy demonstrates a statistically significant improvement under a paired test.

### Decision pipeline

```text
Dynamic instance
      |
      v
DVRP simulator
      |
      v
Current observation
(Revealed requests only)
      |
      v
GATv2 graph encoder
(3 layers, 8 heads, dimension 128)
      |
      v
Pointer decoder
+ routing state
+ feasibility mask
      |
      v
Next action
(Customer or depot)
      |
      v
Environment transition
      |
      +---- New request arrives
      |           |
      +-----------+----> Update observation and re-encode
      |
      v
Episode completion and reward
      |
      v
REINFORCE policy update
```

### Model and training configuration

| Component | Configuration |
|---|---|
| Graph encoder | GATv2 |
| Encoder depth | 3 layers |
| Attention heads | 8 |
| Embedding dimension | 128 |
| Edge attributes | Euclidean distance |
| Decoder | Pointer-based attention |
| Attention clipping | \(C=10\) |
| Optimizer | Adam |
| Learning rate | \(10^{-4}\) |
| Gradient clipping | 1.0 |
| Entropy bonus coefficient | 0.01 |
| Training algorithm | REINFORCE |
| Baseline | Frozen greedy policy |

The training reward is based on negative episode completion time (makespan). The experiments also report total travel distance as a performance metric. Under the main no-congestion setting, travel time and distance are closely related, subject to the environment's service-time and waiting conventions.

## Handling Hidden Information

A central requirement is that the online policy must not use information about requests that have not yet appeared.

The simulator maintains the complete instance internally, while the policy receives its observation through `env.observe()`.

The implementation addresses information leakage in several ways:

- **Request visibility:** unreleased customers are excluded from the policy's observation.
- **Graph construction:** hidden customers are excluded from the visible graph and its neighbourhood information.
- **Graph summaries:** aggregate features and counters are calculated using the visible set rather than the full instance.
- **Event-driven re-encoding:** the graph is re-encoded when a new request becomes visible, only for batch elements whose visible sets have changed.
- **Automated validation:** tests check that instances differing only in hidden future requests produce identical observations and decisions under the same conditions.

An earlier implementation exposed information through a served-customer ratio calculated using the total number of requests, including hidden ones. A dedicated leakage probe identified the issue, and the implementation was corrected.

See [Validation and Tests](#validation-and-tests) for the associated checks.

## Repository Structure

```text
.
├── config.py
├── data/
│   └── generator.py
├── rl/
│   ├── env.py
│   ├── gnn_encoder.py
│   ├── policy.py
│   ├── train.py
│   ├── baselines.py
│   ├── reopt.py
│   └── exact_solver.py
├── scripts/
│   ├── 17_dynamic_campaign.py
│   ├── 18_analyse_dynamic.py
│   ├── 19_leak_probe.py
│   ├── 14_exact_cpsat.py
│   └── 15_latency_single.py
├── tests/
├── viz/
├── outputs/
│   ├── dynamic/
│   └── supp/
├── requirements.txt
└── README.md
```

The main modules are:

- `config.py`: configuration for the instances, model, and training process.
- `data/generator.py`: reproducible instance generation.
- `rl/env.py`: dynamic simulator, observations, transitions, rewards, and metrics.
- `rl/gnn_encoder.py`: GATv2 graph encoder.
- `rl/policy.py`: policy encoding, action selection, and episode rollout.
- `rl/train.py`: REINFORCE training, baseline management, and checkpoints.
- `rl/baselines.py`: random and nearest-neighbour routing policies.
- `rl/reopt.py`: repeated static optimization using OR-Tools.
- `rl/exact_solver.py`: exact static CVRP solving using CP-SAT and dynamic programming.
- `scripts/`: experiment execution, statistical analysis, leakage checks, and latency measurements.
- `tests/`: automated checks for the simulator and information constraints.
- `viz/`: visualization utilities for instances and routing solutions.
- `outputs/`: experiment results, analysis files, and available checkpoints.

This tree describes the documented project layout; confirm that the files and output folders are present in the public repository before publishing.

## Installation

### Requirements

- Python 3.13, as reported for the tested environment
- PyTorch
- PyTorch Geometric
- Gymnasium
- NumPy and the other packages listed in `requirements.txt`
- OR-Tools for re-optimization and exact-solver experiments
- pytest for running tests

A GPU is optional. CPU execution is supported, although training takes longer.

### 1. Clone the repository

Replace the URL below with the URL of your GitHub repository.

```bash
git clone <your-repo-url>
cd <your-repo-directory>
```

### 2. Create a virtual environment

**Windows CMD:**

```cmd
python -m venv venv
venv\Scripts\activate
```

**Linux or macOS:**

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install ortools pytest
```

If your `requirements.txt` already includes OR-Tools and pytest, the final installation command is unnecessary.

## Quick Start

The following example illustrates the intended inference workflow using a trained checkpoint.

```python
from data.generator import generate_dataset
from rl.env import DVRPEnv
from rl.train import load_checkpoint

# Load a trained policy
policy = load_checkpoint("outputs/dynamic/online_s0.pt")
policy.eval()

# Generate evaluation instances
instances = generate_dataset(
    8,
    base_seed=20_000_000,
    n_customers=20,
    degree_of_dynamism=0.5,
)

# Create the online environment
env = DVRPEnv(
    instances,
    information="online",
    congestion_mode="off",
    objective="time",
)

# Evaluate using greedy action selection
_, reward, _, metrics = policy.rollout(env, greedy=True)

print("Mean travel distance:", metrics["distance"].mean())
```

**Note:** This example assumes that the specified checkpoint exists and that the API matches the current version of the code. The repository must include a compatible checkpoint, or the model must be trained first. The precise return values of `rollout()` should be checked against the implemented API.

## Reproducing Experiments

Run the following commands from the repository root in an activated Python environment.

### 1. Run the automated tests

```bash
pytest -q tests
```

The documented test suite contains 37 tests.

### 2. Run the dynamic training campaign

```bash
EP=200 python scripts/17_dynamic_campaign.py
```

The reported campaign uses three random seeds and compares the online and clairvoyant information regimes.

The environment-variable syntax above is appropriate for Unix-like shells. In Windows CMD, use:

```cmd
set EP=200
python scripts/17_dynamic_campaign.py
```

The reported CPU runtime is approximately 35 minutes per campaign run, although actual execution time depends on the hardware and configuration.

### 3. Analyse the results

```bash
python scripts/18_analyse_dynamic.py
```

The analysis produces statistical summaries in `outputs/dynamic/analyse.json`.

### 4. Probe for information leakage

```bash
python scripts/19_leak_probe.py
```

This experiment checks whether hidden future requests affect policy observations or decisions.

### 5. Evaluate the static lower bound

```bash
python scripts/14_exact_cpsat.py
```

This experiment evaluates the exact static CVRP solution on small instances. It provides a lower-bound reference, not the optimal solution to the dynamic problem.

### 6. Measure inference latency

```bash
python scripts/15_latency_single.py
```

This script measures the policy's inference latency for a single instance.

**Additional experiment:** A longer training run using `EP=600` was identified as a useful next step but had not been completed at the time of this documentation.

## Experimental Setup

The main experimental campaign uses the following configuration:

| Parameter | Value |
|---|---|
| Customers | 20 |
| Degree of dynamism | 0.5 |
| Vehicle capacity | 30 |
| Planning horizon | 480 minutes |
| Congestion | Disabled |
| Training epochs | 200 |
| Batches per epoch | 10 |
| Instances per batch | 64 |
| Training seeds | 3 |
| Evaluation instances | 256 held-out instances |
| Action selection during evaluation | Greedy |
| Statistical evaluation | Paired tests |

Training, validation, and test seeds are disjoint.

### Baselines

The learned policy is evaluated against the following methods:

- **Random:** selects routing actions randomly among permitted choices.
- **Nearest neighbour (NN):** selects a nearby feasible customer.
- **REOPT (OR-Tools):** repeatedly solves a static routing problem over known, unserved requests when a new request arrives. Each solve uses a time limit of 0.2 seconds, after which the current plan is followed.
- **Exact static CVRP optimum:** computes an optimal solution for small static instances. Since this reference ignores request release times, it is a lower-bound reference rather than the true dynamic optimum.

### Clairvoyant ablation

A second learned policy uses the same architecture but receives all customer requests at the beginning of the planning horizon.

This **clairvoyant policy** serves as an ablation to investigate the effect of future information. It is not a fair online competitor because it has access to information unavailable to the online policy.

## Results

### Main comparison

The following results report mean travel distance over 256 test instances with 20 customers and DoD 0.5, using three training seeds.

| Method | Mean distance | Difference vs. NN | Difference vs. REOPT |
|---|---:|---:|---:|
| Random | 1173.8 | — | — |
| Nearest neighbour | 737.6 | Reference | — |
| REOPT (OR-Tools) | 632.5 | -14.2% | Reference |
| **RL online** | **651.2** | **-11.7% ± 0.9** | **+2.9%** |
| RL clairvoyant | 654.9 | -11.2% ± 0.5 | +3.5% |

Lower distance is better.

The online RL policy substantially improves on the nearest-neighbour heuristic, reducing mean distance by 11.7%. However, REOPT achieves a lower mean distance than the learned policy.

The clairvoyant ablation does not outperform the online policy under the current training budget. This result does not establish that future information is generally unhelpful; it shows only that the tested clairvoyant model did not obtain an advantage in this experimental setting.

### Effect of the degree of dynamism

The following table reports the online policy's relative distance gap to REOPT under different degrees of dynamism.

| DoD | Distance gap vs. REOPT |
|---:|---:|
| 0 | +16.1% |
| 0.25 | +7.4% |
| 0.5 | +2.9% |
| 0.75 | +1.9% |
| 1 | +4.4% |

A positive gap means that the RL policy has a higher mean distance than REOPT. The gap varies with the proportion of dynamically arriving requests, indicating that performance depends on the information regime.

### Inference latency

The measured inference times on a CPU using two threads are approximately:

| Number of customers | RL inference time |
|---:|---:|
| 10 | 15 ms |
| 20 | 31 ms |
| 50 | 75 ms |

At 20 customers, REOPT takes approximately 1.4 seconds under the reported configuration. This makes the learned policy approximately 40 times faster in the measured setting.

This is an **unequal-compute comparison**: REOPT receives a 0.2-second solve limit at each event, while the learned policy performs a forward pass. REOPT was not evaluated under a comparable millisecond-scale compute budget, so the latency results should not be interpreted as a comparison under equal computational resources.

### Interpretation

The experiments support the following conclusions:

- The learned online policy consistently outperforms the nearest-neighbour baseline across the three training seeds.
- REOPT remains more effective in terms of mean distance, outperforming the online policy by 2.9% in the main configuration.
- The gap between the learned policy and REOPT varies with the degree of dynamism.
- The tested clairvoyant policy does not improve on the online policy under the current training budget.
- At 50 customers, although the model was trained on 20 customers, the gap to REOPT increases to 7.6%.
- On small instances with 8–12 customers, all tested methods, including REOPT, are reported to be 22–37% above the exact static lower bound. This gap is not the distance to the true dynamic optimum because the lower bound ignores request release times.

Ablation experiments also report no measurable effect from removing distance edge attributes or changing the baseline configuration under the tested budget. These findings are specific to the experiments performed and do not establish that the components are generally ineffective.

## Validation and Tests

The documented automated test suite contains **37 tests**, divided between:

- `tests/test_dynamic.py`: 13 tests.
- `tests/test_invariants.py`: 24 tests.

The tests cover the following properties:

- Instances that differ only in the content and number of hidden future requests produce identical observations and decisions under the same conditions.
- Requests become visible at their release times.
- Customers cannot be served before they appear.
- Customers cannot be served more than once.
- Vehicle capacity and episode termination rules are respected.
- Waiting at the depot does not expose future information.
- Training and inference follow consistent information constraints.
- Policies trained under one information regime cannot be used in an incompatible regime.

Run the tests with:

```bash
pytest -q tests
```

The documented result is 37 passing tests. This count should be reconfirmed when reproducing the current repository version.

## Limitations

The current implementation and experiments have several limitations:

- **Training convergence:** learning curves have not stabilized after 200 epochs, partly because of the available CPU training budget.
- **Limited configurations:** the principal evaluation uses one main configuration and three training seeds.
- **Compute fairness:** the latency comparison against REOPT uses unequal computational budgets.
- **Congestion:** several congestion modes are implemented, but the main experimental campaign runs without congestion.
- **Model reproducibility:** the original GPU campaign used a clairvoyant regime and 300 epochs. Its results are archived in `outputs/final/`, but the original trained model files were lost, preventing direct re-evaluation.
- **Scalability:** performance deteriorates relative to REOPT when the model is evaluated on larger instances than those used during training.

These limitations motivate further experiments before drawing broader conclusions about generalization, scalability, or superiority over optimization-based methods.

## Future Work

Potential extensions include:

- Longer GPU training runs to investigate whether the gap to REOPT can be reduced.
- Comparisons under more closely matched computational budgets.
- A learned voluntary-waiting policy.
- Multiple vehicles.
- Time-window constraints.
- Sparse k-nearest-neighbour graphs for larger instances.
- Evaluation on public benchmarks.

These are proposed directions rather than completed features.

## References

1. Kool, W., van Hoof, H., and Welling, M. *Attention, Learn to Solve Routing Problems!* ICLR, 2019.
2. Brody, S., Alon, U., and Yahav, E. *How Attentive are Graph Attention Networks?* ICLR, 2022.
3. Williams, R. J. *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning.* Machine Learning, 1992.
4. Pillac, V., Gendreau, M., Guéret, C., and Medaglia, A. L. *A Review of Dynamic Vehicle Routing Problems.* European Journal of Operational Research, 2013.
5. Google. *OR-Tools: Optimization Tools for Combinatorial Problems.*

## License

A license has not yet been specified. Add a `LICENSE` file before distributing the project under an open-source license.

