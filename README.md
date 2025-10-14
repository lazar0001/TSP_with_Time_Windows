# Traveling Salesman Problem with Time Windows (TSPTW)

Project comparing exact and metaheuristic approaches for solving the Traveling Salesman Problem with Time Windows (TSPTW).
This project was developed as part of coursework for the Computational Intelligence course at the Faculty of Mathematics, University of Belgrade.

## Overview

This project implements and compares:
- **Brute Force (Branch-and-Bound)** - Exact solver for small instances
- **Simulated Annealing** - Metaheuristic for larger instances

## Project Structure

```
.
├── tsptw_common.py          # Shared data structures and utilities
├── tsptw_bruteforce.py      # Exact solver
├── tsptw_sa_v2.py           # Simulated Annealing solver
├── bench_runner.py          # Benchmarking framework
├── viz_suite.py             # Visualization tools
├── tsptw_edge_suite/        # Test instances (edge cases)
└── tsptw_SA_suite/          # Test instances (easy/medium/hard)
```

## Installation

```bash
pip install pandas matplotlib numpy
```

## Quick Start

### 1. Run Individual Solvers

**Brute Force:**
```bash
python tsptw_bruteforce.py tsptw_edge_suite/wide_overlapping.txt
```

**Simulated Annealing:**
```bash
python tsptw_sa_v2.py tsptw_edge_suite/wide_overlapping.txt
```

### 2. Run Benchmark Comparison

```bash
# Run on all instances
python bench_runner.py

# Or limit to 15 instances
python bench_runner.py --sample 15
```

This generates:
- `tsptw_bench_results.csv` - Detailed results
- `bench_gap.png` - Gap between SA and optimal
- `bench_runtime.png` - Runtime comparison

### 3. Generate Visualizations

```bash
# Gap by problem size
python viz_suite.py gap_by_n tsptw_bench_results.csv

# Gap by difficulty
python viz_suite.py gap_box tsptw_bench_results.csv

# Runtime comparison
python viz_suite.py runtime tsptw_bench_results.csv

# Performance profile
python viz_suite.py profile tsptw_bench_results.csv
```

## Command-Line Options

### Brute Force Solver
```bash
python tsptw_bruteforce.py <instance.txt> [options]
  --time_limit SECONDS    # Maximum runtime (default: no limit)
  --order {earliest|due|nearest}  # Node ordering heuristic
  --out FILE.json         # Save solution to JSON
```

### Simulated Annealing Solver
```bash
python tsptw_sa_v2.py <instance.txt> [options]
  --seed INT              # Random seed (default: 1234)
  --time_limit SECONDS    # Maximum runtime
  --alpha FLOAT           # Cooling rate (default: 0.985)
  --iters_per_temp INT    # Iterations per temperature (default: 300)
  --T0 FLOAT              # Initial temperature (default: auto-calibrated)
  --out FILE.json         # Output file (default: best_solution.json)
```

### Benchmarking
```bash
python bench_runner.py [options]
  --sample N              # Limit to N instances
  --bf_time_limit SEC     # BF time limit (default: 60s)
  --sa_time_limit SEC     # SA time limit (default: 10s)
  --max_bf_n SIZE         # Max instance size for BF (default: 20)
  --out_csv FILE          # Output CSV (default: tsptw_bench_results.csv)
```

## Project Information

**Course:** Računarska inteligencija
**Institution:** Faculty of Mathematics, University of Belgrade
**Year:** 2025

## References

- TSPTW problem formulation: Classical operations research literature
- Simulated Annealing: Kirkpatrick et al. (1983)
