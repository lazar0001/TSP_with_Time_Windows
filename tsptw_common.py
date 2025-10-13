"""
Common data structures, I/O, and scheduling logic for TSPTW solvers.
Eliminates duplication across tsptw_bruteforce.py, tsptw_sa_v2.py, etc.
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class TSPTWInstance:
    """Time-windowed TSP instance with n nodes (node 0 = depot)."""
    T: List[List[int]]  # n x n travel time matrix
    e: List[int]        # earliest service times
    l: List[int]        # latest service times (due dates)
    s: List[int]        # service durations
    n: int              # number of nodes

    def __init__(self, T: List[List[int]], e: List[int], l: List[int], s: List[int]):
        self.T = T
        self.e = e
        self.l = l
        self.s = s
        self.n = len(T)
        # Precompute min outgoing edge for each node (useful for lower bounds)
        self.min_out = [
            min([T[i][j] for j in range(self.n) if j != i], default=0)
            for i in range(self.n)
        ]


@dataclass
class ScheduleResult:
    """Result of scheduling a route."""
    feasible: bool
    cost: int
    times: List[int]        # service start times at each position
    tardiness_sum: int = 0  # sum of lateness (for soft TW)


def load_instance(path: str) -> TSPTWInstance:
    """
    Load TSPTW instance from text file.
    Format:
      n
      T[0][0] T[0][1] ... T[0][n-1]
      ...
      T[n-1][0] ... T[n-1][n-1]
      0 e[0] l[0] s[0]
      1 e[1] l[1] s[1]
      ...
      n-1 e[n-1] l[n-1] s[n-1]
    """
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]

    it = iter(lines)
    n = int(next(it))

    # Read travel time matrix
    T = []
    for _ in range(n):
        row = list(map(int, next(it).split()))
        if len(row) != n:
            raise ValueError(f"Matrix row length {len(row)} != n={n}")
        T.append(row)

    # Read time windows and service times
    e, l, s = [0] * n, [0] * n, [0] * n
    for _ in range(n):
        i, ei, li, si = map(int, next(it).split())
        e[i], l[i], s[i] = ei, li, si

    return TSPTWInstance(T, e, l, s)


def schedule_route(route: List[int], inst: TSPTWInstance,
                   tw_soft: bool = False, beta: float = 0.0) -> ScheduleResult:
    """
    Compute service start times and travel cost; enforce time windows.

    Args:
        route: Sequence of nodes (should start and end with depot 0)
        inst: TSPTW instance
        tw_soft: If False (hard TW), reject infeasible routes.
                 If True (soft TW), allow lateness with penalty.
        beta: Penalty weight for lateness in soft TW mode.

    Returns:
        ScheduleResult with feasibility, cost, times, and tardiness.
    """
    T, e, l, s = inst.T, inst.e, inst.l, inst.s

    times = [0] * len(route)
    times[0] = e[route[0]]  # Start at depot at earliest opening time

    travel_cost = 0
    tardiness_sum = 0

    for k in range(1, len(route)):
        i, j = route[k-1], route[k]
        travel_cost += T[i][j]

        # Arrive at j and wait if necessary
        t = times[k-1] + s[i] + T[i][j]
        t = max(t, e[j])  # Wait until window opens

        # Check lateness
        lateness = max(0, t - l[j])
        if lateness > 0:
            if not tw_soft:
                # Hard time windows: infeasible
                return ScheduleResult(False, math.inf, [], 0)
            tardiness_sum += lateness

        times[k] = t

    # Total cost includes travel + soft TW penalty
    total_cost = travel_cost + (beta * tardiness_sum if tw_soft else 0)

    return ScheduleResult(True, int(total_cost), times, tardiness_sum)


def compute_forward_backward(route: List[int], inst: TSPTWInstance) -> Tuple[List[int], List[int]]:
    """
    Compute earliest (E) and latest (L) feasible service start times along the fixed route.
    If infeasible, some E[k] > L[k].

    Returns:
        (E, L) where E[k] and L[k] are earliest/latest start times at route[k]
    """
    T, e, l, s = inst.T, inst.e, inst.l, inst.s
    m = len(route)

    E = [0] * m
    L = [0] * m

    # Forward pass: earliest feasible times
    E[0] = e[route[0]]
    for k in range(1, m):
        i, j = route[k-1], route[k]
        t = E[k-1] + s[i] + T[i][j]
        E[k] = max(t, e[j])

    # Backward pass: latest feasible times
    L[-1] = l[route[-1]]
    for k in range(m-2, -1, -1):
        i, j = route[k], route[k+1]
        latest_start_i = min(l[i], L[k+1] - T[i][j] - s[i])
        L[k] = latest_start_i

    return E, L


def forward_precheck_partial(route: List[int], inst: TSPTWInstance,
                             change_start_idx: int = 0) -> bool:
    """
    Fast feasibility precheck: forward schedule from change_start_idx onwards,
    early-exit on first time window violation.

    Args:
        route: Route to check
        inst: TSPTW instance
        change_start_idx: Index where route was modified (optimization hint)

    Returns:
        True if route appears feasible under hard time windows
    """
    T, e, l, s = inst.T, inst.e, inst.l, inst.s

    # Start from beginning for simplicity (could optimize using change_start_idx)
    t = e[route[0]]
    for k in range(1, len(route)):
        i, j = route[k-1], route[k]
        t = max(e[j], t + s[i] + T[i][j])
        if t > l[j]:
            return False

    return True
