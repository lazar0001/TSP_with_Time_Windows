
from typing import List, Tuple, Dict, Optional
import math, time, argparse, json
from tsptw_common import TSPTWInstance, load_instance

def brute_force_tsptw(inst: TSPTWInstance,
                      time_limit_sec: Optional[float]=None,
                      order: str = "earliest") -> Dict:
    """
    Depth-first enumeration with pruning for hard TSPTW.
    order: 'earliest' (by e_i), 'due' (by l_i), 'nearest' (by T from current).
    Returns dict with best route/cost/times and stats.
    """
    n = inst.n
    T,e,l,s = inst.T, inst.e, inst.l, inst.s
    start_time = time.time()

    # Candidate ordering base list (excluding depot)
    base_nodes = list(range(1, n))
    if order == "earliest":
        base_nodes.sort(key=lambda i: (e[i], l[i]))
    elif order == "due":
        base_nodes.sort(key=lambda i: (l[i], e[i]))
    # 'nearest' handled dynamically in recursion

    best_cost = math.inf
    best_route = None
    best_times = None

    stats = {"expanded":0, "pruned_tw":0, "pruned_lb":0, "finished": True}

    # Lower bound on remaining travel cost (very simple, optimistic)
    def lb_travel(curr: int, unvisited: List[int]) -> int:
        if not unvisited:
            return T[curr][0]  # must return to depot
        # Sum of min outgoing for each unvisited + cost to return from current
        return T[curr][0] + sum(inst.min_out[u] for u in unvisited)

    # DFS recursion
    def dfs(curr: int, t_start_curr: int, travel_cost_so_far: int,
            visited_mask: int, route: List[int], times: List[int]):
        nonlocal best_cost, best_route, best_times, stats

        # Time limit
        if time_limit_sec is not None and (time.time() - start_time) > time_limit_sec:
            stats["finished"] = False
            return

        # If all customers visited, try to return to depot
        if visited_mask == (1 << n) - 1:
            # return to depot
            t_arrive = t_start_curr + s[curr] + T[curr][0]
            t0 = max(t_arrive, e[0])
            if t0 > l[0]:
                stats["pruned_tw"] += 1
                return
            cost_final = travel_cost_so_far + T[curr][0]
            if cost_final < best_cost:
                best_cost = cost_final
                best_route = route + [0]
                best_times = times + [t0]
            return

        # Lower bound pruning
        # Build list of unvisited
        unvisited = [i for i in range(1, n) if not (visited_mask & (1 << i))]
        # quick bound
        bound = travel_cost_so_far + lb_travel(curr, unvisited)
        if bound >= best_cost:
            stats["pruned_lb"] += 1
            return

        # Expand children
        stats["expanded"] += 1

        # Determine candidate order
        if order == "nearest":
            cand = sorted(unvisited, key=lambda j: T[curr][j])
        else:
            cand = base_nodes if order in ("earliest","due") else unvisited
            # filter only unvisited
            cand = [j for j in cand if (visited_mask & (1 << j)) == 0]

        for j in cand:
            # forward feasibility step
            t_arrive = t_start_curr + s[curr] + T[curr][j]
            t_j = max(t_arrive, e[j])
            if t_j > l[j]:
                stats["pruned_tw"] += 1
                continue
            # new travel cost
            new_travel = travel_cost_so_far + T[curr][j]
            # second bound: optimistic remaining after choosing j
            rem = [u for u in unvisited if u != j]
            bound2 = new_travel + lb_travel(j, rem)
            if bound2 >= best_cost:
                stats["pruned_lb"] += 1
                continue
            dfs(j, t_j, new_travel, visited_mask | (1 << j), route + [j], times + [t_j])

    # Start at depot at e[0]
    dfs(curr=0, t_start_curr=e[0], travel_cost_so_far=0,
        visited_mask=1, route=[0], times=[e[0]])

    return {
        "optimal_cost": best_cost if best_route is not None else None,
        "optimal_route": best_route,
        "times": best_times,
        "stats": stats,
        "n": n,
        "order": order,
        "time_spent_sec": round(time.time() - start_time, 3)
    }

def main():
    ap = argparse.ArgumentParser(description="Brute-force (DFS) solver for TSPTW (hard windows)")
    ap.add_argument("instance", help="Path to .txt instance")
    ap.add_argument("--time_limit", type=float, default=None, help="Time limit in seconds")
    ap.add_argument("--order", type=str, default="earliest", choices=["earliest","due","nearest"],
                    help="Node expansion order heuristic")
    ap.add_argument("--out", type=str, default=None, help="JSON output path")
    args = ap.parse_args()

    inst = load_instance(args.instance)
    res = brute_force_tsptw(inst, time_limit_sec=args.time_limit, order=args.order)
    if res["optimal_route"] is None:
        print("No solution found (possibly infeasible or time limit reached).")
    else:
        print("Optimal cost:", res["optimal_cost"])
        print("Optimal route:", " -> ".join(map(str, res["optimal_route"])))
        print("Times:", res["times"])
    print("Stats:", res["stats"], "Time:", res["time_spent_sec"], "s")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)

if __name__ == "__main__":
    main()
