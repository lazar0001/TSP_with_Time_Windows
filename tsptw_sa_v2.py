from typing import List, Tuple, Dict, Optional
import math, random, sys, time, argparse, json
from tsptw_common import (
    TSPTWInstance, load_instance, schedule_route,
    compute_forward_backward, forward_precheck_partial
)

# ==============================
# Scheduling & feasibility - now imported from tsptw_common
# ==============================

# ==============================
# Construction heuristic
# ==============================

def greedy_feasible_insertion(inst: 'TSPTWInstance', order: Optional[List[int]]=None, max_tries: int=200) -> List[int]:
    """
    Build a feasible tour [0, ..., 0] by inserting nodes one by one at the best feasible place.
    Returns a route if found; otherwise returns a simple 0..n-1..0 sequence (may be infeasible).
    """
    n = inst.n
    nodes = list(range(1, n))
    if order is None:
        nodes.sort(key=lambda i: (inst.e[i], inst.l[i]))
    else:
        nodes = order[:]
    rng = random.Random(0xC0FFEE)
    for _ in range(max_tries):
        route = [0, 0]
        ok = True
        for v in nodes:
            best_pos, best_cost = None, math.inf
            for pos in range(1, len(route)):  # insert before route[pos]
                cand = route[:pos] + [v] + route[pos:]
                result = schedule_route(cand, inst, tw_soft=False)
                if result.feasible and result.cost < best_cost:
                    best_pos, best_cost = pos, result.cost
            if best_pos is None:
                ok = False
                break
            route = route[:best_pos] + [v] + route[best_pos:]
        if ok:
            return route
        rng.shuffle(nodes)
    return list(range(0, n)) + [0]  # fallback (may be infeasible)

# ==============================
# Neighborhoods
# ==============================

def move_relocate(route: List[int], i: int, j: int) -> Tuple[List[int], int, int]:
    if i == 0 or i == len(route)-1:
        return route, i, j
    r = route[:]
    node = r.pop(i)
    j = min(j, len(r))
    r.insert(j, node)
    return r, min(i,j), max(i,j)

def move_swap(route: List[int], i: int, j: int) -> Tuple[List[int], int, int]:
    if i == 0 or j == 0 or i == len(route)-1 or j == len(route)-1:
        return route, i, j
    if i == j: return route, i, j
    r = route[:]
    r[i], r[j] = r[j], r[i]
    return r, min(i,j), max(i,j)

def move_two_opt(route: List[int], i: int, j: int) -> Tuple[List[int], int, int]:
    if i < 1: i = 1
    if j > len(route)-2: j = len(route)-2
    if i >= j: return route, i, j
    r = route[:i] + list(reversed(route[i:j+1])) + route[j+1:]
    return r, i, j

def random_move(route: List[int], rng: random.Random) -> Tuple[str, List[int], int, int]:
    mtype = rng.choice(["relocate", "swap", "2opt"])
    i = rng.randint(1, len(route)-2)
    j = rng.randint(1, len(route)-2)
    if mtype == "relocate":
        r,i1,i2 = move_relocate(route, i, j)
        return mtype, r, i1, i2
    elif mtype == "swap":
        r,i1,i2 = move_swap(route, i, j)
        return mtype, r, i1, i2
    else:
        if i > j: i, j = j, i
        r,i1,i2 = move_two_opt(route, i, j)
        return mtype, r, i1, i2

# ==============================
# Temperature calibration
# ==============================

def auto_temperature(inst: TSPTWInstance, route: List[int], rng: random.Random,
                     sample_moves: int=200, target_accept: float=0.75,
                     tw_soft: bool=False, beta: float=0.0) -> float:
    pos_deltas = []
    result0 = schedule_route(route, inst, tw_soft=tw_soft, beta=beta)
    if not result0.feasible:
        return 100.0
    for _ in range(sample_moves):
        _, cand, i, j = random_move(route, rng)
        if not forward_precheck_partial(cand, inst, min(i,j)):
            continue
        result2 = schedule_route(cand, inst, tw_soft=tw_soft, beta=beta)
        if result2.feasible and result2.cost > result0.cost:
            pos_deltas.append(result2.cost - result0.cost)
    if not pos_deltas:
        return 1.0
    avg_delta = sum(pos_deltas) / len(pos_deltas)
    try:
        return max(1e-6, -avg_delta / math.log(max(1e-6, target_accept)))
    except ValueError:
        return 10.0

# ==============================
# Simulated Annealing
# ==============================

def sa_solve(inst: TSPTWInstance,
             route0: Optional[List[int]] = None,
             T0: Optional[float] = None,
             alpha: float = 0.98,
             iters_per_temp: int = 200,
             min_T: float = 1e-3,
             max_iter_no_improve: int = 5000,
             time_limit_sec: Optional[float] = None,
             rng_seed: int = 1234,
             tw_soft: bool = False,
             beta: float = 0.0) -> Dict:
    rng = random.Random(rng_seed)
    # Initial route
    if route0 is None:
        route = greedy_feasible_insertion(inst)
        if route[0] != 0: route = [0] + route
        if route[-1] != 0: route = route + [0]
    else:
        route = route0[:]
    result = schedule_route(route, inst, tw_soft=tw_soft, beta=beta)
    if not result.feasible and not tw_soft:
        order = sorted(range(1, inst.n), key=lambda i: (inst.e[i], inst.l[i]))
        route = greedy_feasible_insertion(inst, order=order)
        if route[0] != 0: route = [0] + route
        if route[-1] != 0: route = route + [0]
        result = schedule_route(route, inst, tw_soft=tw_soft, beta=beta)

    cost, times, td = result.cost, result.times, result.tardiness_sum

    # Auto temperature if not provided
    if T0 is None:
        T0 = auto_temperature(inst, route, rng, tw_soft=tw_soft, beta=beta)
    Tcur = T0

    best_route, best_cost, best_times, best_td = route[:], cost, times[:], td
    curr_route, curr_cost, curr_times, curr_td = route[:], cost, times[:], td

    no_improve = 0
    total_iters = 0
    start = time.time()
    logs = []
    epoch = 0

    while Tcur > min_T and no_improve < max_iter_no_improve:
        if time_limit_sec is not None and (time.time() - start) > time_limit_sec:
            break
        tries = 0
        accepted = 0
        for _ in range(iters_per_temp):
            tries += 1
            mtype, cand_route, i1, i2 = random_move(curr_route, rng)
            if not forward_precheck_partial(cand_route, inst, i1):
                continue
            result2 = schedule_route(cand_route, inst, tw_soft=tw_soft, beta=beta)
            if not result2.feasible and not tw_soft:
                continue
            delta = result2.cost - curr_cost
            accept = False
            if delta <= 0:
                accept = True
            else:
                if rng.random() < math.exp(-delta / max(1e-9, Tcur)):
                    accept = True
            if accept:
                accepted += 1
                curr_route, curr_cost, curr_times, curr_td = cand_route, result2.cost, result2.times, result2.tardiness_sum
                if result2.cost < best_cost:
                    best_route, best_cost, best_times, best_td = cand_route, result2.cost, result2.times, result2.tardiness_sum
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1

        logs.append({
            "epoch": epoch, "T": Tcur, "best_cost": best_cost,
            "curr_cost": curr_cost, "accept_rate": accepted / max(1, tries),
            "elapsed_sec": round(time.time() - start, 3)
        })
        epoch += 1
        Tcur *= alpha

    return {
        "route": best_route,
        "cost": best_cost,
        "times": best_times,
        "tardiness_sum": best_td,
        "accepted_cost": curr_cost,
        "iterations": total_iters,
        "final_T": Tcur,
        "log": logs,
        "params": {
            "T0": T0, "alpha": alpha, "iters_per_temp": iters_per_temp,
            "min_T": min_T, "max_iter_no_improve": max_iter_no_improve,
            "time_limit_sec": time_limit_sec, "tw_soft": tw_soft, "beta": beta,
            "seed": rng_seed
        }
    }

# ==============================
# CLI
# ==============================

def main():
    ap = argparse.ArgumentParser(description="TSPTW Simulated Annealing (enhanced)")
    ap.add_argument("instance", help="Path to .txt instance")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--time_limit", type=float, default=None, help="Time limit in seconds")
    ap.add_argument("--alpha", type=float, default=0.985)
    ap.add_argument("--iters_per_temp", type=int, default=300)
    ap.add_argument("--min_T", type=float, default=1e-3)
    ap.add_argument("--T0", type=float, default=None)
    ap.add_argument("--soft", action="store_true", help="Use soft time windows with penalty")
    ap.add_argument("--beta", type=float, default=10.0, help="Penalty weight for lateness in soft mode")
    ap.add_argument("--out", type=str, default="best_solution.json")
    args = ap.parse_args()

    inst = load_instance(args.instance)
    res = sa_solve(inst,
                   T0=args.T0, alpha=args.alpha, iters_per_temp=args.iters_per_temp,
                   min_T=args.min_T, time_limit_sec=args.time_limit, rng_seed=args.seed,
                   tw_soft=args.soft, beta=args.beta)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Best cost: {res['cost']} (tardiness_sum={res['tardiness_sum']})")
    print("Best route:", " -> ".join(map(str, res["route"])))
    print("Saved result to:", args.out)

if __name__ == "__main__":
    main()
