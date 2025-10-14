
import os, re, json
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple
from tsptw_common import schedule_route as schedule_common, TSPTWInstance

# --------------------
# CSV loading helpers
# --------------------
def load_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Derive size n (already present) and difficulty (if encoded in 'suite')
    def parse_diff(s: str) -> Optional[str]:
        if isinstance(s, str):
            m = re.search(r'(easy|medium|hard)', s, re.IGNORECASE)
            return m.group(1).lower() if m else None
        return None
    if 'difficulty' not in df.columns:
        df['difficulty'] = df['suite'].apply(parse_diff) if 'suite' in df.columns else None
    return df

# --------------------
# Plots (each separate)
# --------------------
def plot_gap_by_n(csv_path: str, out_path: str = "gap_by_n.png") -> str:
    df = load_results(csv_path)
    sdf = df.dropna(subset=["gap_pct"])
    if sdf.empty:
        raise ValueError("No gap data (need BF and SA on same instances).")
    gp = sdf.groupby("n")["gap_pct"].mean().reset_index()
    plt.figure()
    plt.plot(gp["n"], gp["gap_pct"], marker="o")
    plt.xlabel("n (number of nodes)")
    plt.ylabel("Mean gap (%)")
    plt.title("SA mean gap vs optimal (by n)")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path

def plot_gap_box_by_difficulty(csv_path: str, out_path: str = "gap_box_by_difficulty.png") -> str:
    df = load_results(csv_path)
    sdf = df.dropna(subset=["gap_pct", "difficulty"])
    if sdf.empty:
        raise ValueError("No gap data with difficulty labels.")
    # Boxplot by difficulty order
    order = ["easy","medium","hard"]
    data = [sdf.loc[sdf["difficulty"]==d, "gap_pct"].values for d in order if (sdf["difficulty"]==d).any()]
    labels = [d for d in order if (sdf["difficulty"]==d).any()]
    plt.figure()
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.xlabel("Difficulty")
    plt.ylabel("Gap (%)")
    plt.title("SA gap distribution by difficulty")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path

def plot_runtime_scatter(csv_path: str, out_path: str = "runtime_scatter.png") -> str:
    df = load_results(csv_path)
    plt.figure()
    if "sa_time_sec" in df.columns and df["sa_time_sec"].notna().any():
        plt.scatter(df["n"], df["sa_time_sec"], label="SA time (s)")
    if "bf_time_sec" in df.columns and df["bf_time_sec"].notna().any():
        plt.scatter(df["n"], df["bf_time_sec"], label="BF time (s)")
    plt.xlabel("n (number of nodes)")
    plt.ylabel("Runtime (s)")
    plt.title("Runtime vs problem size")
    plt.legend()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path

def plot_performance_profile(csv_path: str, out_path: str = "performance_profile.png") -> str:
    """
    Dolan-Moré style (simplified) performance profile for quality:
    r_i = SA_cost / OPT; plot fraction of instances with r_i <= tau.
    """
    df = load_results(csv_path)
    sdf = df.dropna(subset=["gap_pct", "bf_cost", "sa_cost"])
    if sdf.empty:
        raise ValueError("No instances with both BF and SA results.")
    ratios = (sdf["sa_cost"] / sdf["bf_cost"]).values
    ratios = sorted(ratios)
    taus = sorted(set(ratios + [1.0, max(1.0, max(ratios))]))
    xs, ys = [], []
    n = len(ratios)
    for tau in taus:
        xs.append(tau)
        ys.append(sum(1 for r in ratios if r <= tau) / n)
    plt.figure()
    plt.plot(xs, ys)
    plt.xlabel("Quality ratio τ = cost(SA)/cost(OPT)")
    plt.ylabel("Fraction of instances ≤ τ")
    plt.title("Performance profile (quality)")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path

# --------------------
# Route timeline / Gantt-like plot
# --------------------
def _load_instance(inst_path: str):
    with open(inst_path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
    it = iter(lines)
    n = int(next(it))
    T = [list(map(int, next(it).split())) for _ in range(n)]
    e = [0]*n; l=[0]*n; s=[0]*n
    for _ in range(n):
        i, ei, li, si = map(int, next(it).split())
        e[i], l[i], s[i] = ei, li, si
    return n, T, e, l, s

def _schedule_route(route: List[int], T, e, l, s) -> Tuple[bool, List[int]]:
    """Wrapper for backward compatibility with old signature."""
    inst = TSPTWInstance(T, e, l, s)
    result = schedule_common(route, inst)
    return result.feasible, result.times

def plot_route_timeline(instance_path: str, route_json_or_list, out_path: str = "route_timeline.png") -> str:
    n, T, e, l, s = _load_instance(instance_path)
    if isinstance(route_json_or_list, str):
        with open(route_json_or_list, "r") as f:
            js = json.load(f)
        route = js.get("route")
        if route is None:
            raise ValueError("JSON must contain a 'route' list.")
    else:
        route = list(route_json_or_list)
    feas, times = _schedule_route(route, T, e, l, s)
    if not feas:
        raise ValueError("Provided route is infeasible under hard TW.")

    # Exclude the starting depot at index 0 for plotting rows; plot customers (and optionally final depot)
    nodes = route[1:]
    y_positions = list(range(len(nodes)))
    plt.figure()
    # For each node, draw its time window and the service interval
    for y, node in enumerate(nodes):
        # time window [e[node], l[node]]
        plt.hlines(y, e[node], l[node])
        # service interval [times[k], times[k] + s[node]]
        # find k index for this node
        k = y + 1  # since nodes = route[1:]
        plt.hlines(y, times[k], times[k] + s[node])
        # arrival marker at times[k]
        plt.scatter([times[k]], [y])
    plt.yticks(y_positions, [str(node) for node in nodes])
    plt.xlabel("Time")
    plt.ylabel("Node")
    plt.title("Route timeline (windows vs arrivals/services)")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path

# --------------------
# CLI
# --------------------
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Visualization suite for TSPTW results")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("gap_by_n")
    p1.add_argument("csv")
    p1.add_argument("--out", default="gap_by_n.png")

    p2 = sub.add_parser("gap_box")
    p2.add_argument("csv")
    p2.add_argument("--out", default="gap_box_by_difficulty.png")

    p3 = sub.add_parser("runtime")
    p3.add_argument("csv")
    p3.add_argument("--out", default="runtime_scatter.png")

    p4 = sub.add_parser("profile")
    p4.add_argument("csv")
    p4.add_argument("--out", default="performance_profile.png")

    p5 = sub.add_parser("timeline")
    p5.add_argument("instance")
    p5.add_argument("route_json_or_list")
    p5.add_argument("--out", default="route_timeline.png")

    args = ap.parse_args()

    if args.cmd == "gap_by_n":
        print(plot_gap_by_n(args.csv, args.out))
    elif args.cmd == "gap_box":
        print(plot_gap_box_by_difficulty(args.csv, args.out))
    elif args.cmd == "runtime":
        print(plot_runtime_scatter(args.csv, args.out))
    elif args.cmd == "profile":
        print(plot_performance_profile(args.csv, args.out))
    elif args.cmd == "timeline":
        # Support both JSON path or a comma-separated list (e.g., "0,1,3,2,0")
        inp = args.route_json_or_list
        if os.path.isfile(inp):
            print(plot_route_timeline(args.instance, inp, args.out))
        else:
            route = [int(x) for x in inp.split(",")]
            print(plot_route_timeline(args.instance, route, args.out))

if __name__ == "__main__":
    main()
