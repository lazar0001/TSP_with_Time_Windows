
import argparse, os, time, math, json
import importlib.util
from typing import List, Dict, Any, Tuple
import pandas as pd
import matplotlib.pyplot as plt
from tsptw_common import load_instance

# ---------- Utility: dynamic import by path ----------
def import_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod

# ---------- Load our solvers ----------
HERE = os.path.dirname(os.path.abspath(__file__))
BF_PATH = os.path.join(HERE, "tsptw_bruteforce.py")
SA_PATH = os.path.join(HERE, "tsptw_sa_v2.py")
if not os.path.exists(BF_PATH) or not os.path.exists(SA_PATH):
    raise FileNotFoundError("Expected tsptw_bruteforce.py and tsptw_sa_v2.py in the same folder as this script.")

bf = import_from_path("tsptw_bruteforce", BF_PATH)
sa = import_from_path("tsptw_sa_v2", SA_PATH)

# ---------- Instance discovery ----------
def find_instances(roots: List[str]) -> List[str]:
    files = []
    for root in roots:
        if not os.path.exists(root):
            continue
        for dp, _, fns in os.walk(root):
            for fn in fns:
                if fn.lower().endswith(".txt"):
                    files.append(os.path.join(dp, fn))
    files.sort()
    return files

def read_n_quick(path: str) -> int:
    with open(path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            try:
                return int(ln.split()[0])
            except:
                break
    raise ValueError(f"Cannot read n from {path}")

# ---------- Benchmark ----------
def run_bench(inst_paths: List[str],
              out_csv: str,
              bf_time_limit: float = 60.0,
              sa_time_limit: float = 10.0,
              max_bf_n: int = 20,
              sa_alpha: float = 0.985,
              sa_iters_per_temp: int = 300,
              sa_min_T: float = 1e-3,
              sa_seed: int = 1234,
              sa_soft: bool = False,
              sa_beta: float = 0.0,
              sample: int = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows = []
    if sample is not None and sample > 0:
        inst_paths = inst_paths[:sample]

    for idx, p in enumerate(inst_paths, 1):
        try:
            n = read_n_quick(p)
        except Exception as e:
            print(f"[skip] {p}: cannot read n ({e})")
            continue

        suite = os.path.basename(os.path.dirname(p))
        name = os.path.relpath(p, os.path.commonpath([os.path.dirname(p), os.path.dirname(p)]))

        # --- Brute force (only if n <= max_bf_n) ---
        bf_cost = None
        bf_time = None
        bf_ok = False
        bf_stats = None
        if n <= max_bf_n:
            try:
                inst = load_instance(p)
                t0 = time.time()
                res_bf = bf.brute_force_tsptw(inst, time_limit_sec=bf_time_limit, order="earliest")
                bf_time = time.time() - t0
                bf_cost = res_bf.get("optimal_cost")
                bf_ok = bf_cost is not None
                bf_stats = res_bf.get("stats", {})
            except Exception as e:
                print(f"[BF error] {p}: {e}")

        # --- SA ---
        sa_cost = None
        sa_time = None
        sa_td = None
        sa_route = None
        try:
            inst_sa = load_instance(p)
            t0 = time.time()
            res_sa = sa.sa_solve(inst_sa, T0=None, alpha=sa_alpha, iters_per_temp=sa_iters_per_temp,
                                 min_T=sa_min_T, time_limit_sec=sa_time_limit, rng_seed=sa_seed,
                                 tw_soft=sa_soft, beta=sa_beta)
            sa_time = time.time() - t0
            sa_cost = res_sa.get("cost")
            sa_td = res_sa.get("tardiness_sum")
            sa_route = res_sa.get("route")
        except Exception as e:
            print(f"[SA error] {p}: {e}")

        gap = None
        if bf_ok and sa_cost is not None and bf_cost and bf_cost > 0:
            gap = 100.0 * (sa_cost - bf_cost) / bf_cost

        rows.append({
            "suite": suite,
            "instance": os.path.basename(p),
            "path": p,
            "n": n,
            "bf_cost": bf_cost,
            "bf_time_sec": round(bf_time,3) if bf_time is not None else None,
            "bf_ok": bf_ok,
            "bf_pruned_tw": (bf_stats or {}).get("pruned_tw"),
            "bf_pruned_lb": (bf_stats or {}).get("pruned_lb"),
            "sa_cost": sa_cost,
            "sa_time_sec": round(sa_time,3) if sa_time is not None else None,
            "sa_tardiness": sa_td,
            "gap_pct": None if gap is None else round(gap,3),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    # Prepare simple summary
    summary = {
        "num_instances": len(df),
        "num_bf": int(df["bf_cost"].notna().sum()) if not df.empty else 0,
        "gap_mean_pct": float(df["gap_pct"].dropna().mean()) if "gap_pct" in df and df["gap_pct"].notna().any() else None,
        "gap_median_pct": float(df["gap_pct"].dropna().median()) if "gap_pct" in df and df["gap_pct"].notna().any() else None,
    }
    return df, summary

def plot_gap(df: 'pd.DataFrame', out_path: str):
    sdf = df.dropna(subset=["gap_pct"]).reset_index(drop=True)
    if sdf.empty:
        return None
    plt.figure()
    plt.plot(range(len(sdf)), sdf["gap_pct"])
    plt.xlabel("Instance (where BF available)")
    plt.ylabel("SA Gap (%)")
    plt.title("SA gap vs Optimal (BF)")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path

def plot_runtime(df: 'pd.DataFrame', out_path: str):
    if df.empty:
        return None
    plt.figure()
    # Plot SA times
    if df["sa_time_sec"].notna().any():
        plt.scatter(df["n"], df["sa_time_sec"], label="SA time (s)")
    # Plot BF times where available
    if df["bf_time_sec"].notna().any():
        plt.scatter(df["n"], df["bf_time_sec"], label="BF time (s)")
    plt.xlabel("n (number of nodes)")
    plt.ylabel("Runtime (s)")
    plt.title("Runtime vs Problem Size")
    plt.legend()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path

def main():
    ap = argparse.ArgumentParser(description="Benchmark runner: Brute-force vs Simulated Annealing on TSPTW instances")
    ap.add_argument("--roots", nargs="+", default=[
        os.path.join(os.path.dirname(__file__), "tsptw_edge_suite"),
        os.path.join(os.path.dirname(__file__), "tsptw_SA_suite"),
    ], help="Folders to scan for .txt instances")
    ap.add_argument("--out_csv", default="tsptw_bench_results.csv")
    ap.add_argument("--bf_time_limit", type=float, default=60.0)
    ap.add_argument("--sa_time_limit", type=float, default=10.0)
    ap.add_argument("--max_bf_n", type=int, default=20)
    ap.add_argument("--sa_alpha", type=float, default=0.985)
    ap.add_argument("--sa_iters_per_temp", type=int, default=300)
    ap.add_argument("--sa_min_T", type=float, default=1e-3)
    ap.add_argument("--sa_seed", type=int, default=1234)
    ap.add_argument("--sa_soft", action="store_true")
    ap.add_argument("--sa_beta", type=float, default=0.0)
    ap.add_argument("--sample", type=int, default=None, help="Optional: limit number of instances processed")
    ap.add_argument("--out_prefix", type=str, default="bench")
    args = ap.parse_args()

    inst_paths = find_instances(args.roots)
    if not inst_paths:
        print("No .txt instances found under:", args.roots)
        return

    df, summary = run_bench(inst_paths, args.out_csv,
                            bf_time_limit=args.bf_time_limit,
                            sa_time_limit=args.sa_time_limit,
                            max_bf_n=args.max_bf_n,
                            sa_alpha=args.sa_alpha,
                            sa_iters_per_temp=args.sa_iters_per_temp,
                            sa_min_T=args.sa_min_T,
                            sa_seed=args.sa_seed,
                            sa_soft=args.sa_soft,
                            sa_beta=args.sa_beta,
                            sample=args.sample)
    print("Summary:", summary)
    print("Saved CSV:", args.out_csv)

    gap_png = plot_gap(df, args.out_prefix + "_gap.png")
    rt_png  = plot_runtime(df, args.out_prefix + "_runtime.png")
    if gap_png: print("Saved:", gap_png)
    if rt_png: print("Saved:", rt_png)

if __name__ == "__main__":
    main()
