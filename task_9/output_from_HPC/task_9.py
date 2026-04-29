from __future__ import annotations

import argparse
import csv
import time
from os.path import join

import numpy as np
from numba import cuda

import cupy as cp

LOAD_DIR_DEFAULT = "/dtu/projects/02613_2025/data/modified_swiss_dwellings"


def load_data(load_dir: str, bid: str) -> tuple[cp.ndarray, cp.ndarray]:
    size = 512
    u = cp.zeros((size + 2, size + 2), dtype=cp.float64)
    u[1:-1, 1:-1] = cp.load(join(load_dir, f"{bid}_domain.npy"))
    interior = cp.load(join(load_dir, f"{bid}_interior.npy")).astype(cp.bool_)
    return u, interior


def jacobi_cupy(u: cp.ndarray, interior: cp.ndarray) -> None:
    u[1 : -1, 1 : -1] = 0.25 * (
                u[ : -2, 1 : -1]
                + u[2 : ,1 : -1]
                + u[1 : -1, : -2]
                + u[1 : -1, 2 : ]
            )
    return cp.array(u)

def jacobi_cupy(u, interior):
    u[1:-1, 1:-1][interior] = 0.25 * (
        u[:-2, 1:-1] + u[2:, 1:-1] +
        u[1:-1, :-2] + u[1:-1, 2:]
    )[interior]
    return u
    

def jacobi_cupy_main(
    u0: cp.ndarray,
    interior_mask: cp.ndarray,
    max_iter: int,
    tolerance: float = 1e-6
) -> cp.ndarray:
    # Fixed-iteration helper (no atol), as requested in task 8.
    u = cp.array(u0)
    d_interior = cp.array(interior_mask)

    for _ in range(max_iter):
        u_old = u.copy()
        u = jacobi_cupy(u_old, d_interior)
        if cp.max(cp.abs(u - u_old)) < tolerance:
            return cp.array(u)
    return cp.array(u)


def summary_stats(u: cp.ndarray, interior_mask: cp.ndarray) -> dict[str, float]:
    u_interior = u[1:-1, 1:-1][interior_mask]
    return {
        "mean_temp": float(u_interior.mean()),
        "std_temp": float(u_interior.std()),
        "pct_above_18": float(cp.sum(u_interior > 18) / u_interior.size * 100.0),
        "pct_below_15": float(cp.sum(u_interior < 15) / u_interior.size * 100.0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 9: GPU Jacobi")
    parser.add_argument("--n-buildings", type=int, default=20)
    parser.add_argument("--max-iter", type=int, default=20_000)
    parser.add_argument("--load-dir", type=str, default=LOAD_DIR_DEFAULT)
    parser.add_argument("--csv-out", type=str, default="task9_results.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(join(args.load_dir, "building_ids.txt"), "r", encoding="utf-8") as f:
        building_ids = f.read().splitlines()[: args.n_buildings]

    # Warm-up kernel compile.
    warm_u = cp.zeros((514, 514), dtype=cp.float64)
    warm_mask = cp.zeros((512, 512), dtype=cp.bool_)
    _ = jacobi_cupy_main(warm_u, warm_mask, 1)

    t0 = time.perf_counter()
    rows: list[dict[str, float | str]] = []
    for bid in building_ids:
        u0, interior = load_data(args.load_dir, bid)
        u = jacobi_cupy_main(u0, interior, args.max_iter)
        stats = summary_stats(u, interior)
        rows.append({"building_id": bid, **stats})
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - t0

    print(f'The elapsed time per building is {elapsed / args.n_buildings} with the total elapsing time for 4571 floors being {4571 * elapsed / args.n_buildings}')


    fields = ["building_id", "mean_temp", "std_temp", "pct_above_18", "pct_below_15"]
    with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Processed {len(building_ids)} buildings in {elapsed:.3f} s")
    print(f"CSV written to {args.csv_out}")


if __name__ == "__main__":
    main()
