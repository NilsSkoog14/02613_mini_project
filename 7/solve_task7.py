from __future__ import annotations

import argparse
import csv
import time
from os.path import join

import numpy as np
from numba import njit

LOAD_DIR_DEFAULT = "/dtu/projects/02613_2025/data/modified_swiss_dwellings"


def load_data(load_dir: str, bid: str) -> tuple[np.ndarray, np.ndarray]:
    size = 512
    u = np.zeros((size + 2, size + 2), dtype=np.float64)
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior = np.load(join(load_dir, f"{bid}_interior.npy")).astype(np.bool_)
    return u, interior


@njit(cache=True)
def jacobi_numba_cpu(
    u0: np.ndarray,
    interior_mask: np.ndarray,
    max_iter: int,
    atol: float,
) -> np.ndarray:
    # Ping-pong buffers avoid extra allocations inside the loop.
    u_old = u0.copy()
    u_new = u0.copy()

    ny = u0.shape[0]
    nx = u0.shape[1]

    for _ in range(max_iter):
        max_delta = 0.0

        # Row-major traversal keeps accesses cache-friendly in NumPy/Numba.
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                if interior_mask[i - 1, j - 1]:
                    new_val = 0.25 * (
                        u_old[i, j - 1]
                        + u_old[i, j + 1]
                        + u_old[i - 1, j]
                        + u_old[i + 1, j]
                    )
                    delta = abs(new_val - u_old[i, j])
                    if delta > max_delta:
                        max_delta = delta
                    u_new[i, j] = new_val
                else:
                    u_new[i, j] = u_old[i, j]

        # If converged, return the latest iterate computed in u_new.
        if max_delta < atol:
            return u_new

        # Swap pointers for next iteration.
        tmp = u_old
        u_old = u_new
        u_new = tmp

    return u_old


def summary_stats(u: np.ndarray, interior_mask: np.ndarray) -> dict[str, float]:
    u_interior = u[1:-1, 1:-1][interior_mask]
    return {
        "mean_temp": float(u_interior.mean()),
        "std_temp": float(u_interior.std()),
        "pct_above_18": float(np.sum(u_interior > 18) / u_interior.size * 100.0),
        "pct_below_15": float(np.sum(u_interior < 15) / u_interior.size * 100.0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 7: CPU Numba Jacobi")
    parser.add_argument("--n-buildings", type=int, default=20, help="How many buildings to process")
    parser.add_argument("--max-iter", type=int, default=20_000)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--load-dir", type=str, default=LOAD_DIR_DEFAULT)
    parser.add_argument("--csv-out", type=str, default="task7_results.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(join(args.load_dir, "building_ids.txt"), "r", encoding="utf-8") as f:
        building_ids = f.read().splitlines()[: args.n_buildings]

    # Warm-up compile once to exclude JIT startup from measured runtime.
    warm_u = np.zeros((514, 514), dtype=np.float64)
    warm_mask = np.zeros((512, 512), dtype=np.bool_)
    jacobi_numba_cpu(warm_u, warm_mask, 1, 1e-4)

    t0 = time.perf_counter()
    rows: list[dict[str, float | str]] = []
    for bid in building_ids:
        u0, interior = load_data(args.load_dir, bid)
        u = jacobi_numba_cpu(u0, interior, args.max_iter, args.atol)
        stats = summary_stats(u, interior)
        rows.append({"building_id": bid, **stats})
    elapsed = time.perf_counter() - t0

    fields = ["building_id", "mean_temp", "std_temp", "pct_above_18", "pct_below_15"]
    with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Processed {len(building_ids)} buildings in {elapsed:.3f} s")
    print(f"CSV written to {args.csv_out}")


if __name__ == "__main__":
    main()
