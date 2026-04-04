from __future__ import annotations

import argparse
import csv
import time
from os.path import join

import numpy as np
from numba import cuda

LOAD_DIR_DEFAULT = "/dtu/projects/02613_2025/data/modified_swiss_dwellings"


def load_data(load_dir: str, bid: str) -> tuple[np.ndarray, np.ndarray]:
    size = 512
    u = np.zeros((size + 2, size + 2), dtype=np.float64)
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior = np.load(join(load_dir, f"{bid}_interior.npy")).astype(np.bool_)
    return u, interior


@cuda.jit
def jacobi_step_kernel(u_old: np.ndarray, u_new: np.ndarray, interior: np.ndarray) -> None:
    i, j = cuda.grid(2)
    # interior is 512x512, and maps to u indices [1:513, 1:513]
    if i < interior.shape[0] and j < interior.shape[1]:
        ui = i + 1
        uj = j + 1
        if interior[i, j]:
            u_new[ui, uj] = 0.25 * (
                u_old[ui, uj - 1]
                + u_old[ui, uj + 1]
                + u_old[ui - 1, uj]
                + u_old[ui + 1, uj]
            )
        else:
            u_new[ui, uj] = u_old[ui, uj]


def jacobi_cuda_fixed_iter(
    u0: np.ndarray,
    interior_mask: np.ndarray,
    max_iter: int,
) -> np.ndarray:
    # Fixed-iteration helper (no atol), as requested in task 8.
    d_u_old = cuda.to_device(u0)
    d_u_new = cuda.to_device(u0.copy())
    d_interior = cuda.to_device(interior_mask)

    threads = (16, 16)
    blocks = (
        (interior_mask.shape[0] + threads[0] - 1) // threads[0],
        (interior_mask.shape[1] + threads[1] - 1) // threads[1],
    )

    for _ in range(max_iter):
        jacobi_step_kernel[blocks, threads](d_u_old, d_u_new, d_interior)
        d_u_old, d_u_new = d_u_new, d_u_old

    cuda.synchronize()
    return d_u_old.copy_to_host()


def summary_stats(u: np.ndarray, interior_mask: np.ndarray) -> dict[str, float]:
    u_interior = u[1:-1, 1:-1][interior_mask]
    return {
        "mean_temp": float(u_interior.mean()),
        "std_temp": float(u_interior.std()),
        "pct_above_18": float(np.sum(u_interior > 18) / u_interior.size * 100.0),
        "pct_below_15": float(np.sum(u_interior < 15) / u_interior.size * 100.0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 8: Numba CUDA Jacobi")
    parser.add_argument("--n-buildings", type=int, default=20)
    parser.add_argument("--max-iter", type=int, default=20_000)
    parser.add_argument("--load-dir", type=str, default=LOAD_DIR_DEFAULT)
    parser.add_argument("--csv-out", type=str, default="task8_results.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(join(args.load_dir, "building_ids.txt"), "r", encoding="utf-8") as f:
        building_ids = f.read().splitlines()[: args.n_buildings]

    # Warm-up kernel compile.
    warm_u = np.zeros((514, 514), dtype=np.float64)
    warm_mask = np.zeros((512, 512), dtype=np.bool_)
    jacobi_cuda_fixed_iter(warm_u, warm_mask, 1)

    t0 = time.perf_counter()
    rows: list[dict[str, float | str]] = []
    for bid in building_ids:
        u0, interior = load_data(args.load_dir, bid)
        u = jacobi_cuda_fixed_iter(u0, interior, args.max_iter)
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
