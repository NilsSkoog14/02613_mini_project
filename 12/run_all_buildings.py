from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
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
def jacobi_numba_cpu(u0: np.ndarray, interior_mask: np.ndarray, max_iter: int, atol: float) -> np.ndarray:
    u_old = u0.copy()
    u_new = u0.copy()
    ny, nx = u0.shape

    for _ in range(max_iter):
        max_delta = 0.0
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                if interior_mask[i - 1, j - 1]:
                    new_val = 0.25 * (
                        u_old[i, j - 1] + u_old[i, j + 1] + u_old[i - 1, j] + u_old[i + 1, j]
                    )
                    delta = abs(new_val - u_old[i, j])
                    if delta > max_delta:
                        max_delta = delta
                    u_new[i, j] = new_val
                else:
                    u_new[i, j] = u_old[i, j]
        if max_delta < atol:
            return u_new
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


def _worker_init(max_iter: int, atol: float) -> None:
    global _WORKER_MAX_ITER, _WORKER_ATOL
    _WORKER_MAX_ITER = max_iter
    _WORKER_ATOL = atol
    warm_u = np.zeros((514, 514), dtype=np.float64)
    warm_mask = np.zeros((512, 512), dtype=np.bool_)
    jacobi_numba_cpu(warm_u, warm_mask, 1, 1e-4)


def _process_building(args: tuple[str, str]) -> dict[str, float | str]:
    load_dir, bid = args
    u0, interior = load_data(load_dir, bid)
    u = jacobi_numba_cpu(u0, interior, _WORKER_MAX_ITER, _WORKER_ATOL)
    return {"building_id": bid, **summary_stats(u, interior)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 12: process all floorplans")
    parser.add_argument("--max-iter", type=int, default=20_000)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--load-dir", type=str, default=LOAD_DIR_DEFAULT)
    parser.add_argument("--csv-out", type=str, default="all_buildings_results.csv")
    parser.add_argument("--n-workers", type=int, default=1,
                        help="Number of parallel worker processes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(join(args.load_dir, "building_ids.txt"), "r", encoding="utf-8") as f:
        building_ids = f.read().splitlines()

    n_workers = min(args.n_workers, len(building_ids))
    print(f"Using {n_workers} worker(s)", flush=True)

    worker_args = [(args.load_dir, bid) for bid in building_ids]

    t0 = time.perf_counter()
    with mp.Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(args.max_iter, args.atol),
    ) as pool:
        rows = []
        for idx, row in enumerate(pool.imap(_process_building, worker_args), start=1):
            rows.append(row)
            if idx % 200 == 0:
                print(f"{idx}/{len(building_ids)} buildings complete", flush=True)
    elapsed = time.perf_counter() - t0

    fields = ["building_id", "mean_temp", "std_temp", "pct_above_18", "pct_below_15"]
    with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Processed {len(building_ids)} buildings in {elapsed:.3f} s", flush=True)
    print(f"CSV written to {args.csv_out}", flush=True)


if __name__ == "__main__":
    main()
