#!/usr/bin/env python3
"""Generate two small synthetic snapshots (300 x 32) for the vecdiff quickstart.

Usage:
    python scripts/make_demo_snapshots.py TARGET_DIR

Creates TARGET_DIR/snapA (blue) and TARGET_DIR/snapB (green) in the native
snapshot format. snapB simulates a sloppy re-embedding:

  * ids chunk_0000..0029 (all in ``src/auth/``) got heavy noise -> their
    neighborhoods collapse (N1 should flag them, concentrated in src/auth)
  * 4 near-duplicate pairs were injected (N4 should find them)
  * one vector was scaled to an extreme norm (N2 should flag it)
  * 5 ids were renamed (removed + added) to exercise shared-id logic

Deterministic: fixed seed (default 42), so everyone gets the same demo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    from vecdiff.snapshot import write_native_snapshot
except ImportError:  # allow running from a source checkout without install
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from vecdiff.snapshot import write_native_snapshot

MODULES = ("auth", "api", "core", "ui", "data", "tools")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path, help="output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=300, help="vectors per snapshot")
    parser.add_argument("--dim", type=int, default=32)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    n, dim = args.n, args.dim

    ids = [f"chunk_{i:04d}" for i in range(n)]
    paths = [
        f"src/{MODULES[min(i // (n // len(MODULES)), len(MODULES) - 1)]}/file_{i:04d}.py"
        for i in range(n)
    ]
    base = rng.standard_normal((n, dim)).astype(np.float32)
    created_a = "2026-09-01T00:00:00+00:00"
    created_b = "2026-09-05T00:00:00+00:00"

    write_native_snapshot(
        args.target / "snapA",
        ids=ids,
        vectors=base,
        model="demo-model-v1",
        chunk_paths=paths,
        created_at=created_a,
    )

    # --- snapB: a sloppy re-embedding of the same chunks --------------------
    noisy = base + rng.standard_normal(base.shape).astype(np.float32) * 0.02

    # 1) heavy damage in one directory region (src/auth = first n//6 ids)
    damaged = max(30, n // 10)
    noisy[:damaged] = base[:damaged] + (
        rng.standard_normal((damaged, dim)).astype(np.float32) * 1.5
    )

    # 2) near-duplicate pairs (cosine ~1.0)
    for a, b in ((100, 101), (150, 151), (210, 211), (250, 251)):
        if b < n:
            noisy[b] = base[a] + rng.standard_normal(dim).astype(np.float32) * 1e-4

    # 3) one extreme-norm outlier (same direction, huge magnitude)
    noisy[200] = base[200] * 8.0

    # 4) id churn: last 5 ids replaced by 5 new ones
    keep = n - 5
    ids_b = ids[:keep] + [f"chunk_new_{i:03d}" for i in range(5)]
    vecs_b = np.vstack(
        [
            noisy[:keep],
            rng.standard_normal((5, dim)).astype(np.float32),
        ]
    )
    paths_b = paths[:keep] + [f"src/data/file_new_{i:03d}.py" for i in range(5)]

    write_native_snapshot(
        args.target / "snapB",
        ids=ids_b,
        vectors=vecs_b,
        model="demo-model-v2",
        chunk_paths=paths_b,
        created_at=created_b,
    )

    print(f"wrote {args.target}/snapA and {args.target}/snapB ({n} x {dim} each)")
    print("try:")
    print(f"  vecdiff {args.target}/snapA {args.target}/snapB --gate")
    print("  vecdiff {a} {b} --full --json r.json --markdown r.md".format(
        a=f"{args.target}/snapA", b=f"{args.target}/snapB"
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
