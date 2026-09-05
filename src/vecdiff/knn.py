"""Exact brute-force cosine kNN over numpy arrays.

Memory safety: similarity matrices are computed in blocked matrix
multiplications. The per-block score matrix is capped at ~64 MB regardless
of index size, so a 100k x 1024 snapshot never allocates a 100k x 100k
matrix. Time is O(queries * n * dim) — exact, not ANN.

Determinism: fixed block sizes; ties among the *selected* top-k
candidates are broken by a stable sort (lower index first). Which of
several exactly-tied values at the k-th boundary gets selected is not
guaranteed — same inputs on the same numpy build give the same output,
but boundary ties may resolve differently across numpy versions.
"""

from __future__ import annotations

import numpy as np

DEFAULT_BLOCK_BYTES = 64 * 1024 * 1024  # ~64 MB per similarity block


def block_rows(n: int, block_bytes: int = DEFAULT_BLOCK_BYTES) -> int:
    """Rows per block so that a (rows, n) float32 block stays <= block_bytes."""
    if n <= 0:
        return 1
    return max(1, int(block_bytes // (4 * n)))


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Row-normalize to unit length. Zero-norm rows stay zero (cosine 0)."""
    v = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    safe = np.where(norms > 0.0, norms, np.float32(1.0))
    return v / safe


def topk_cosine(
    vectors: np.ndarray,
    query_indices: np.ndarray | list[int],
    k: int,
    *,
    block_bytes: int = DEFAULT_BLOCK_BYTES,
    unit: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-k cosine neighbors for the given query rows.

    The candidate pool is *all* rows of ``vectors`` (the full index), and
    each query's own row is excluded from its result. Pass ``unit`` (the
    pre-normalized candidate matrix, e.g. ``Snapshot.unit_vectors()``) to
    skip a redundant full-array normalization.

    Returns ``(neighbor_indices (q, k_eff) int64, similarities (q, k_eff)
    float32)`` sorted by similarity descending. Ties within the selected
    k are broken stably (lower index first); which exactly-tied value at
    the k-th boundary is kept is not guaranteed. ``k_eff = min(k, n - 1)``;
    if ``n <= 1`` the arrays have zero columns.
    """
    v = unit if unit is not None else l2_normalize(vectors)
    n = v.shape[0]
    q_idx = np.asarray(query_indices, dtype=np.int64)
    q = v[q_idx]
    m = q.shape[0]
    k_eff = min(int(k), n - 1)
    if k_eff <= 0:
        return (
            np.zeros((m, 0), dtype=np.int64),
            np.zeros((m, 0), dtype=np.float32),
        )

    out_idx = np.empty((m, k_eff), dtype=np.int64)
    out_sim = np.empty((m, k_eff), dtype=np.float32)
    step = block_rows(n, block_bytes)
    for start in range(0, m, step):
        end = min(m, start + step)
        sims = q[start:end] @ v.T  # (b, n) float32
        # exclude each query from its own candidate list
        sims[np.arange(end - start), q_idx[start:end]] = -np.inf
        part = np.argpartition(-sims, k_eff - 1, axis=1)[:, :k_eff]
        cand = np.take_along_axis(sims, part, axis=1)
        order = np.argsort(-cand, axis=1, kind="stable")
        out_idx[start:end] = np.take_along_axis(part, order, axis=1)
        out_sim[start:end] = np.take_along_axis(cand, order, axis=1)
    return out_idx, out_sim
