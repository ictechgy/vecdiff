"""Shared fixtures/helpers for vecdiff tests. All data is deterministic."""

from __future__ import annotations

import zlib

import numpy as np
import pytest

from vecdiff.snapshot import Snapshot, write_native_snapshot


def make_ids(n: int) -> list[str]:
    return [f"chunk_{i:04d}" for i in range(n)]


def make_paths(ids: list[str]) -> list[str]:
    def group(id_: str) -> int:
        # ids are usually zero-padded numbers, but a few tests use
        # arbitrary strings — fall back to a stable hash bucket.
        try:
            return int(id_[-4:]) // 10
        except ValueError:
            return zlib.crc32(id_.encode()) % 20

    return [f"src/mod_{group(i)}/file_{i}.py" for i in ids]


def make_snapshot(
    ids: list[str],
    vectors: np.ndarray,
    model: str = "test-model",
    paths: list[str] | None = "auto",
) -> Snapshot:
    vec = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
    if paths == "auto":
        paths = make_paths(ids)
    return Snapshot(
        ids=list(ids),
        vectors=vec,
        model=model,
        dim=vec.shape[1],
        paths=paths,
        created_at="2026-09-05T00:00:00+00:00",
        source="<memory>",
        adapter="native",
        notes=[],
    )


def write_dir(
    path,
    ids: list[str],
    vectors: np.ndarray,
    model: str = "test-model",
    paths: list[str] | None = None,
    created_at: str | None = "2026-09-05T00:00:00+00:00",
):
    vec = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
    return write_native_snapshot(
        path,
        ids=ids,
        vectors=vec,
        model=model,
        chunk_paths=paths if paths is not None else make_paths(ids),
        created_at=created_at,
    )


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)
