"""FAISS adapter integration tests — run only when faiss is installed.

CI installs no faiss, so these skip there (importorskip). Locally:
``uv pip install faiss-cpu`` (the optional [faiss] extra).
"""

from __future__ import annotations

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from vecdiff.errors import SnapshotError
from vecdiff.snapshot import load_snapshot

from conftest import make_ids


def test_faiss_flat_round_trip(tmp_path, rng):
    vecs = np.ascontiguousarray(rng.standard_normal((40, 16)).astype(np.float32))
    index = faiss.IndexFlatL2(16)
    index.add(vecs)
    path = tmp_path / "snap.faiss"
    faiss.write_index(index, str(path))

    snap = load_snapshot(path)
    assert snap.adapter == "faiss"
    assert snap.n == 40
    assert snap.dim == 16
    assert np.array_equal(snap.vectors, vecs)
    assert snap.ids == [str(i) for i in range(40)]  # positional ids (caveat)
    assert snap.model == "unknown"
    assert any("positional" in note for note in snap.notes)


def test_faiss_empty_index(tmp_path):
    index = faiss.IndexFlatIP(8)
    path = tmp_path / "empty.faiss"
    faiss.write_index(index, str(path))
    snap = load_snapshot(path)
    assert snap.n == 0
    assert snap.dim == 8


def test_faiss_non_finite_rejected(tmp_path):
    vecs = np.ascontiguousarray(
        np.random.default_rng(3).standard_normal((5, 8)).astype(np.float32)
    )
    vecs[2, 1] = np.float32("nan")
    index = faiss.IndexFlatL2(8)
    index.add(vecs)
    path = tmp_path / "bad.faiss"
    faiss.write_index(index, str(path))
    with pytest.raises(SnapshotError, match="non-finite"):
        load_snapshot(path)
