"""Snapshot adapter + metadata validation tests."""

from __future__ import annotations

import json
import sqlite3
import struct

import numpy as np
import pytest

from vecdiff.errors import SnapshotError
from vecdiff.snapshot import detect_format, load_snapshot, write_native_snapshot

from conftest import make_ids, make_paths, write_dir


# ---------------------------------------------------------------------------
# native
# ---------------------------------------------------------------------------


def test_native_dir_round_trip(tmp_path, rng):
    ids = make_ids(20)
    vecs = rng.standard_normal((20, 8)).astype(np.float32)
    out = write_dir(tmp_path / "snap", ids, vecs, model="m1")
    snap = load_snapshot(out)
    assert snap.ids == ids
    assert snap.model == "m1"
    assert snap.dim == 8
    assert snap.n == 20
    assert np.array_equal(snap.vectors, vecs)
    assert snap.paths is not None and len(snap.paths) == 20
    assert snap.created_at is not None
    assert snap.adapter == "native"


def test_native_npz_single_file(tmp_path, rng):
    ids = make_ids(10)
    vecs = rng.standard_normal((10, 4)).astype(np.float32)
    meta = {
        "ids": ids,
        "model": "m",
        "dim": 4,
        "chunk_paths": [f"p/{i}.py" for i in range(10)],
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    path = tmp_path / "snap.npz"
    np.savez(
        path,
        vectors=vecs,
        ids=np.array(ids),
        meta_json=np.array(json.dumps(meta)),
    )
    snap = load_snapshot(path)
    assert snap.ids == ids
    assert np.array_equal(snap.vectors, vecs)
    assert snap.adapter == "native"


def test_bare_npy_rejected(tmp_path, rng):
    np.save(tmp_path / "vectors.npy", rng.standard_normal((5, 4)))
    with pytest.raises(SnapshotError, match="not a self-describing snapshot"):
        load_snapshot(tmp_path / "vectors.npy")


def test_dir_missing_vectors_file(tmp_path):
    (tmp_path / "snap").mkdir()
    (tmp_path / "snap" / "meta.json").write_text("{}")
    with pytest.raises(SnapshotError, match="no vectors file"):
        load_snapshot(tmp_path / "snap")


def test_dir_missing_meta(tmp_path, rng):
    (tmp_path / "snap").mkdir()
    np.save(tmp_path / "snap" / "vectors.npy", rng.standard_normal((3, 2)))
    with pytest.raises(SnapshotError, match="no meta.json"):
        load_snapshot(tmp_path / "snap")


# ---------------------------------------------------------------------------
# metadata validation errors
# ---------------------------------------------------------------------------


def _write_and_expect_error(tmp_path, meta, vectors, match):
    d = tmp_path / "snap"
    d.mkdir()
    (d / "meta.json").write_text(json.dumps(meta))
    np.save(d / "vectors.npy", vectors)
    with pytest.raises(SnapshotError, match=match):
        load_snapshot(d)


def test_validation_missing_required_keys(tmp_path, rng):
    _write_and_expect_error(
        tmp_path,
        {"ids": ["a"]},
        rng.standard_normal((1, 3)),
        r"missing required keys",
    )


def test_validation_dim_mismatch_meta_vs_vectors(tmp_path, rng):
    _write_and_expect_error(
        tmp_path,
        {"ids": ["a", "b"], "model": "m", "dim": 7},
        rng.standard_normal((2, 3)),
        r"metadata dim=7 but vectors have width 3",
    )


def test_validation_ids_length_mismatch(tmp_path, rng):
    _write_and_expect_error(
        tmp_path,
        {"ids": ["a", "b"], "model": "m", "dim": 3},
        rng.standard_normal((3, 3)),
        r"2 ids but vectors has 3 rows",
    )


def test_validation_duplicate_ids(tmp_path, rng):
    _write_and_expect_error(
        tmp_path,
        {"ids": ["a", "a", "b"], "model": "m", "dim": 3},
        rng.standard_normal((3, 3)),
        r"duplicate",
    )


def test_validation_non_string_ids(tmp_path, rng):
    _write_and_expect_error(
        tmp_path,
        {"ids": [1, 2], "model": "m", "dim": 3},
        rng.standard_normal((2, 3)),
        r"list of strings",
    )


def test_validation_chunk_paths_length_mismatch(tmp_path, rng):
    _write_and_expect_error(
        tmp_path,
        {
            "ids": ["a", "b"],
            "model": "m",
            "dim": 3,
            "chunk_paths": ["only-one"],
        },
        rng.standard_normal((2, 3)),
        r"chunk_paths",
    )


def test_float64_cast_with_note(tmp_path, rng):
    # write_native_snapshot casts to float32 before writing, so exercise the
    # loader directly with a float64 vectors file on disk.
    d = tmp_path / "snap"
    d.mkdir()
    np.save(d / "vectors.npy", rng.standard_normal((6, 4)).astype(np.float64))
    ids = make_ids(6)
    (d / "meta.json").write_text(
        json.dumps({"ids": ids, "model": "m", "dim": 4, "chunk_paths": make_paths(ids)}),
        encoding="utf-8",
    )
    snap = load_snapshot(d)
    assert snap.vectors.dtype == np.float32
    assert any("cast to float32" in n for n in snap.notes)


def test_empty_snapshot_accepted(tmp_path):
    d = write_dir(tmp_path / "snap", [], np.zeros((0, 4), dtype=np.float32))
    snap = load_snapshot(d)
    assert snap.n == 0
    assert any("empty" in n for n in snap.notes)


# ---------------------------------------------------------------------------
# sqlite
# ---------------------------------------------------------------------------


def _write_sqlite(path, ids, vecs, meta_rows=None):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE chunks(id TEXT PRIMARY KEY, vec BLOB)")
    for cid, vec in zip(ids, vecs):
        blob = np.asarray(vec, dtype="<f4").tobytes()
        con.execute("INSERT INTO chunks(id, vec) VALUES (?, ?)", (cid, blob))
    if meta_rows:
        con.execute("CREATE TABLE meta(key TEXT, value TEXT)")
        con.executemany("INSERT INTO meta(key, value) VALUES (?, ?)", meta_rows)
    con.commit()
    con.close()


def test_sqlite_round_trip(tmp_path, rng):
    ids = make_ids(15)
    vecs = rng.standard_normal((15, 6)).astype(np.float32)
    db = tmp_path / "index.db"
    _write_sqlite(db, ids, vecs, meta_rows=[("model", "sqlite-model"),
                                           ("created_at", "2026-08-01")])
    snap = load_snapshot(db)
    assert snap.adapter == "sqlite"
    assert snap.ids == ids  # ordered by id
    assert snap.model == "sqlite-model"
    assert snap.created_at == "2026-08-01"
    assert snap.dim == 6
    assert np.array_equal(snap.vectors, vecs)
    assert snap.paths is None  # schema has no path column


def test_sqlite_missing_chunks_table(tmp_path):
    db = tmp_path / "bad.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE other(x)")
    con.commit()
    con.close()
    with pytest.raises(SnapshotError, match="no 'chunks' table"):
        load_snapshot(db)


def test_sqlite_inconsistent_blob_lengths(tmp_path):
    db = tmp_path / "bad.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE chunks(id TEXT PRIMARY KEY, vec BLOB)")
    con.execute("INSERT INTO chunks VALUES ('a', ?)", (np.zeros(4, "<f4").tobytes(),))
    con.execute("INSERT INTO chunks VALUES ('b', ?)", (np.zeros(8, "<f4").tobytes(),))
    con.commit()
    con.close()
    with pytest.raises(SnapshotError, match="inconsistent vec blob"):
        load_snapshot(db)


def test_sqlite_blob_not_multiple_of_four(tmp_path):
    db = tmp_path / "bad.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE chunks(id TEXT PRIMARY KEY, vec BLOB)")
    con.execute("INSERT INTO chunks VALUES ('a', ?)", (b"\x01\x02\x03",))
    con.commit()
    con.close()
    with pytest.raises(SnapshotError, match="multiple of 4"):
        load_snapshot(db)


# ---------------------------------------------------------------------------
# faiss adapter (import-guarded)
# ---------------------------------------------------------------------------


def test_faiss_graceful_without_faiss(tmp_path):
    pytest.importorskip = getattr(pytest, "importorskip")
    faiss = None
    try:
        import faiss as faiss_mod  # noqa: F401
        faiss = faiss_mod
    except Exception:
        faiss = None
    target = tmp_path / "idx.faiss"
    target.write_bytes(b"not-a-real-index")
    if faiss is not None:
        pytest.skip("faiss is installed; graceful-degradation path not exercised")
    with pytest.raises(SnapshotError, match="faiss-cpu"):
        load_snapshot(target)


# ---------------------------------------------------------------------------
# format detection
# ---------------------------------------------------------------------------


def test_detect_format(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    assert detect_format(d) == "native"
    assert detect_format(tmp_path / "x.npz") == "native"
    assert detect_format(tmp_path / "x.db") == "sqlite"
    assert detect_format(tmp_path / "x.sqlite3") == "sqlite"
    assert detect_format(tmp_path / "x.index") == "faiss"
    assert detect_format(tmp_path / "x.faiss") == "faiss"
    with pytest.raises(SnapshotError, match="cannot infer"):
        detect_format(tmp_path / "x.weird")


def test_missing_path(tmp_path):
    with pytest.raises(SnapshotError, match="does not exist"):
        load_snapshot(tmp_path / "nope")


def test_sqlite_path_with_special_characters(tmp_path):
    d = tmp_path / "we?ird #dir"
    d.mkdir()
    con = sqlite3.connect(d / "snap.db")
    con.execute("CREATE TABLE chunks(id TEXT PRIMARY KEY, vec BLOB)")
    for i in range(4):
        con.execute(
            "INSERT INTO chunks VALUES (?,?)",
            (f"c{i}", np.eye(4, dtype=np.float32)[i].tobytes()),
        )
    con.commit()
    con.close()
    snap = load_snapshot(d / "snap.db")
    assert snap.n == 4
