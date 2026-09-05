"""Snapshot loading (native, SQLite, FAISS) and metadata validation.

A *snapshot* is a model-agnostic dump of an embedding index: chunk ids +
float32 vectors + metadata (model, dim, chunk paths, created_at). vecdiff
never needs the original vector DB and never touches a network.

Adapters
--------
native  (required)
  * directory: ``vectors.npy`` (or ``vectors.npz`` with a ``vectors`` key)
    plus ``meta.json``
  * single self-describing ``.npz`` file with keys ``vectors`` (float32
    ``(n, dim)``), ``ids`` (string array) and ``meta_json`` (0-d string
    array containing the same JSON metadata)

sqlite  (required, stdlib only)
  * a SQLite database with table ``chunks(id TEXT PRIMARY KEY, vec BLOB)``
    where ``vec`` is a float32 little-endian blob of length ``4 * dim``.
    An optional ``meta(key TEXT, value TEXT)`` table may provide ``model``
    and ``created_at``.

jsonl  (required, stdlib only — the universal escape hatch)
  * ``.jsonl`` / ``.ndjson`` (optionally ``.gz``): one JSON object per line
    with keys ``id`` (string), ``vector`` (list of floats) and optionally
    ``path`` (chunk file path). A sidecar ``meta.json`` next to the file may
    supply ``model`` / ``created_at``; without it they are reported as
    unknown. Any vector DB can produce this with a few lines of its own
    client code — see ``docs/export_recipes.md``.

faiss  (optional, import-guarded)
  * a FAISS index file (``.index`` / ``.faiss``). If the ``faiss`` package
    is not installed the adapter fails with a clear, actionable error
    instead of a crash — everything else in vecdiff works without faiss.

All loaders return a validated :class:`Snapshot` with C-contiguous float32
vectors and unique string ids. Anything inconsistent (missing metadata
keys, length mismatches, duplicate ids, dim mismatches) raises
:class:`~vecdiff.errors.SnapshotError`.
"""

from __future__ import annotations

import json
import posixpath
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .errors import SnapshotError

NATIVE_VEC_NAMES = ("vectors.npy", "vectors.npz")
SQLITE_SUFFIXES = {".db", ".sqlite", ".sqlite3"}
FAISS_SUFFIXES = {".index", ".faiss"}
NATIVE_SUFFIXES = {".npz", ".npy"}
JSONL_SUFFIXES = {".jsonl", ".ndjson"}
JSONL_CHUNK_ROWS = 8192  # rows buffered as Python floats before float32 conversion


@dataclass
class Snapshot:
    """A validated, model-agnostic dump of one embedding index."""

    ids: list[str]
    vectors: np.ndarray  # (n, dim), float32, C-contiguous
    model: str
    dim: int
    paths: list[str] | None  # chunk file paths, aligned with ids; None if unknown
    created_at: str | None
    source: str  # filesystem path this was loaded from
    adapter: str  # "native" | "jsonl" | "sqlite" | "faiss" | "memory"
    notes: list[str] = field(default_factory=list)  # honest provenance notes
    _unit: np.ndarray | None = field(default=None, repr=False, compare=False)

    @property
    def n(self) -> int:
        return len(self.ids)

    def unit_vectors(self) -> np.ndarray:
        """L2-normalized vectors, computed once and cached.

        N1 and N4 both need normalized copies; normalizing is a full (n, dim)
        allocation per call, so at 100k x 1024 the cache saves ~400 MB per
        redundant pass.
        """
        if self._unit is None:
            from .knn import l2_normalize

            self._unit = l2_normalize(self.vectors)
        return self._unit

    def index_of(self) -> dict[str, int]:
        return {id_: i for i, id_ in enumerate(self.ids)}

    def summary(self) -> dict:
        return {
            "path": self.source,
            "adapter": self.adapter,
            "model": self.model,
            "dim": self.dim,
            "n": self.n,
            "created_at": self.created_at,
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# public entry points
# ---------------------------------------------------------------------------


def load_snapshot(path: str | Path, fmt: str = "auto") -> Snapshot:
    """Load a snapshot from ``path``.

    ``fmt`` is one of ``auto`` (infer from path), ``native``, ``jsonl``,
    ``sqlite`` or ``faiss``.
    """
    p = Path(path)
    if not p.exists():
        raise SnapshotError(f"snapshot path does not exist: {p}")

    chosen = fmt if fmt != "auto" else detect_format(p)
    if chosen == "native":
        return _load_native(p)
    if chosen == "jsonl":
        return _load_jsonl(p)
    if chosen == "sqlite":
        return _load_sqlite(p)
    if chosen == "faiss":
        return _load_faiss(p)
    raise SnapshotError(
        f"unknown input format {fmt!r} (expected auto|native|jsonl|sqlite|faiss)"
    )


def detect_format(p: Path) -> str:
    if p.is_dir():
        return "native"
    suffix = p.suffix.lower()
    if p.name.lower().endswith(".gz"):
        stem = p.name.lower()[: -len(".gz")]
        if any(stem.endswith(s) for s in JSONL_SUFFIXES):
            return "jsonl"
    if suffix in NATIVE_SUFFIXES:
        return "native"
    if suffix in JSONL_SUFFIXES:
        return "jsonl"
    if suffix in SQLITE_SUFFIXES:
        return "sqlite"
    if suffix in FAISS_SUFFIXES:
        return "faiss"
    raise SnapshotError(
        f"cannot infer snapshot format for '{p}': unknown extension "
        f"'{suffix}'. Supported: native directory or .npz, JSONL "
        f"({', '.join(sorted(JSONL_SUFFIXES))}, optionally .gz), SQLite "
        f"({', '.join(sorted(SQLITE_SUFFIXES))}), FAISS "
        f"({', '.join(sorted(FAISS_SUFFIXES))}). Pass --format to override."
    )


def write_native_snapshot(
    out_dir: str | Path,
    *,
    ids: list[str],
    vectors: np.ndarray,
    model: str,
    chunk_paths: list[str] | None = None,
    created_at: str | None = None,
    extra: dict | None = None,
) -> Path:
    """Write a native directory snapshot (vectors.npy + meta.json)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    vec = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
    meta: dict = {
        "ids": list(ids),
        "model": str(model),
        "dim": int(vec.shape[1]),
    }
    if chunk_paths is not None:
        meta["chunk_paths"] = [str(x) for x in chunk_paths]
    if created_at is not None:
        meta["created_at"] = str(created_at)
    if extra:
        for key, value in extra.items():
            meta.setdefault(key, value)
    # Validate through the same path as loading, so a writer bug can never
    # produce a snapshot vecdiff itself cannot read.
    _build_snapshot(meta, vec, source=str(out), adapter="native")
    np.save(out / "vectors.npy", vec)
    (out / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out


def snapshot_from_arrays(
    *,
    ids: list[str],
    vectors: np.ndarray,
    model: str,
    chunk_paths: list[str] | None = None,
    created_at: str | None = None,
    source: str = "<in-memory>",
) -> Snapshot:
    """Build a validated Snapshot directly from arrays, in memory.

    The programmatic entry point for "any vector DB": pull ids + vectors
    out of your database with its own client library and hand them to
    vecdiff without touching the filesystem. Validation is identical to
    the file adapters, so a bad build fails here, not mid-diff.
    """
    arrays = np.asarray(vectors)
    if arrays.ndim != 2:
        raise SnapshotError(
            f"vectors must be a 2-D (n, dim) array, got shape {arrays.shape}; "
            "pass e.g. np.zeros((0, dim)) for an empty snapshot"
        )
    meta: dict = {
        "ids": list(ids),
        "model": str(model),
        "dim": int(arrays.shape[1]),
    }
    if chunk_paths is not None:
        meta["chunk_paths"] = [str(x) for x in chunk_paths]
    if created_at is not None:
        meta["created_at"] = str(created_at)
    return _build_snapshot(meta, arrays, source=source, adapter="memory")


# ---------------------------------------------------------------------------
# native adapter
# ---------------------------------------------------------------------------


def _load_native(p: Path) -> Snapshot:
    if p.is_dir():
        return _load_native_dir(p)
    if p.suffix.lower() == ".npz":
        return _load_native_npz(p)
    raise SnapshotError(
        f"'{p}': a bare .npy file is not a self-describing snapshot. Use a "
        "directory containing vectors.npy + meta.json, or a single .npz with "
        "keys vectors/ids/meta_json."
    )


def _load_native_dir(p: Path) -> Snapshot:
    vec_path = next((p / name for name in NATIVE_VEC_NAMES if (p / name).is_file()), None)
    if vec_path is None:
        raise SnapshotError(
            f"native snapshot directory '{p}' has no vectors file; expected "
            f"one of {', '.join(NATIVE_VEC_NAMES)}"
        )
    meta_path = p / "meta.json"
    if not meta_path.is_file():
        raise SnapshotError(f"native snapshot directory '{p}' has no meta.json")
    vectors = _read_vectors_file(vec_path)
    meta = _read_json(meta_path)
    return _build_snapshot(meta, vectors, source=str(p), adapter="native")


def _load_native_npz(p: Path) -> Snapshot:
    try:
        with np.load(p, allow_pickle=False) as data:
            keys = set(data.files)
            missing = {"vectors", "meta_json"} - keys
            if missing:
                raise SnapshotError(
                    f"native .npz snapshot '{p}' is missing keys {sorted(missing)}; "
                    "expected vectors, meta_json (and optionally ids)"
                )
            vectors = np.ascontiguousarray(data["vectors"])
            meta = json.loads(str(data["meta_json"]))
            if "ids" in keys and isinstance(meta, dict) and "ids" in meta:
                npz_ids = [str(x) for x in data["ids"].tolist()]
                if npz_ids != [str(x) for x in meta["ids"]]:
                    raise SnapshotError(
                        f"native .npz snapshot '{p}': 'ids' array and metadata ids differ"
                    )
    except SnapshotError:
        raise
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise SnapshotError(f"could not read native .npz snapshot '{p}': {exc}") from exc
    return _build_snapshot(meta, vectors, source=str(p), adapter="native")


def _read_vectors_file(vec_path: Path) -> np.ndarray:
    suffix = vec_path.suffix.lower()
    try:
        if suffix == ".npy":
            arr = np.load(vec_path, allow_pickle=False)
        else:  # .npz
            with np.load(vec_path, allow_pickle=False) as data:
                if "vectors" not in data.files:
                    raise SnapshotError(
                        f"'{vec_path}' has no 'vectors' key; keys: {data.files}"
                    )
                arr = np.ascontiguousarray(data["vectors"])
    except SnapshotError:
        raise
    except (OSError, ValueError) as exc:
        raise SnapshotError(f"could not read vectors file '{vec_path}': {exc}") from exc
    return np.ascontiguousarray(arr)


def _read_json(meta_path: Path) -> dict:
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SnapshotError(f"could not read metadata '{meta_path}': {exc}") from exc
    return meta


# ---------------------------------------------------------------------------
# jsonl adapter (stdlib only — universal interchange for any vector DB)
# ---------------------------------------------------------------------------


def _load_jsonl(p: Path) -> Snapshot:
    """Load a JSONL snapshot: one ``{"id", "vector", "path"?}`` object per line.

    gz-compressed files are transparently decompressed. A sidecar
    ``meta.json`` (same stem, .json extension) may provide ``model`` /
    ``created_at``; without it they are reported as unknown.

    Memory: rows are converted to float32 in ``JSONL_CHUNK_ROWS``-sized
    chunks, so the Python-float intermediate never exceeds one chunk; the
    final concatenation transiently holds ~2x the float32 array.
    """
    if p.is_dir():
        raise SnapshotError(f"jsonl snapshot must be a file, got directory: {p}")
    import gzip

    meta: dict = {}
    # sidecar: strip .gz then .jsonl/.ndjson, append .meta.json
    # (snap.jsonl / snap.ndjson / snap.jsonl.gz all look for snap.meta.json)
    stem = p.name
    if stem.lower().endswith(".gz"):
        stem = stem[: -len(".gz")]
    stem = stem[: stem.rfind(".")] if "." in stem else stem
    sidecar = p.with_name(stem + ".meta.json")
    if sidecar.is_file():
        meta = _read_json(sidecar)
        if not isinstance(meta, dict):
            raise SnapshotError(f"sidecar metadata must be a JSON object ({sidecar})")

    ids: list[str] = []
    paths: list[str] = []
    chunks: list[np.ndarray] = []
    buf: list[list[float]] = []
    dim = 0
    opener = gzip.open if p.name.lower().endswith(".gz") else open
    try:
        with opener(p, "rt", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SnapshotError(
                        f"jsonl snapshot '{p}': line {lineno} is not valid JSON: {exc}"
                    ) from exc
                if not isinstance(obj, dict):
                    raise SnapshotError(
                        f"jsonl snapshot '{p}': line {lineno} must be a JSON object, "
                        f"got {type(obj).__name__}"
                    )
                missing = [key for key in ("id", "vector") if key not in obj]
                if missing:
                    raise SnapshotError(
                        f"jsonl snapshot '{p}': line {lineno} is missing key(s) "
                        f"{missing}; expected id, vector (and optionally path)"
                    )
                if not isinstance(obj["id"], str):
                    raise SnapshotError(
                        f"jsonl snapshot '{p}': line {lineno} 'id' must be a string, "
                        f"got {type(obj['id']).__name__}"
                    )
                vec = obj["vector"]
                if not isinstance(vec, list) or not all(
                    isinstance(x, (int, float)) and not isinstance(x, bool) for x in vec
                ):
                    raise SnapshotError(
                        f"jsonl snapshot '{p}': line {lineno} 'vector' must be a "
                        "list of numbers"
                    )
                if not vec:
                    raise SnapshotError(
                        f"jsonl snapshot '{p}': line {lineno} 'vector' is empty"
                    )
                path = obj.get("path")
                if path is not None and not isinstance(path, str):
                    raise SnapshotError(
                        f"jsonl snapshot '{p}': line {lineno} 'path' must be a string"
                    )
                row = [float(x) for x in vec]
                if dim == 0:
                    dim = len(row)
                elif len(row) != dim:
                    raise SnapshotError(
                        f"jsonl snapshot '{p}': line {lineno} has vector length "
                        f"{len(row)}, expected {dim} (inconsistent vector lengths)"
                    )
                ids.append(obj["id"])
                buf.append(row)
                paths.append(path if path is not None else "")
                if len(buf) >= JSONL_CHUNK_ROWS:
                    chunks.append(np.asarray(buf, dtype=np.float32))
                    buf.clear()
    except SnapshotError:
        raise
    except (OSError, gzip.BadGzipFile, EOFError, UnicodeDecodeError) as exc:
        raise SnapshotError(f"could not read jsonl snapshot '{p}': {exc}") from exc
    if buf:
        chunks.append(np.asarray(buf, dtype=np.float32))

    notes = [
        "jsonl adapter: model/created_at come from the optional sidecar "
        "<name>.meta.json (defaults: model 'unknown')."
    ]
    if not ids:
        notes.append("jsonl snapshot is empty (0 records).")
        return Snapshot(
            ids=[],
            vectors=np.zeros((0, 0), dtype=np.float32),
            model=str(meta.get("model", "unknown")),
            dim=0,
            paths=None,
            created_at=str(meta["created_at"]) if meta.get("created_at") else None,
            source=str(p),
            adapter="jsonl",
            notes=notes,
        )
    vectors = np.concatenate(chunks)
    built_meta: dict = {"ids": ids, "model": str(meta.get("model", "unknown")), "dim": dim}
    if created := meta.get("created_at"):
        built_meta["created_at"] = str(created)
    has_paths = any(x for x in paths)
    if has_paths:
        built_meta["chunk_paths"] = paths
    else:
        notes.append(
            "jsonl adapter: no 'path' fields found; N1 path concentration "
            "will be unavailable."
        )
    snap = _build_snapshot(
        built_meta, vectors, source=str(p), adapter="jsonl"
    )
    snap.notes = notes + snap.notes
    return snap


# ---------------------------------------------------------------------------
# sqlite adapter (stdlib sqlite3 only)
# ---------------------------------------------------------------------------


def _load_sqlite(p: Path) -> Snapshot:
    if p.is_dir():
        raise SnapshotError(f"sqlite snapshot must be a file, got directory: {p}")
    # as_uri() percent-encodes '?'/'#'/spaces etc.; appending '?mode=ro' to a
    # raw path would let those characters split the URI and silently open an
    # empty database.
    uri = p.resolve().as_uri() + "?mode=ro"
    try:
        con = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as exc:
        raise SnapshotError(f"could not open sqlite snapshot '{p}': {exc}") from exc
    try:
        tables = {
            row[0]
            for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "chunks" not in tables:
            raise SnapshotError(
                f"sqlite snapshot '{p}' has no 'chunks' table. Expected schema: "
                "chunks(id TEXT PRIMARY KEY, vec BLOB) with float32 "
                "little-endian vectors."
            )
        try:
            rows = con.execute("SELECT id, vec FROM chunks ORDER BY id").fetchall()
        except sqlite3.Error as exc:
            raise SnapshotError(
                f"sqlite snapshot '{p}': could not read chunks(id, vec): {exc}. "
                "Expected schema: chunks(id TEXT PRIMARY KEY, vec BLOB)."
            ) from exc
        meta: dict = {}
        if "meta" in tables:
            try:
                meta = {
                    str(k): ("" if v is None else str(v))
                    for k, v in con.execute("SELECT key, value FROM meta")
                }
            except sqlite3.Error:
                meta = {}
    finally:
        con.close()

    notes = [
        "sqlite adapter: ids come from chunks.id ordered by id; "
        "model/created_at come from the optional meta(key, value) table "
        "(defaults: 'unknown')."
    ]
    if not rows:
        notes.append("sqlite snapshot is empty (0 rows in chunks).")
        return Snapshot(
            ids=[],
            vectors=np.zeros((0, 0), dtype=np.float32),
            model=meta.get("model", "unknown"),
            dim=0,
            paths=None,
            created_at=meta.get("created_at"),
            source=str(p),
            adapter="sqlite",
            notes=notes,
        )

    dim = -1
    ids: list[str] = []
    vecs: list[np.ndarray] = []
    for row_id, blob in rows:
        if not isinstance(row_id, str):
            raise SnapshotError(
                f"sqlite snapshot '{p}': chunks.id must be TEXT, got "
                f"{type(row_id).__name__} ({row_id!r})"
            )
        if not isinstance(blob, (bytes, bytearray, memoryview)):
            raise SnapshotError(
                f"sqlite snapshot '{p}': chunks.vec for id {row_id!r} is not a BLOB"
            )
        buf = bytes(blob)
        if len(buf) % 4 != 0:
            raise SnapshotError(
                f"sqlite snapshot '{p}': vec blob for id {row_id!r} has length "
                f"{len(buf)}, not a multiple of 4 (float32 expected)"
            )
        row_dim = len(buf) // 4
        if dim == -1:
            dim = row_dim
        elif row_dim != dim:
            raise SnapshotError(
                f"sqlite snapshot '{p}': inconsistent vec blob lengths "
                f"({dim} vs {row_dim} floats) — id {row_id!r}"
            )
        ids.append(row_id)
        vecs.append(np.frombuffer(buf, dtype="<f4"))
    vectors = np.ascontiguousarray(np.stack(vecs).astype(np.float32, copy=False))
    _require_finite(vectors, ids, str(p))
    if "meta" not in tables:
        notes.append("sqlite snapshot has no meta table; model reported as 'unknown'.")
    if "chunk_paths" not in meta:
        notes.append(
            "sqlite adapter: no per-chunk path metadata in the chunks schema; "
            "N1 path concentration will be unavailable unless ids encode paths."
        )
    return Snapshot(
        ids=ids,
        vectors=vectors,
        model=meta.get("model", "unknown"),
        dim=dim,
        paths=None,
        created_at=meta.get("created_at"),
        source=str(p),
        adapter="sqlite",
        notes=notes,
    )


# ---------------------------------------------------------------------------
# faiss adapter (optional, import-guarded)
# ---------------------------------------------------------------------------


def _load_faiss(p: Path) -> Snapshot:
    if p.is_dir():
        raise SnapshotError(f"faiss snapshot must be a file, got directory: {p}")
    try:
        import faiss  # type: ignore[import-untyped]
    except Exception as exc:  # ImportError or broken install
        raise SnapshotError(
            f"FAISS snapshot '{p}' requested but the 'faiss' package is not "
            "usable. Install it with: pip install faiss-cpu  (or pip install "
            "'vecdiff[faiss]'), or export the index to the native/sqlite "
            "snapshot format instead."
        ) from exc
    try:
        index = faiss.read_index(str(p))
    except Exception as exc:
        raise SnapshotError(f"could not read FAISS index '{p}': {exc}") from exc
    n, d = int(index.ntotal), int(index.d)
    faiss_ids = [str(i) for i in range(n)]
    if n == 0:
        vectors = np.zeros((0, d), dtype=np.float32)
    else:
        try:
            vectors = np.ascontiguousarray(
                np.asarray(index.reconstruct_n(0, n), dtype=np.float32)
            )
        except Exception as exc:
            raise SnapshotError(
                f"FAISS index '{p}' does not support vector reconstruction "
                "(only flat-style indexes whose vectors can be reconstructed "
                "can be loaded). Export to the native/sqlite format instead."
            ) from exc
        _require_finite(vectors, faiss_ids, str(p))
    return Snapshot(
        ids=faiss_ids,
        vectors=vectors,
        model="unknown",
        dim=d,
        paths=None,
        created_at=None,
        source=str(p),
        adapter="faiss",
        notes=[
            "faiss adapter: ids are positional (0..n-1); FAISS indexes carry "
            "no model/created_at/path metadata, so those are reported as unknown."
        ],
    )


# ---------------------------------------------------------------------------
# shared validation
# ---------------------------------------------------------------------------


def _require_finite(vectors: np.ndarray, ids: list[str], source: str) -> None:
    """Reject NaN/inf embeddings.

    NaN never compares equal, so cosine top-k over a poisoned row returns
    an arbitrary neighbor set — the worst pipeline bug would render as a
    clean report. Fail at load instead.
    """
    row_ok = np.isfinite(vectors).all(axis=1)
    if not row_ok.all():
        bad = np.flatnonzero(~row_ok)
        shown = [ids[i] for i in bad[:5]]
        raise SnapshotError(
            f"{len(bad)} vector(s) contain non-finite (NaN/inf) values "
            f"after float32 cast, first few ids: {shown} ({source}). Fix or "
            "drop these rows in the export — their neighbor sets would be "
            "arbitrary."
        )


def _build_snapshot(
    meta: dict,
    vectors: np.ndarray,
    *,
    source: str,
    adapter: str,
) -> Snapshot:
    """Validate raw metadata + vectors and build a Snapshot. Shared by all
    native loaders and by write_native_snapshot."""
    if not isinstance(meta, dict):
        raise SnapshotError(f"snapshot metadata must be a JSON object ({source})")

    missing = [key for key in ("ids", "model", "dim") if key not in meta]
    if missing:
        raise SnapshotError(
            f"snapshot metadata is missing required keys {missing} "
            f"(need ids, model, dim; source: {source})"
        )

    ids = meta["ids"]
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        raise SnapshotError(f"'ids' must be a list of strings ({source})")
    dup_counter = Counter(ids)
    dupes = sorted(x for x, c in dup_counter.items() if c > 1)
    if dupes:
        raise SnapshotError(
            f"'ids' contains {len(dupes)} duplicate value(s), first few: {dupes[:5]} "
            f"({source})"
        )

    dim = meta["dim"]
    if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
        raise SnapshotError(f"'dim' must be a positive integer, got {dim!r} ({source})")
    if not isinstance(meta["model"], str):
        raise SnapshotError(f"'model' must be a string, got {meta['model']!r} ({source})")

    vectors = np.asarray(vectors)
    if vectors.ndim == 1 and dim == 1:
        vectors = vectors.reshape(-1, 1)
    if vectors.ndim != 2:
        raise SnapshotError(
            f"vectors must be a 2-D (n, dim) array, got shape {vectors.shape} ({source})"
        )
    if vectors.shape[0] != len(ids):
        raise SnapshotError(
            f"{len(ids)} ids but vectors has {vectors.shape[0]} rows ({source})"
        )
    if vectors.shape[1] != dim:
        raise SnapshotError(
            f"metadata dim={dim} but vectors have width {vectors.shape[1]} ({source})"
        )

    notes: list[str] = []
    if vectors.dtype != np.float32:
        notes.append(
            f"vectors stored as {vectors.dtype}, cast to float32 on load."
        )
    vectors = np.ascontiguousarray(vectors.astype(np.float32, copy=False))
    _require_finite(vectors, list(ids), source)

    paths: list[str] | None = None
    raw_paths = meta.get("chunk_paths", meta.get("sources"))
    if raw_paths is not None:
        if not isinstance(raw_paths, list) or not all(
            isinstance(x, str) for x in raw_paths
        ):
            raise SnapshotError(
                f"'chunk_paths'/'sources' must be a list of strings ({source})"
            )
        if len(raw_paths) != len(ids):
            raise SnapshotError(
                f"'chunk_paths' has {len(raw_paths)} entries but there are "
                f"{len(ids)} ids ({source})"
            )
        paths = list(raw_paths)

    created_at = meta.get("created_at")
    if created_at is not None and not isinstance(created_at, str):
        created_at = str(created_at)

    if len(ids) == 0:
        notes.append("snapshot is empty (0 vectors).")

    return Snapshot(
        ids=list(ids),
        vectors=vectors,
        model=meta["model"],
        dim=dim,
        paths=paths,
        created_at=created_at,
        source=source,
        adapter=adapter,
        notes=notes,
    )


def dir_of(path: str) -> str:
    """Directory portion of a chunk path, POSIX-normalized ('' if none)."""
    return posixpath.dirname(path.replace("\\", "/"))
