# Export recipes — diffing any vector DB

vecdiff never talks to your database. It diffs *snapshots*: ids + float32
vectors + metadata. Two universal paths exist, in order of preference:

1. **Programmatic** — pull vectors through your DB's client and build the
   snapshot in memory (`snapshot_from_arrays`). No intermediate file.
2. **JSONL export** — dump one JSON object per chunk; `vecdiff` reads it
   directly (`.jsonl` / `.ndjson`, optionally `.gz`).

Native (`.npz`/dir), sqlite and FAISS adapters also exist (see README).
The snippets below run in *your* environment with *your* DB client —
vecdiff itself stays numpy-only.

## JSONL format

```
{"id": "chunk_0001", "vector": [0.12, ...], "path": "src/auth/login.py"}
{"id": "chunk_0002", "vector": [0.03, ...]}
```

- `id` (string, required, unique) — the stable chunk id; N1 matches on it
  across snapshots, so it must survive re-embedding.
- `vector` (list of numbers, required) — one embedding.
- `path` (string, optional) — chunk source path; enables N1 heavy-loss
  concentration by directory.
- Sidecar `snap.meta.json` next to `snap.jsonl` (same stem) may provide
  `{"model": "...", "created_at": "..."}`.

Then:

```bash
vecdiff old.jsonl new.jsonl --full --gate
```

## Qdrant

```python
from qdrant_client import QdrantClient
import json

client = QdrantClient(url="http://localhost:6333")
points, offset = None, None
with open("qdrant.jsonl", "w") as f:
    while points is None or offset:
        points, offset = client.scroll(
            "my_collection", limit=256, offset=offset, with_vectors=True,
            with_payload=True,
        )
        for p in points:
            rec = {"id": str(p.id), "vector": list(p.vector)}
            if isinstance(p.payload, dict) and "path" in p.payload:
                rec["path"] = str(p.payload["path"])
            f.write(json.dumps(rec) + "\n")
```

## Chroma

```python
import chromadb, json

col = chromadb.PersistentClient(path="./chroma").get_collection("code")
with open("chroma.jsonl", "w") as f:
    for batch in col.get(include=["embeddings", "metadatas", "documents"], limit=10_000):
        for i, emb in enumerate(batch["embeddings"]):
            rec = {"id": batch["ids"][i], "vector": [float(x) for x in emb]}
            md = batch["metadatas"][i] or {}
            if "path" in md:
                rec["path"] = str(md["path"])
            f.write(json.dumps(rec) + "\n")
```

## LanceDB

```python
import lancedb, json

tbl = lancedb.connect("./lance").open_table("code")
with open("lance.jsonl", "w") as f:
    for batch in tbl.to_arrow().to_batches(max_chunksize=2048):
        ids, vecs = batch.column("id").to_pylist(), batch.column("vector").to_pylist()
        paths = batch.column("path").to_pylist() if "path" in batch.schema.names else [None] * len(ids)
        for i, v in enumerate(vecs):
            rec = {"id": str(ids[i]), "vector": [float(x) for x in v]}
            if paths[i]:
                rec["path"] = str(paths[i])
            f.write(json.dumps(rec) + "\n")
```

## pgvector (psycopg)

```python
import psycopg, json

with psycopg.connect("dbname=mydb") as conn, conn.cursor() as cur, \
     open("pg.jsonl", "w") as f:
    cur.execute("SELECT id, embedding::float8[], path FROM chunks")
    for cid, vec, path in cur:  # psycopg3 streams rows server-side
        rec = {"id": str(cid), "vector": list(vec)}
        if path:
            rec["path"] = str(path)
        f.write(json.dumps(rec) + "\n")
```

## In-memory (no file at all)

```python
from vecdiff.snapshot import snapshot_from_arrays
from vecdiff.checks import check_n1, check_n2, check_n4

snap_a = snapshot_from_arrays(ids=ids_a, vectors=vecs_a, model="bge-m3",
                              chunk_paths=paths_a)
snap_b = snapshot_from_arrays(ids=ids_b, vectors=vecs_b, model="voyage-3",
                              chunk_paths=paths_b)
n2_stats, n2 = check_n2(snap_a, snap_b)
n1_stats, n1 = check_n1(snap_a, snap_b, full=True)
```

## Rules that make any export correct

- **Ids must be stable across the two snapshots** — N1 pairs chunks by id.
  Never export row numbers as ids (this is why the FAISS adapter, whose ids
  are positional, is N2/N4-only in practice).
- **Keep `path` if you have it** — it unlocks N1's heavy-loss concentration
  by directory, usually the most actionable finding.
- **Dump everything** — N1 neighbor identity depends on the full candidate
  pool; a filtered export changes neighbors for chunks you didn't touch.
- Same `dim` on both sides; a dimension mismatch is a hard error (exit 3)
  and means the two snapshots are not comparable.
