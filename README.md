# vecdiff

Diff two embedding-index snapshots and get graded, evidence-first findings — for **codebase vector-DB migrations** (model swaps, re-chunking) and **index rot audits** (orphans, duplicates).

Fully local, deterministic, `numpy`-only. vecdiff never needs your original vector DB and never touches a network.

> The standard advice for re-embedding is "run the new index side by side (blue/green), compare, then cut over." Nobody ships the *compare* step — teams throw a few queries at both indexes and go by feel. vecdiff mechanizes that comparison.

```console
$ pip install vecdiff
$ vecdiff old-snapshot/ new-snapshot/ --gate
```

## Why

A code-embedding index has three chronic anxieties:

1. **Model swap** — embedding spaces are not comparable vector-to-vector, so "did retrieval quality survive?" was unanswerable by machine.
2. **Re-chunking / re-indexing** — *what exactly changed?* had no answer at all.
3. **Rot** — code moves, indexes sit still. Chunks pointing at dead symbols and accidental duplicates accumulate quietly.

vecdiff answers all three with the same primitive: **index-vs-index diff** plus per-chunk health checks.

## Quickstart

```console
# 1. install (numpy is the only runtime dependency)
pip install vecdiff        # or: uv tool install vecdiff

# 2. generate two demo snapshots (deterministic; B simulates a sloppy re-embed)
python scripts/make_demo_snapshots.py /tmp/vecdemo

# 3. diff them
vecdiff /tmp/vecdemo/snapA /tmp/vecdemo/snapB --markdown report.md --gate
echo $?
```

You get console findings plus `report.md`, and `--gate` turns them into an exit code for CI (see below).

## What it checks (v0.1)

| # | Check | What it catches |
|---|---|---|
| **N1** | Neighbor stability | Model-swap / re-index regressions: per-chunk top-k neighbor sets compared across snapshots (Jaccard + rank inversions), heavy-loss chunks grouped by directory |
| **N2** | Population stats | Pipeline breakage early warnings: norm distribution shift, extreme-norm outliers, dimension mismatch (hard error) |
| **N4** | Duplicates | Re-chunking accidents and boilerplate floods: within-index pairs at cosine ≥ threshold |
| **N5** | Constant vectors | Pipeline bugs (cached API response, constant fallback, broken batch): one bit-identical embedding reused across many chunk ids |
| **Q1** (`--queries-a/b`) | Canonical queries (supervised) | What real retrieval traffic would see: the same query set run through both indexes, per-query top-k overlap + rank inversions |
| **N3** (`--paths-manifest`) | Orphans + ghosts | Rot: chunks whose source file no longer exists; with a symbol-graph manifest (e.g. [cartograph](https://github.com/ictechgy/cartograph) for Swift), chunks whose file survives but whose declared symbols are gone |

Roadmap: **N3** orphan/ghost chunks audited against a symbol graph (iOS IndexStoreDB / Kotlin), canonical-query supervised comparison, time-series rot monitoring.

### Signal thresholds

vecdiff reports graded signals, never a verdict like "model B is better". Thresholds are stated inline in every finding:

| Signal | Green | Yellow | Red |
|---|---|---|---|
| N1 mean neighbor Jaccard | ≥ 0.90 | ≥ 0.70 | < 0.70 |
| N1 heavy-loss chunks (Jaccard ≤ 0.30, i.e. ≥ 70% of top-k lost) | < 2% of sampled | < 10% | ≥ 10% |
| N2 norm mean shift A→B | ≤ 5% | ≤ 20% | > 20% |
| N2 extreme-norm outliers (\|z\| > 3) | ≤ 1% | < 5% | ≥ 5% |

The N2 outlier check is skipped (reported green, with the reason inline) when norm variance is ≈ 0 — e.g. embedders that return pre-normalized unit vectors, where z-scores would be float rounding noise.
| N4 duplicate pairs (cosine ≥ threshold) / n | 0 | < 1% | ≥ 1% |
| N5 largest bit-identical group | < 5 members | ≥ 5 members | ≥ 5% of index |
| Q1 mean query Jaccard | ≥ 0.90 | ≥ 0.70 | < 0.70 |
| Q1 heavy-loss queries (Jaccard ≤ 0.30) | < 2% | < 10% | ≥ 10% |
| N3 rot chunks (orphans + ghosts) / n | 0 | < 5% | ≥ 5% |

`--gate` exit codes: `0` all green, `1` any yellow, `2` any red. (Note: argparse usage errors — a mistyped flag — also exit 2; check stderr to tell them apart from a red verdict. Hard errors — unreadable snapshot, dimension mismatch — exit 3.)

### Sampling

N1 is exact per queried chunk but samples shared ids by default (`--sample 0.2`, `--seed 0` — same seed, same sample, reproducible reports). Use `--full` for an exact run; cost is O(queried × n × dim).

## Case study — model swap over a real code index

371 chunks (~12k lines of real Python/Swift code), same chunks embedded with `bge-small-en-v1.5` vs `all-MiniLM-L6-v2`:

- mean neighbor Jaccard **0.33**; **42.6%** of chunks lost ≥ 70% of their top-10 neighbors
- loss concentrated by directory (e.g. 32/50 chunks in one Swift core module, 22/59 in vecdiff itself) — the spot-check list for cutover
- N2/N4 confirmed both pipelines mechanically healthy (no scaling bug, no duplicates); gate exit **2** = do not cut over blind

Full story + reproducible commands: [docs/case_study](docs/case_study/README.md). Dogfooding this run also fixed a real tool bug (N2 now skips its norm-outlier check for pre-normalized embedders, with the reason inline).

## Snapshot formats

A *snapshot* is a model-agnostic dump: chunk ids + float32 vectors + metadata (`model`, `dim`, `chunk_paths`, `created_at`). Loading validates the dump strictly and rejects duplicate ids, dimension/length mismatches, and non-finite (NaN/inf) vectors — a poisoned row would make cosine top-k silently arbitrary, so it fails at load instead of mid-report.

| Adapter | Input | Notes |
|---|---|---|
| `native` (built-in) | directory with `vectors.npy` + `meta.json`, or a single self-describing `.npz` | the interchange format — export once, diff forever |
| `jsonl` (built-in) | `.jsonl` / `.ndjson` (optionally `.gz`): `{"id", "vector", "path"?, "symbols"?}` per line; optional `<stem>.meta.json` sidecar | the universal escape hatch — any vector DB can dump this in a few lines of client code |
| `sqlite` (built-in) | `chunks(id TEXT PRIMARY KEY, vec BLOB)` float32 little-endian; optional `meta(key, value)` table | stdlib only |
| `faiss` (optional) | `.index` / `.faiss` | requires `pip install faiss-cpu`; only flat-style indexes whose vectors can be reconstructed |

Format is auto-detected from the path; override with `--format`.

**Any other vector DB** (Qdrant, Chroma, LanceDB, pgvector, …): dump a JSONL snapshot with your DB's own client, or build one in memory with `snapshot_from_arrays(ids=..., vectors=..., model=...)` — vecdiff stays numpy-only and never talks to your database. Ready-to-run snippets live in [docs/export_recipes.md](docs/export_recipes.md). One rule above all: ids must be stable across the two snapshots (N1 pairs chunks by id), so never export row numbers.

## CI migration gate

```yaml
# .github/workflows/reindex-gate.yml — run before cutting traffic to the new index
- run: vecdiff snapshots/blue/ snapshots/green/ --full --gate
  # exit 2 (red) blocks the cutover step
```

vecdiff collects evidence for a human decision; the gate just makes "nobody looked" impossible.

Exit codes: `0` all green, `1` any yellow, `2` any red — and `3` for hard errors (bad snapshot, dimension mismatch, I/O failure), distinct from gate verdicts so CI can tell "the comparison failed" from "the comparison said no".

## Method notes & honesty

- **Cross-model comparison is impossible vector-to-vector** — embedding spaces are unrelated. That is precisely why N1 compares *neighbor-graph structure* (per-chunk kNN Jaccard, Vectory-style) instead of coordinates. The method's prior art: [Vectory](https://github.com/pentoai/vectory) (pentoai), which established kNN-IoU for embedding-space comparison in an ML-experiment-tracking frame; vecdiff operationalizes it for production code-index migrations and adds health checks Vectory does not have.
- **N1 is a function of the candidate pool**: neighbor identity depends on the whole index, so removing chunks changes even "clean" chunks' neighbor sets. Heavy-loss concentration by directory (`chunk_paths`) is where you should look first.
- **N2 is an early-warning system, not a quality metric**: a shifted norm distribution usually means pipeline scaling changed, not that retrieval got worse.
- **Judgment stays human.** vecdiff's job is findings like *"14 chunks lost ≥70% of their top-10 neighbors, concentrated in src/auth (11)"* — not scores.

## Differentiation

| Adjacent tool | Difference |
|---|---|
| [Vectory](https://github.com/pentoai/vectory) | Established the kNN-IoU method (cited). It is an ML *experiment tracking* toolkit (SQLite + Elasticsearch frame, image/IMDB demos); vecdiff is a production *index operations* tool: DB adapters, path-concentrated heavy-loss findings, CI gate, rot checks. |
| Ragas / RAG eval frameworks | Evaluate end-task answer quality; do not diff index-vs-index. |
| MTEB | Benchmark leaderboards for models; unrelated to *your* index. |
| Vendor migration guides / dual-index blog posts | Describe the pattern; none ship the comparison tool. |
| Vector-index visualizers (e.g. zilliztech's) | Visualize ANN search internals; not migration verdicts. |

## Privacy

Everything runs locally: no network calls, no telemetry, snapshots stay on your machine. Reports contain chunk ids, paths, and similarity numbers — check them into the repo only if you're comfortable with that.

## License

Apache-2.0
