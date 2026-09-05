# vecdiff

Diff two embedding-index snapshots and get graded, evidence-first findings — for **codebase vector-DB migrations** (model swaps, re-chunking) and **index rot audits** (orphans, duplicates).

Fully local, deterministic, `numpy`-only. vecdiff never needs your original vector DB and never touches a network.

> The standard advice for re-embedding is "run the new index side by side (blue/green), compare, then cut over." Nobody ships the *compare* step — teams throw a few queries at both indexes and go by feel. vecdiff mechanizes that comparison.

```console
$ pip install vecdiff   # from a checkout; PyPI release planned
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

Roadmap: **N3** orphan/ghost chunks audited against a symbol graph (iOS IndexStoreDB / Kotlin), canonical-query supervised comparison, time-series rot monitoring.

### Signal thresholds

vecdiff reports graded signals, never a verdict like "model B is better". Thresholds are stated inline in every finding:

| Signal | Green | Yellow | Red |
|---|---|---|---|
| N1 mean neighbor Jaccard | ≥ 0.90 | ≥ 0.70 | < 0.70 |
| N1 heavy-loss chunks (Jaccard ≤ 0.30, i.e. ≥ 70% of top-k lost) | < 2% of sampled | < 10% | ≥ 10% |
| N2 norm mean shift A→B | ≤ 5% | ≤ 20% | > 20% |
| N2 extreme-norm outliers (\|z\| > 3) | ≤ 1% | < 5% | ≥ 5% |
| N4 duplicate pairs (cosine ≥ threshold) / n | 0 | < 1% | ≥ 1% |

`--gate` exit codes: `0` all green, `1` any yellow, `2` any red.

### Sampling

N1 is exact per queried chunk but samples shared ids by default (`--sample 0.2`, `--seed 0` — same seed, same sample, reproducible reports). Use `--full` for an exact run; cost is O(queried × n × dim).

## Snapshot formats

A *snapshot* is a model-agnostic dump: chunk ids + float32 vectors + metadata (`model`, `dim`, `chunk_paths`, `created_at`).

| Adapter | Input | Notes |
|---|---|---|
| `native` (built-in) | directory with `vectors.npy` + `meta.json`, or a single self-describing `.npz` | the interchange format — export once, diff forever |
| `sqlite` (built-in) | `chunks(id TEXT PRIMARY KEY, vec BLOB)` float32 little-endian; optional `meta(key, value)` table | stdlib only |
| `faiss` (optional) | `.index` / `.faiss` | requires `pip install faiss-cpu`; only flat-style indexes whose vectors can be reconstructed |

Format is auto-detected from the path; override with `--format`.

Exporting from your vector DB into `native` is ~10 lines: dump ids and float32 vectors, write `meta.json`. (Per-DB export helpers are on the roadmap.)

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
