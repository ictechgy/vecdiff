# Case study — swapping an embedding model over a real code index

The launch case study for vecdiff: one real corpus, two real embedding
models, same chunks on both sides. Everything below is reproducible from
this directory; the raw reports are [`report.md`](report.md) /
[`report.json`](report.json) (paths in the report are repo-relative —
nothing was scrubbed after the fact).

## Setup

- **Corpus**: ~12,000 lines of real production code — Python and Swift
  sources from six projects (one of them vecdiff itself; the other five are
  labeled `py_corpus_a..c` / `swift_corpus_d..e` — private projects,
  anonymized labels). Chunked into **371 chunks** of 36-line windows, no
  overlap. Chunk ids encode `project:file:start-line`, so they are stable
  across both snapshots.
- **A (blue)**: `BAAI/bge-small-en-v1.5` (384-dim)
- **B (green)**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- Both embedded with fastembed 0.8.0 (which returns pre-normalized unit
  vectors), exported to the universal **jsonl** snapshot format with the
  [`export_snapshots.py`](export_snapshots.py) script.

## What the diff said

```
vecdiff A.jsonl B.jsonl --full --gate   # → exit 2 (RED)
```

| Signal | Result |
|---|---|
| N1 mean neighbor Jaccard | **0.33** (p10 0.18 / p50 0.33 / p90 0.54) |
| N1 heavy-loss chunks (≥ 70% of top-10 neighbors lost) | **158 / 371 = 42.6%** |
| N1 heavy-loss concentration | `swift_corpus_e/CoronerCore` (32), `py_corpus_c/yield_audit` (27), `py_vecdiff/vecdiff` (22) |
| N2 norm stats | both sides unit-norm (pre-normalized embedder) — outlier check skipped with the reason inline |
| N4 duplicates | 0 pairs on both sides |

## Reading it (evidence, not a verdict)

A model swap is **not a drop-in index replacement**: even though both
pipelines are mechanically healthy (N2 finds no scaling bug, N4 finds no
duplicates), the neighbor graph is rebuilt almost everywhere — only a
third of each chunk's top-10 neighborhood survives the swap, and 56% of
chunks see a neighbor rank inversion of ≥ 5 positions. The loss is not
uniform: `swift_corpus_e/CoronerCore` lost 64% of its chunks' neighborhoods
(32/50) and `py_vecdiff/vecdiff` 37% (22/59), while e.g. `py_corpus_a` stayed
off the heavy-loss podium entirely — that concentration list is exactly what
you want in front of you before cutting traffic: those directories are where
retrieval behavior changed the most and where spot-check queries should go
first.

RED here means "do not cut over blind" — it does *not* mean one model is
better than the other. Whether the new neighborhoods are *worse* is a
human judgment over canonical queries (see the roadmap's supervised check).

## Reproduce

```bash
pip install fastembed vecdiff          # or: pip install -e . from the repo root
python docs/case_study/export_snapshots.py OUT_DIR \
    /path/to/project1=py_project1 /path/to/project2=swift_project2 ...
cd OUT_DIR && vecdiff A.jsonl B.jsonl --full --json report.json --markdown report.md --gate
```

The export script is documentation-grade and runs in your environment;
`fastembed` is its dependency, never vecdiff's (vecdiff stays numpy-only
and never sees the embedding step — it only reads the jsonl dump).
