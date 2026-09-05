# AGENTS.md — vecdiff

Diffs two embedding-index snapshots (blue/green of a vector-DB migration, or
now-vs-later of a rot audit) and reports graded findings: N1 neighbor stability,
N2 population stats, N4 duplicates. Evidence for human judgment — never a verdict
like "model B is better". See README.md for signal thresholds and differentiation
vs Vectory / Ragas / MTEB.

## Layout

```
src/vecdiff/
  snapshot.py  Snapshot dataclass + adapters (native dir/npz, sqlite, faiss-optional) + strict validation
  knn.py       exact brute-force cosine top-k in memory-bounded blocks; l2_normalize
  checks.py    N1/N2/N4 — thresholds are module constants; Finding dataclass; gate semantics
  report.py    console / JSON / Markdown rendering of the plain-dict report
  cli.py       argparse; arg validation; error → exit 3, gate verdicts → exit 0/1/2
  errors.py    VecdiffError hierarchy (SnapshotError, DimensionMismatchError)
scripts/make_demo_snapshots.py  deterministic 300×32 pair with planted defects (quickstart)
tests/        pytest on synthetic deterministic data; conftest.make_snapshot builds in-memory Snapshots
```

## Commands

```bash
pip install -e . && pip install pytest   # numpy is the only runtime dep
pytest
python scripts/make_demo_snapshots.py /tmp/vecdemo
vecdiff /tmp/vecdemo/snapA /tmp/vecdemo/snapB --full --json r.json --markdown r.md --gate
```

## Invariants — do not break

- **numpy is the only runtime dependency**; sqlite adapter is stdlib; faiss is
  import-guarded and optional. Fully local, zero network at runtime, deterministic
  (fixed seeds; `--seed 0` must reproduce the same N1 sample).
- **Judgment discipline**: checks emit graded signals (green/yellow/red) with thresholds
  stated inline in every finding. Never add wording that ranks models/snapshots overall.
  Thresholds live as constants at the top of `checks.py` AND in the README signal table —
  update both in the same change.
- **N1 is a function of the full candidate pool**: neighbor identity depends on the whole
  index, so removing chunks changes even untouched chunks' neighbor sets. Any test
  asserting subset-comparison equality is wrong (this bit us once). Heavy-loss
  concentration by directory requires `chunk_paths` metadata.
- **Normalization is cached**: use `Snapshot.unit_vectors()` (and
  `topk_cosine(..., unit=...)`) instead of calling `l2_normalize` per check — at
  100k×1024 each redundant call is a ~400 MB allocation.
- **sqlite adapter**: schema is `chunks(id TEXT PRIMARY KEY, vec BLOB)` float32
  little-endian; the loader builds its URI via `Path.as_uri() + "?mode=ro"` — never
  concatenate raw paths into `file:` URIs (a `?` or `#` in the path silently opens an
  empty database; regression-tested).
- **Snapshot validation** rejects duplicate ids, dim mismatches (`dim > 0`, vectors width
  == dim), length mismatches; float64 inputs are cast with a note; empty snapshots are
  valid. `np.load` must always run with `allow_pickle=False`.
- **Exit codes**: 0 all-green / 1 yellow / 2 red are gate verdicts; 3 is a hard error
  (bad snapshot, dimension mismatch, I/O). Keep them disjoint — CI needs to distinguish
  "comparison failed" from "comparison said no".
- **Cost honesty**: N4 is exact O(n²·d) blocked; N1 samples shared ids (`--sample 0.2`,
  `--full` for exact). If you change cost characteristics, update the README notes.

## Testing gotchas

- Data is deterministic (`np.random.default_rng(fixed_seed)`); tests assert on exact
  structural outcomes, so keep seeds.
- Watch for mathematically wrong test constructions (both bit us once): a "near-duplicate"
  built as `v + ε·v̂` is *parallel* → cosine exactly 1.0; scaling k outlier rows can leave
  their z-scores below 3 because the outliers inflate the population std — give outliers
  identical norms instead. See comments in `tests/test_checks.py`.
- `tests/test_demo.py` runs the demo script via subprocess with `PYTHONPATH=src` so it
  works from a source checkout without install.
