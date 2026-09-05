# HANDOFF — what the next session should do

State at writing: v0.1.x, 8 commits, 75/75 tests green, clean tree,
**public at https://github.com/ictechgy/vecdiff (CI green on 3.10 + 3.13), not on PyPI**.
Read `AGENTS.md` first (invariants, judgment discipline, test gotchas). (The original
Korean planning doc lives outside the repo — local `기획서.md`, gitignored; it was
stripped from history with `git filter-repo` before going public, backup bundle at
`/tmp/vecdiff-pre-filter.bundle`.)

## 1. Do first — dogfood and capture the launch story

- [ ] **Run it on a real code-embedding index.** The maintainer has a dev-side code
      semantic-search pipeline (local vector DB over Android/iOS codebases). Export two
      real snapshots (e.g. current index vs a candidate new embedding model) via the
      native adapter, run `vecdiff --full --json --markdown`, and keep the full report.
- [ ] That report becomes the README/launch case study ("N1 flagged X chunks concentrated
      in src/auth; N4 caught Y re-chunking duplicates") — the launch
      strategy explicitly calls for one real case, and the re-embedding guide blogs
      (no diff tool exists) are the natural citation path. **Scrub filesystem paths from
      the report before publishing** (reports embed absolute paths).

## 2. Ship it

- [x] **GitHub repo + push — done 2026-09-06.** Public at
      https://github.com/ictechgy/vecdiff; CI green on 3.10 + 3.13. The first
      real Linux run caught one latent test bug (float32/BLAS boundary assert —
      see the "exact float boundary" gotcha in AGENTS.md); fixed.
- [ ] PyPI publish: `vecdiff` name probed clean in 2026-09-05 search (only a SignumData
      library function and a Rust `VecDiff` enum — no CLI tool), but re-check
      pypi.org before upload. Keep `uv build` in the release flow (verified working).

## 3. v0.2 — adapter + supervised line

- [x] **Any-vector-DB support shipped (v0.1.x)**: jsonl adapter (stdlib, gz-capable,
      path-aware) + `snapshot_from_arrays()` in-memory API + `docs/export_recipes.md`
      (Qdrant/Chroma/LanceDB/pgvector snippets). LanceDB/Qdrant *native-file* adapters
      remain open below but are no longer the only path — the universal escape hatch
      covers them today; only add a native-file adapter if users ask to diff the DB
      files directly without an export step (follow the sqlite adapter pattern — never
      build `file:` URIs by string concatenation; `as_uri()` regression test exists).
- [ ] **N5 constant-vector check**: same-index pairs at cosine ≈ 1.0 between *different*
      chunk ids is today folded into N4's report; consider splitting it out as its own
      finding class (pipeline-bug semantics vs re-chunking-accident semantics).
- [ ] **Canonical-query supervised check**: optional query set (extracted from user query
      logs) compared across both indexes with rank-inversion report — complements the
      unsupervised N1.

## 4. v0.3 — the differentiator

- [ ] **N3 orphan/ghost check via symbol graphs**: chunks whose symbols no longer exist,
      audited against IndexStoreDB (iOS) / Kotlin symbol extraction. This is the
      maintainer's static-analysis edge and the main structural advantage over Vectory —
      design the Snapshot↔symbol-graph interface carefully (paths are the join key today).

## 5. Known debts / honesty notes

- [ ] The **FAISS adapter is import-guarded but untested against a real faiss install**
      (faiss-cpu not present in the dev env). If you install faiss-cpu, add an integration
      test (build a flat index, reconstruct, round-trip) marked `@pytest.mark.skipif` on
      faiss availability.
- [ ] `check_n4` on very large indexes is exact O(n²·d) — README documents it; if a user
      reports 100k+ chunk pain, consider an optional ANN prefilter with an exactness note.
- [ ] Docstring drift check: `tests/test_golden.py`-style staleness doesn't exist here, but
      keep the README signal table and `checks.py` constants in lockstep (AGENTS.md rule).

## Context pointers

- Method root: kNN-IoU neighborhood comparison, prior art Vectory (pentoai) — cited in
  README as the established method we operationalize for production code-index ops.
- Sibling projects: `../tombstone` (agent negative memory) and `../agent2perfetto`
  (session traces) share the local-first/graded-findings grammar; `../yield-audit` M9
  measures the ROI of the vector-DB investment this tool protects.
