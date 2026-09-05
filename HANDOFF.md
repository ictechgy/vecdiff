# HANDOFF — what the next session should do

State at writing: v0.2.0 **published on PyPI** (`pip install vecdiff`,
release https://github.com/ictechgy/vecdiff/releases/tag/v0.2.0), 12 commits,
76/76 tests green, clean tree, repo public with CI green on 3.10 + 3.13.
Read `AGENTS.md` first (invariants, judgment discipline, test gotchas). (The original
Korean planning doc lives outside the repo — local `기획서.md`, gitignored; it was
stripped from history with `git filter-repo` before going public, backup bundle at
`/tmp/vecdiff-pre-filter.bundle`.)

## 1. Do first — dogfood and capture the launch story

- [x] **Case study captured 2026-09-06.** The maintainer's private
      code-search pipeline was not present on this machine, so the case study
      was built from ~12k lines of real local code (six projects, five
      anonymized) × two real models (bge-small-en-v1.5 vs all-MiniLM-L6-v2,
      371 stable chunk ids), exported via the jsonl adapter:
      `docs/case_study/` (README + report.md + report.json + export script).
      Headline: mean Jaccard 0.33, 42.6% heavy-loss, concentrated by
      directory, gate exit 2 — framed as evidence for cutover review, not a
      model ranking. Dogfooding also surfaced and fixed the N2
      pre-normalized-embedder false positive (norm variance ≈ 0 → outlier
      check skipped with reason).
- [ ] (Optional) An additional internal run on the real dev pipeline, if it
      lives on another machine, would strengthen the story further.
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
- [x] **PyPI publish — done 2026-09-06.** Trusted publishing via
      `.github/workflows/pypi.yml` (tests → `uv build` → OIDC, no tokens in
      the repo). `vecdiff 0.2.0` is live: https://pypi.org/project/vecdiff/
      (clean-venv install + `--version` verified). Future releases: bump
      `__version__`, commit, `git tag vX.Y.Z && git push origin vX.Y.Z &&
      gh release create vX.Y.Z`. If a future pending-publisher change adds
      an environment name, uncomment `environment:` in pypi.yml to match.

## 3. v0.2 — adapter + supervised line

- [x] **Any-vector-DB support shipped (v0.1.x)**: jsonl adapter (stdlib, gz-capable,
      path-aware) + `snapshot_from_arrays()` in-memory API + `docs/export_recipes.md`
      (Qdrant/Chroma/LanceDB/pgvector snippets). LanceDB/Qdrant *native-file* adapters
      remain open below but are no longer the only path — the universal escape hatch
      covers them today; only add a native-file adapter if users ask to diff the DB
      files directly without an export step (follow the sqlite adapter pattern — never
      build `file:` URIs by string concatenation; `as_uri()` regression test exists).
- [x] **N5 constant-vector check — done 2026-09-06.** `check_n5` flags
      bit-identical vectors reused across chunk ids (np.unique, exact,
      O(n log n·d) — no O(n²)); small groups defer to N4 (re-chunking
      semantics), a group ≥ 5 members or ≥ 5% of the index escalates to
      constant/cached-embedding suspicion (pipeline-bug semantics). Wired
      into CLI + console/JSON/Markdown reports; README signal table in
      lockstep. Note: N4 still counts bit-identical pairs in its cosine
      threshold scan — N5 is the diagnostic lens on top, not a partition.
- [x] **Canonical-query supervised check (Q1) — done 2026-09-06.**
      `--queries-a/--queries-b` (jsonl {id, vector} per side, embedded with
      that side's model — loader `snapshot.load_query_vectors`, non-finite
      rejected): per-query top-k Jaccard + rank inversions over both indexes,
      worst-query drill-down, N1-style thresholds as Q1_* constants (README
      in lockstep). `knn.topk_cosine_queries` is the external-query variant
      (no self-exclusion) with the same blocking.

## 4. v0.3 — the differentiator

- [x] **N3 interface landed 2026-09-06 (file-level)**: `--paths-manifest`
      (newline-separated existing source paths) + `check_orphans` grade
      chunks whose path vanished (0 / <5% / >=5% thresholds). The
      Snapshot<->symbol-graph join is paths, as specified.
- [ ] **N3 symbol-level ghosts**: IndexStoreDB (iOS) / Kotlin symbol
      extraction emitting a *richer* manifest (path -> surviving symbols) so
      "file exists but all its symbols moved" also counts as rot. The check
      interface is ready; the extractor is user-side tooling (never a
      vecdiff dependency).

## 5. Known debts / honesty notes

- [x] **FAISS integration test — done 2026-09-06.** `faiss-cpu` installed in
      the dev venv; `tests/test_faiss_integration.py` (flat round trip, empty
      index, non-finite rejection) runs when faiss is present, skips cleanly
      in CI via importorskip. Version on main bumped to 0.3.0.dev0 after the
      0.2.0 release; N5 ships in 0.3.0.
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
