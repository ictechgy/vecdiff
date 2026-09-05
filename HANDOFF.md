# HANDOFF — what the next session should do

State at writing: v0.2.0 (bumped for first release — jsonl/any-DB support +
hardening since the 0.1.0 marker; 0.1.0 was never published), 11 commits,
76/76 tests green, clean tree, **public at
https://github.com/ictechgy/vecdiff (CI green on 3.10 + 3.13), not yet on
PyPI**.
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
- [ ] **PyPI publish — one release away (2026-09-06).** Pending publisher
      (trusted publishing) configured on PyPI for owner=ictechgy repo=vecdiff
      workflow=`pypi.yml`; the workflow (`.github/workflows/pypi.yml`) runs
      tests → `uv build` → OIDC publish on every published GitHub release.
      `uv build` verified locally (sdist + wheel, no stray files). To ship:
      fill `[project.urls]` in pyproject.toml, then
      `git tag v0.2.0 && git push origin v0.2.0 && gh release create v0.2.0`.
      If the pending-publisher form had an environment name, uncomment
      `environment:` in pypi.yml (must match exactly). Re-check the name is
      still unclaimed right before the first release.

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
