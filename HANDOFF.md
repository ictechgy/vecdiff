# HANDOFF — what the next session should do

Read `AGENTS.md` first (layout, invariants, judgment discipline, test gotchas).

**State at writing (2026-09-07):** main = `0.4.0` **published on PyPI**
(N3 symbol-level ghosts;
https://github.com/ictechgy/vecdiff/releases/tag/v0.4.0), 21 commits,
104 tests: 103 green + 1 intended skip¹. Repo public at
https://github.com/ictechgy/vecdiff, CI green on 3.10 + 3.13.
The full v0.1–v0.3 roadmap from the original planning doc is shipped; what
remains is below, ordered.

¹ skip = the faiss graceful-degradation test, which only runs *without*
faiss installed; the dev venv has `faiss-cpu`, so the integration tests run
locally and this one skips. In CI (no faiss) the inverse. Both are correct.

## 1. Do next — in this order

- [x] **0.4.0 released 2026-09-07** — published on PyPI (workflow green,
      wheel verified by direct download + import). Procedure used for
      0.2.0/0.3.0/0.4.0: set `__version__` in `src/vecdiff/__init__.py`,
      commit, `git tag vX.Y.Z && git push origin main vX.Y.Z`, `gh release
      create vX.Y.Z` → `.github/workflows/pypi.yml` runs tests → `uv build`
      → OIDC trusted publishing (no tokens anywhere; PyPI pending publisher
      is owner=ictechgy repo=vecdiff workflow=pypi.yml, environment blank —
      don't add `environment:` to the workflow unless PyPI side changes).
      After the run, verify with the PyPI JSON API + a clean-venv install;
      note `uv pip install vecdiff` may serve a stale cache — pin
      `vecdiff==X.Y.Z` when verifying.
- [x] **kartograph recipe added 2026-09-07.** kartograph ships a
      `code-graph` JSON export with opt-in project-relative paths since
      v0.3.0 (v0.3.1+ recommended; schema verified in
      `export/GraphJsonRenderer.kt`: nodes carry `usr`/`qualifiedName`/
      `location.path` + `pathKind`, absolute paths never emitted — join on
      project-relative paths). Recipe with CLI + Gradle-plugin invocations
      and the conversion snippet now lives next to the cartograph one in
      `docs/export_recipes.md`; the vecdiff interface needed no changes.
- [ ] **(Optional) dogfood the real dev pipeline.** The maintainer's
      private code-search pipeline (vector DB over Android/iOS codebases)
      was NOT on this machine (searched 2026-09-06). If it lives elsewhere,
      two snapshots through the jsonl export + a `--queries-a/b` run on real
      query logs would strengthen the case study beyond the already-shipped
      real-code surrogate (`docs/case_study/`).
- [ ] **Launch propagation** (from the planning doc's first-publication
      strategy): get the case study cited in re-embedding / dual-index
      migration guide blogs — no diff tool existed there; the README case
      study section links `docs/case_study/` and is written to be quotable.
- [ ] **ANN prefilter for N4 — only on real pain.** Exact O(n²·d) is
      documented; if a user reports 100k+ chunk pain, consider an optional
      ANN prefilter with an explicit exactness note (cost honesty invariant).

## 2. Shipped (history, terse)

- **0.2.0** — jsonl universal adapter (chunked float32 parse, gz, sidecar
  `<stem>.meta.json`), `snapshot_from_arrays()`, export recipes
  (Qdrant/Chroma/LanceDB/pgvector), non-finite (NaN/inf) rejection in every
  adapter, vectorized N4, heavy-loss caps, real case study
  (bge-small vs MiniLM over 371 real code chunks; N2 pre-normalized
  false positive found and fixed by dogfooding).
- **0.3.0** — N5 constant vectors (bit-identical reuse = pipeline-bug
  semantics; exact np.unique), Q1 supervised canonical queries
  (`--queries-a/b`, per-side embedded query jsonl,
  `knn.topk_cosine_queries`), N3 file-level orphans (`--paths-manifest`
  text), FAISS integration tests (local-only).
- **0.4.0** — N3 symbol-level ghosts: chunk `symbols` metadata, jsonl
  `{path, symbols}` manifest, ghost detection + report columns, cartograph
  recipe (usr + location.path schema verified). kartograph recipe added
  post-release (docs-only; see §1).
- **0.4.1.dev0 (unreleased)** — perf/structure review fixes: N4 duplicate-
  flood guard (per-block hit cap; pair count stays exact, affected/examples
  skipped past 1M hits/block) + upper-triangle-only FLOPs (2x), query-jsonl
  loader now chunked like the snapshot loader, paths-manifest parsing
  extracted from cli.py to `snapshot.load_paths_manifest()`, jsonl line
  sanity cap (4 MB) on snapshot/query/manifest loads, markdown escape
  handles newlines, Q1 thresholds alias N1 (drift-proof), stale v0.1
  message and CLI --help description fixed.
- **Repo history note:** the original Korean planning doc (기획서.md) was
  `git filter-repo`-stripped from all history before going public; it now
  lives as a **local-only gitignored file** — never re-commit it. The
  pre-rewrite history (including the doc) is in
  `/tmp/vecdiff-pre-filter.bundle` (may vanish on reboot; the working-tree
  copy is the durable one). All commit hashes were rewritten once.

## 3. Ops knowledge that isn't written anywhere else

- **CI billing quirk:** Actions on this GitHub account failed to *start*
  ("payments have failed / spending limit") while the repo was private;
  going public fixed it (free minutes). For future private repos on this
  account, check Settings → Billing & plans before suspecting code.
- **Platform-fragile tests:** never assert at an exact float boundary —
  N4/Q1 scores are float32 blocked matmuls; BLAS rounding differs
  (macOS Accelerate vs Linux OpenBLAS) and a float64-equal threshold can
  miss by an ulp on CI. Bit us on the first-ever Linux run; margin
  pattern + AGENTS.md gotcha entry exist.
- **Thresholds lockstep:** every constant in `checks.py` must appear in the
  README signal table in the same change (AGENTS.md invariant). Current
  bands: N1 0.90/0.70 + heavy 2%/10%; N2 shift 5%/20%, outliers 1%/5%
  (skipped when norm CV < 1e-6); N4 0/1%/1%; N5 <5 members/5%; Q1 mirrors
  N1; N3 rot 0/<5%/≥5%.
- **Exit codes:** 0/1/2 gate verdicts, 3 hard error; argparse usage errors
  also exit 2 (documented in README — check stderr to distinguish).

## 4. Context pointers

- Method root: kNN-IoU neighborhood comparison; prior art Vectory
  (pentoai), cited in README as the method vecdiff operationalizes for
  production index ops. Differentiation vs Ragas/MTEB also in README.
- Extractor siblings (symbol-graph suppliers for N3): **cartograph**
  (Swift/iOS, github.com/ictechgy/cartograph — usable today) and
  **kartograph** (Kotlin/Android, github.com/ictechgy/kartograph — blocked
  upstream, see §1).
- Other siblings: `../tombstone` (agent negative memory),
  `../agent2perfetto` (session traces) share the local-first /
  graded-findings grammar; `../yield-audit` M9 measures the ROI of the
  vector-DB investment this tool protects.
