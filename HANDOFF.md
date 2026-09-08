# HANDOFF — what the next session should do

## 2026-09-08 전체 코드 검토 — 수정 완료 (0.4.2.dev0)

- 기준: `main@ac2627b`, 27 commits, 검토 시작 시 워킹트리 clean. 아래 4건은
  **2026-09-08 후속 세션에서 모두 수정 + 회귀 테스트 추가 완료** (123 passed
  + 1 intended skip, +11 tests). 아래 원문은 검토 기록으로 보존한다.

### 확인된 수정 필요 사항 — 전부 수정됨

1. **[수정됨] P2 · correctness/reliability — SQLite의 빈 vector BLOB이 정상 입력으로 통과한다.**
   - 수정: 0길이 BLOB을 행 단위 `SnapshotError`로 거부 (`snapshot.py` sqlite
     loader). 빈 테이블의 valid-empty 동작은 유지. 회귀:
     `test_sqlite_empty_blob_rejected`, `test_sqlite_empty_blob_mixed_with_valid_rejected`.
2. **[수정됨] P2 · correctness — 일부 행의 path 미보고를 사라진 파일로 오판한다.**
   - 수정: `check_orphans`가 빈 path 청크를 audit에서 제외하고
     `without_path_metadata`로 분리 계수. 분모(coverage) 규칙을 "path-reported
     chunks"로 명시 — README/README.ko 신호 테이블 동일 커밋 갱신 (lockstep).
     회귀: `test_n3_partial_path_metadata_not_false_orphans`,
     `test_n3_rot_fraction_denominator_is_path_reported`,
     `test_n3_all_paths_empty_skips_yellow`.
3. **[수정됨] P2 · reliability — 큰 JSON 정수에서 CLI hard-error 처리를 빠져나온다.**
   - 수정: snapshot/query 로더의 `float()` 변환을 행 번호 있는
     `SnapshotError`로 래핑 (OverflowError/ValueError). float32 변환 뒤
     finite 검사는 기존 `_require_finite`가 이미 담당. 회귀:
     `test_jsonl_huge_integer_rejected_with_line_number`,
     `test_query_huge_integer_rejected`,
     `test_cli_huge_integer_exits_3_without_traceback` (exit 3 + no traceback).
4. **[수정됨] P3 · performance — paths manifest를 검증 전에 통째로 읽는다.**
   - 수정: `load_paths_manifest`를 줄 단위 스트리밍으로 교체 (전체
     read_text + splitlines 이중 복사 제거). line cap·행 번호 유지. 회귀:
     `test_manifest_last_line_number_preserved_streaming`,
     `test_manifest_line_cap_rejects_huge_line`,
     `test_manifest_large_streamed_load` (100k 라인).

### 실행 검증과 제약 (수정 세션 기준)

- `.venv/bin/python -m pytest -q`: **124 collected, 123 passed, 1 skipped**
  (skip = FAISS 미설정 경로, 의도된 것). CI(FAISS 없음)에서는 적분 테스트가
  skip되고 이 경로가 실행된다.
- 합성 demo snapshots `--full --gate`: RED 검출, exit 2 — 수정 후에도 동일.

### 다음 세션 시작

검토 항목은 소진. 아래 "이전 세션 기록"의 optional 과제(실데이터 dogfood,
홍보 전파, 수요 발생 시 ANN N4)가 다음 순서. 0.4.2 릴리스 준비 완료 상태
(`0.4.2.dev0`).

---

## 이전 세션 기록 — 작성 당시 상태

Read `AGENTS.md` first (layout, invariants, judgment discipline, test gotchas).

**State at writing (2026-09-07):** main = `0.4.1` **published on PyPI**
(perf/hardening review; releases v0.4.0 + v0.4.1 both live), 26 commits,
112 tests: 111 green + 1 intended skip¹. Repo public at
https://github.com/ictechgy/vecdiff with description + topics set, CI green
on 3.10 + 3.13. The full v0.1–v0.3 roadmap from the original planning doc
is shipped; what remains is below, ordered.

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
- **0.4.1** — perf/structure review fixes: N4 duplicate-flood guard
  (per-block hit cap; pair count stays exact, affected/examples skipped
  past 1M hits/block) + upper-triangle-only FLOPs (2x), query-jsonl loader
  chunked like the snapshot loader, paths-manifest parsing extracted to
  `snapshot.load_paths_manifest()`, jsonl line sanity cap (4 MB) on
  snapshot/query/manifest loads, markdown escape handles newlines, Q1
  thresholds alias N1 (drift-proof), stale v0.1 message and CLI --help
  description fixed. 112 tests (+9).
- **0.4.2.dev0 (unreleased)** — 2026-09-08 code-review fixes: sqlite
  empty-vector-blob rejection, N3 partial-path coverage rule (empty path =
  unreported, excluded from the audit; rot fraction over path-reported
  chunks — README/README.ko lockstep), huge-JSON-integer conversion wrapped
  as line-numbered SnapshotError (exit 3, no traceback), streaming
  paths-manifest loader. 123 tests (+11).
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
