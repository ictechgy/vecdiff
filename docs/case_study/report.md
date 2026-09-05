# vecdiff 0.1.0 report

## Snapshots

| side | path | adapter | model | dim | n | created_at |
|---|---|---|---|---|---|---|
| A | A.jsonl | jsonl | BAAI/bge-small-en-v1.5 | 384 | 371 | 2026-09-05T15:28:57+00:00 |
| B | B.jsonl | jsonl | sentence-transformers/all-MiniLM-L6-v2 | 384 | 371 | 2026-09-05T15:28:57+00:00 |

Shared ids: **371** (only in A: 0, only in B: 0)

## Checks

### N1 neighbor stability (kNN Jaccard, Vectory-style)

- queries: 371 shared ids (full)
- mean Jaccard: **0.329**
- Jaccard percentiles: p10 0.176, p25 0.250, p50 0.333, p75 0.429, p90 0.538
- rank inversion: mean |dRank| 2.43, max 9, ids with inversion >= 5: 56.1%
- heavy-loss ids (Jaccard <= 0.30): 158 (42.6%)
- concentrated in: `swift_corpus_e/CoronerCore` (32), `py_corpus_c/yield_audit` (27), `py_vecdiff/vecdiff` (22), `swift_corpus_d/BreadcrumbCore` (22), `py_corpus_b/agent2perfetto` (13)

### N2 population stats

| stat | A | B |
|---|---|---|
| n | 371 | 371 |
| dim | 384 | 384 |
| norm mean | 1.000 | 1.000 |
| norm std | 0.000 | 0.000 |
| norm min | 1.000 | 1.000 |
| norm max | 1.000 | 1.000 |
| outliers (|z| > 3.0) | 0 | 0 |

### N4 duplicates (cosine >= 0.999)

| side | pairs | ratio | affected ids |
|---|---|---|---|
| A | 0 | 0.00% | 0 |
| B | 0 | 0.00% | 0 |

## Findings

| severity | check | finding |
|---|---|---|
| RED | N1 | mean neighbor Jaccard 0.33 (k=10, full, n=371; thresholds: green >= 0.9, yellow >= 0.7) — migration regression suspected; inspect heavy-loss ids |
| RED | N1 | 158 chunks lost >= 70% of their top-10 neighbors (Jaccard <= 0.3; thresholds: green < 2%, yellow < 10% of queried ids), concentrated in: swift_corpus_e/CoronerCore (32), py_corpus_c/yield_audit (27), py_vecdiff/vecdiff (22) |
| GREEN | N2 | dim consistent across snapshots (dim=384) |
| GREEN | N2 | A: no extreme-norm outliers (\|z\| > 3.0) — norm variance ~0 (pre-normalized embedder?); z-scores would be float noise |
| GREEN | N2 | B: no extreme-norm outliers (\|z\| > 3.0) — norm variance ~0 (pre-normalized embedder?); z-scores would be float noise |
| GREEN | N2 | norm mean shift A->B 0.0% (A mu=1.000, B mu=1.000; thresholds: green <= 5%, yellow <= 20%) — norm distribution stable |
| GREEN | N4 | A: 0 duplicate pair(s) at cosine >= 0.999 (0.00% of 371 vectors; thresholds: green == 0, yellow < 1%, red >= 1%) — no duplicates |
| GREEN | N4 | B: 0 duplicate pair(s) at cosine >= 0.999 (0.00% of 371 vectors; thresholds: green == 0, yellow < 1%, red >= 1%) — no duplicates |

**Verdict: RED: 2 red, 0 yellow, 6 green finding(s)**

> vecdiff collects evidence for human judgment; it never claims one index or model is better.

Gate: exit code **2** (2 = any RED, 1 = any YELLOW, 0 = all green).

## Notes (method honesty)

- N1 neighbors are exact brute-force cosine over each snapshot's full index (not ANN), computed in blocked matrix multiplications with bounded memory.
- N4 is an exact O(n^2) scan (blocked); for very large indexes budget time accordingly.
- A: jsonl adapter: model/created_at come from the optional sidecar <name>.meta.json (defaults: model 'unknown').
- B: jsonl adapter: model/created_at come from the optional sidecar <name>.meta.json (defaults: model 'unknown').
