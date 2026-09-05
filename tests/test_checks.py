"""N1 / N2 / N4 check tests on deterministic synthetic data."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from vecdiff import checks
from vecdiff.errors import DimensionMismatchError

from conftest import make_ids, make_snapshot


def _base(n=200, d=32, seed=7):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)).astype(np.float32)


# ---------------------------------------------------------------------------
# N1
# ---------------------------------------------------------------------------


def test_n1_identical_snapshots_all_green():
    ids = make_ids(80)
    v = _base(80, 16)
    a = make_snapshot(ids, v)
    b = make_snapshot(ids, v.copy())
    stats, findings = checks.check_n1(a, b, full=True)
    assert stats["mean_jaccard"] == 1.0
    assert stats["shared_ids"] == 80
    assert stats["heavy_loss_ids"] == []
    assert checks.worst_severity(findings) == "green"


def test_n1_catches_noisy_subset():
    ids = make_ids(200)
    v = _base(200, 32)
    b = v.copy()
    b[:40] += np.random.default_rng(99).standard_normal((40, 32)).astype(np.float32) * 2.0
    a = make_snapshot(ids, v)
    b_snap = make_snapshot(ids, b)
    stats, findings = checks.check_n1(a, b_snap, full=True)

    # damaged ids are recoverable from stats: heavy ids must be a superset of
    # (nearly) all damaged ids, and clean ids must stay stable
    heavy = set(stats["heavy_loss_ids"])
    damaged_ids = set(ids[:40])
    clean_ids = set(ids[40:])
    heavy_clean = heavy & clean_ids
    assert len(heavy & damaged_ids) >= 38          # subset caught
    assert len(heavy_clean) <= 2                   # few false positives
    assert stats["mean_jaccard"] < checks.N1_MEAN_GREEN
    assert checks.worst_severity(findings) in ("yellow", "red")


def test_n1_jaccard_split_damaged_vs_clean():
    ids = make_ids(150)
    v = _base(150, 32)
    b = v.copy()
    b[:50] += np.random.default_rng(1).standard_normal((50, 32)).astype(np.float32) * 3.0
    a = make_snapshot(ids, v)
    stats, _ = checks.check_n1(a, make_snapshot(ids, b), full=True)
    # mean sits between the damaged mass (~0) and the clean mass — but not
    # at 1.0: clean chunks whose neighbors were among the damaged 50 lose
    # those neighbors too, so the clean mass degrades as well.
    assert 0.15 < stats["mean_jaccard"] < 0.85
    assert len(stats["heavy_loss_ids"]) >= 45
    assert stats["heavy_loss_concentration"]["paths_available"] is True


def test_n1_permutation_invariance():
    ids = make_ids(120)
    v = _base(120, 16)
    w = _base(120, 16, seed=8)
    a = make_snapshot(ids, v)
    b = make_snapshot(ids, w)
    perm = np.random.default_rng(5).permutation(120)
    b_perm = make_snapshot([ids[i] for i in perm], w[perm])
    stats1, _ = checks.check_n1(a, b, full=True)
    stats2, _ = checks.check_n1(a, b_perm, full=True)
    assert stats1["mean_jaccard"] == pytest.approx(stats2["mean_jaccard"])
    assert stats1["rank_inversion"] == stats2["rank_inversion"]


def test_n1_renamed_ids_shared_logic():
    ids_a = make_ids(100)
    v = _base(100, 16)
    w = _base(100, 16, seed=3)
    a = make_snapshot(ids_a, v)
    # B keeps only the first 40 ids; 60 renamed
    ids_b = ids_a[:40] + [f"new_{i:04d}" for i in range(60)]
    b = make_snapshot(ids_b, w)
    stats, findings = checks.check_n1(a, b, full=True)
    assert stats["shared_ids"] == 40
    assert stats["only_in_a"] == 60
    assert stats["only_in_b"] == 60
    # v and w are independent spaces, so shared ids should have unstable
    # neighbors -> low Jaccard -> red. (Jaccard is a function of the full
    # candidate pool, so it deliberately does NOT equal a shared-only
    # subset comparison.)
    assert stats["mean_jaccard"] < 0.3
    assert checks.worst_severity(findings) == "red"


def test_n1_deterministic_sampling():
    ids = make_ids(500)
    v = _base(500, 16)
    w = v + np.random.default_rng(11).standard_normal((500, 16)).astype(np.float32) * 0.05
    a = make_snapshot(ids, v)
    b = make_snapshot(ids, w)
    s1, f1 = checks.check_n1(a, b, sample=0.2, seed=0)
    s2, f2 = checks.check_n1(a, b, sample=0.2, seed=0)
    assert s1 == s2
    assert [x.message for x in f1] == [x.message for x in f2]
    assert s1["sampled_ids"] == 100  # ceil(500 * 0.2)


def test_n1_k_clamped_when_index_small():
    ids = make_ids(5)
    v = _base(5, 8)
    stats, findings = checks.check_n1(
        make_snapshot(ids, v), make_snapshot(ids, v.copy()), full=True, k=10
    )
    assert stats["k_effective"] == 4
    assert stats["mean_jaccard"] == 1.0


def test_n1_too_few_shared_ids_yellow():
    a = make_snapshot(["a", "b"], _base(2, 8))
    b = make_snapshot(["c", "d"], _base(2, 8, seed=2))
    stats, findings = checks.check_n1(a, b, full=True)
    assert stats["shared_ids"] == 0
    assert findings[0].severity == "yellow"


def test_n1_no_path_metadata_concentration_unknown():
    ids = make_ids(30)
    v = _base(30, 8)
    a = make_snapshot(ids, v, paths=None)
    b = make_snapshot(ids, v + np.ones((30, 8), dtype=np.float32), paths=None)
    stats, _ = checks.check_n1(a, b, full=True)
    assert stats["heavy_loss_concentration"]["paths_available"] is False


# ---------------------------------------------------------------------------
# N2
# ---------------------------------------------------------------------------


def test_n2_identical_all_green():
    ids = make_ids(60)
    v = _base(60, 16)
    stats, findings = checks.check_n2(make_snapshot(ids, v), make_snapshot(ids, v))
    assert checks.worst_severity(findings) == "green"
    assert stats["a"]["norm"]["mean"] == pytest.approx(stats["b"]["norm"]["mean"])
    assert stats["a"]["outliers"]["count"] == 0


def test_n2_dim_mismatch_hard_error():
    a = make_snapshot(make_ids(5), _base(5, 16))
    b = make_snapshot(make_ids(5), _base(5, 8, seed=1))
    with pytest.raises(DimensionMismatchError, match="dimension mismatch"):
        checks.check_n2(a, b)


def test_n2_outlier_flagged():
    ids = make_ids(100)
    v = _base(100, 16)
    v[42] *= 10.0
    stats, findings = checks.check_n2(make_snapshot(ids, v), make_snapshot(ids, v))
    assert stats["a"]["outliers"]["count"] == 1
    assert stats["a"]["outliers"]["examples"][0]["id"] == "chunk_0042"
    severities = {f.message: f.severity for f in findings}
    assert any(s == "red" for s in severities.values()) is False  # 1% -> green band edge
    # 9 outliers with identical norm 40: all must clear |z| > 3 even after
    # the outliers inflate the population std themselves (scaling raw rows
    # leaves their norms spread out and some dip below the threshold).
    v3 = _base(100, 16)
    for r in (10, 11, 12, 13, 14, 15, 16, 17, 42):
        v3[r] = (v3[r] / np.linalg.norm(v3[r])).astype(np.float32) * np.float32(40.0)
    stats2, findings2 = checks.check_n2(
        make_snapshot(ids, v3), make_snapshot(ids, v3)
    )
    assert stats2["a"]["outliers"]["count"] == 9
    assert any(f.severity == "red" for f in findings2)


def test_n2_norm_shift_graded():
    ids = make_ids(80)
    v = _base(80, 16)
    small = make_snapshot(ids, v * 1.03)
    big = make_snapshot(ids, v * 1.5)
    _, f_small = checks.check_n2(make_snapshot(ids, v), small)
    _, f_big = checks.check_n2(make_snapshot(ids, v), big)
    shift_small = [f for f in f_small if "norm mean shift" in f.message][0]
    shift_big = [f for f in f_big if "norm mean shift" in f.message][0]
    assert shift_small.severity == "green"  # 3% <= 5%
    assert shift_big.severity == "red"      # 50% > 20%


# ---------------------------------------------------------------------------
# N4
# ---------------------------------------------------------------------------


def test_n4_finds_injected_duplicates():
    ids = make_ids(100)
    v = _base(100, 16)
    b = v.copy()
    dup_pairs = [(5, 6), (30, 31), (70, 90)]
    for x, y in dup_pairs:
        b[y] = v[x]  # exact copies -> cosine 1.0
    stats, findings = checks.check_n4(
        make_snapshot(ids, b), "B", threshold=0.999
    )
    assert stats["pairs"] == 3
    assert stats["affected_ids"] == 6
    assert len(stats["examples"]) == 3
    assert all(ex["cosine"] >= 0.999 for ex in stats["examples"])
    finding = findings[0]
    # 3 pairs / 100 vectors = 3% >= N4_YELLOW (1%) -> red
    assert finding.severity == "red"


def test_n4_threshold_respected():
    ids = make_ids(60)
    v = _base(60, 16)
    b = v.copy()
    # near-duplicate of v[10]: unit direction plus a 0.15 orthogonal kicker
    # -> cosine = 1/sqrt(1 + 0.15^2) ~ 0.989, below the 0.999 threshold
    unit = (v[10] / np.linalg.norm(v[10])).astype(np.float32)
    perp = _base(16, 16, seed=13)[0].astype(np.float32)
    perp = perp - np.dot(perp, unit) * unit
    perp = (perp / np.linalg.norm(perp)).astype(np.float32)
    b[11] = unit + np.float32(0.15) * perp
    cos = float(np.dot(unit, b[11] / np.linalg.norm(b[11])))
    assert cos < 0.999
    stats, _ = checks.check_n4(make_snapshot(ids, b), "B", threshold=0.999)
    assert stats["pairs"] == 0
    # lowering the threshold catches it — with a margin: check_n4 scores
    # in float32 blocked matmuls, whose rounding differs across BLAS
    # implementations, so a threshold exactly equal to the float64 cos
    # can sit a hair above the float32 similarity (bit us on Linux CI)
    stats2, _ = checks.check_n4(make_snapshot(ids, b), "B", threshold=cos - 1e-6)
    assert stats2["pairs"] >= 1


def test_n4_zero_duplicates_green():
    ids = make_ids(50)
    v = _base(50, 16)
    stats, findings = checks.check_n4(make_snapshot(ids, v), "A")
    assert stats["pairs"] == 0
    assert findings[0].severity == "green"


def test_n4_small_snapshot_yellow_only():
    ids = make_ids(400)
    v = _base(400, 16)
    b = v.copy()
    b[5] = v[4]  # one duplicate pair -> 1/400 = 0.25% -> yellow
    stats, findings = checks.check_n4(make_snapshot(ids, b), "B")
    assert stats["pairs"] == 1
    assert findings[0].severity == "yellow"


def test_n4_duplicate_explosion_counts_exact_and_caps_examples():
    # every vector identical -> n(n-1)/2 pairs; the Python-level example
    # loop must not scale with the pair count
    n = 60
    ids = make_ids(n)
    v = np.tile(_base(1, 16), (n, 1)).astype(np.float32)
    stats, findings = checks.check_n4(make_snapshot(ids, v), "B")
    assert stats["pairs"] == n * (n - 1) // 2
    assert stats["affected_ids"] == n
    assert stats["examples_truncated"] is True
    assert len(stats["examples"]) == 10  # display slice of the capped list
    assert findings[0].severity == "red"


def test_gate_exit_codes():
    from vecdiff.checks import Finding

    assert checks.gate_exit_code([]) == 0
    assert checks.gate_exit_code([Finding("N1", "green", "x")]) == 0
    assert checks.gate_exit_code([Finding("N1", "green", "x"),
                                  Finding("N4", "yellow", "y")]) == 1
    assert checks.gate_exit_code([Finding("N4", "yellow", "y"),
                                  Finding("N1", "red", "z")]) == 2
    assert checks.worst_severity([Finding("N4", "yellow", "y")]) == "yellow"


def test_n1_k_must_be_positive():
    a = make_snapshot(make_ids(10), _base(10, 8))
    with pytest.raises(ValueError, match="k must be >= 1"):
        checks.check_n1(a, a, k=0)


def test_n1_heavy_loss_ids_capped_count_exact():
    # fully independent embedding spaces: every shared id is heavy-loss
    n = 250
    ids = make_ids(n)
    a = make_snapshot(ids, _base(n, 16, seed=1))
    b = make_snapshot(ids, _base(n, 16, seed=2))
    stats, findings = checks.check_n1(a, b, full=True)
    assert stats["heavy_loss_count"] == n
    assert len(stats["heavy_loss_ids"]) == checks.N1_HEAVY_EXAMPLE_CAP
    assert stats["heavy_loss_ids_truncated"] is True
    assert stats["heavy_loss_ids"] == stats["heavy_loss_ids"][: checks.N1_HEAVY_EXAMPLE_CAP]
    assert checks.worst_severity(findings) == "red"


def test_unit_vectors_cached_and_unit_norm():
    ids = make_ids(16)
    snap = make_snapshot(ids, _base(16, 8))
    u1 = snap.unit_vectors()
    u2 = snap.unit_vectors()
    assert u1 is u2  # cached: N1 + N4 share one normalized copy
    np.testing.assert_allclose(np.linalg.norm(u1, axis=1), 1.0, rtol=1e-5)
