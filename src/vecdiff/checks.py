"""Checks N1 / N2 / N4 — unsupervised, backend-agnostic, deterministic.

Every signal is reported in the yield-audit findings grammar: a graded
signal (green / yellow / red) with its threshold stated inline. vecdiff
never claims "model B is better"; it collects evidence for a human to
judge.

All thresholds live here as module constants and are documented in the
README's signal table.
"""

from __future__ import annotations

import math
import posixpath
from dataclasses import dataclass, field

import numpy as np

from .errors import DimensionMismatchError
from .knn import block_rows, topk_cosine, topk_cosine_queries
from .snapshot import Snapshot, dir_of

# --- N1 neighbor stability -------------------------------------------------
N1_MEAN_GREEN = 0.90   # mean Jaccard >= this -> green
N1_MEAN_YELLOW = 0.70  # >= this -> yellow, below -> red
N1_LOW_JACCARD = 0.50  # ids below this count as "degraded" (stat only)
N1_HEAVY_LOSS_JACCARD = 0.30  # id "heavily damaged" if Jaccard <= this (>=70% lost)
N1_HEAVY_GREEN = 0.02  # fraction of ids with heavy loss: below -> green
N1_HEAVY_YELLOW = 0.10  # below -> yellow, at/above -> red
N1_RANK_INVERSION_BIG = 5  # per-id max |rank delta| considered a big inversion

# --- N2 population stats ----------------------------------------------------
N2_NORM_SHIFT_GREEN = 0.05   # |meanB-meanA|/meanA <= 5% -> green
N2_NORM_SHIFT_YELLOW = 0.20  # <= 20% -> yellow, above -> red
N2_OUTLIER_Z = 3.0           # norm z-score threshold for outliers
N2_OUTLIER_GREEN = 0.01      # outlier fraction <= 1% -> green
N2_OUTLIER_YELLOW = 0.05     # <= 5% -> yellow, above -> red
N2_NORM_CV_EPS = 1e-6        # norm std/mean below this = constant norms
                              # (pre-normalized embedder): z-scores are float
                              # noise, so the outlier check is skipped

# --- N4 duplicates ----------------------------------------------------------
N4_YELLOW = 0.01  # duplicate pairs / n: 0 -> green, < 1% -> yellow, >= 1% -> red

# --- N5 constant vectors ------------------------------------------------------
# Bit-identical float32 vectors across different chunk ids. Small groups are
# re-chunking accidents (N4 territory); a large group means one embedding
# was reused wholesale — cached/constant-vector pipeline bug semantics.
N5_FLOOD_GROUP = 5        # bit-identical group >= this size -> yellow
N5_FLOOD_FRACTION = 0.05  # largest group >= 5% of n -> red

SEVERITY_ORDER = {"green": 0, "yellow": 1, "red": 2}
N4_EXAMPLE_CAP = 1000  # stored examples cap (pair *count* is always exact)
N1_HEAVY_EXAMPLE_CAP = 200  # heavy-loss ids stored in stats (count is exact)


@dataclass
class Finding:
    check: str
    severity: str  # "green" | "yellow" | "red"
    message: str
    details: dict = field(default_factory=dict)


def worst_severity(findings: list[Finding]) -> str:
    worst = "green"
    for f in findings:
        if SEVERITY_ORDER[f.severity] > SEVERITY_ORDER[worst]:
            worst = f.severity
    return worst


def gate_exit_code(findings: list[Finding]) -> int:
    """--gate semantics: 2 if any RED, 1 if any YELLOW, 0 otherwise."""
    return SEVERITY_ORDER[worst_severity(findings)]


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {
        name: float(np.percentile(arr, q))
        for name, q in (
            ("p10", 10),
            ("p25", 25),
            ("p50", 50),
            ("p75", 75),
            ("p90", 90),
        )
    }


# ---------------------------------------------------------------------------
# N1 — neighbor stability
# ---------------------------------------------------------------------------


def check_n1(
    snap_a: Snapshot,
    snap_b: Snapshot,
    *,
    k: int = 10,
    sample: float = 0.2,
    full: bool = False,
    seed: int = 0,
) -> tuple[dict, list[Finding]]:
    """kNN-graph comparison over shared ids (Vectory-style kNN IoU).

    For each sampled shared id we recompute its exact top-k cosine
    neighbors in each snapshot (candidate pool = that snapshot's full
    index) and score the neighbor-set Jaccard. Also reports rank-inversion
    stats over neighbors present in both lists.
    """
    if int(k) < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    stats: dict = {
        "k_requested": int(k),
        "sample_fraction": None if full else float(sample),
        "seed": int(seed),
    }
    findings: list[Finding] = []

    index_a = snap_a.index_of()
    index_b = snap_b.index_of()
    shared = sorted(set(index_a) & set(index_b))
    only_a = sorted(set(index_a) - set(index_b))
    only_b = sorted(set(index_b) - set(index_a))
    stats["shared_ids"] = len(shared)
    stats["only_in_a"] = len(only_a)
    stats["only_in_b"] = len(only_b)

    if snap_a.n < 2 or snap_b.n < 2 or len(shared) < 2:
        findings.append(
            Finding(
                check="N1",
                severity="yellow",
                message=(
                    f"N1 skipped: needs >= 2 shared ids and >= 2 vectors per "
                    f"snapshot (shared={len(shared)}, A n={snap_a.n}, B n={snap_b.n})"
                ),
                details=stats,
            )
        )
        return stats, findings

    # deterministic sampling of query ids
    rng = np.random.default_rng(seed)
    if full or sample >= 1.0:
        sampled = list(shared)
        mode = "full"
    else:
        frac = min(max(float(sample), 0.0), 1.0)
        size = min(len(shared), max(2, int(math.ceil(len(shared) * frac))))
        sel = np.sort(rng.choice(len(shared), size=size, replace=False))
        sampled = [shared[int(i)] for i in sel]
        mode = "sampled"

    k_eff = min(int(k), snap_a.n - 1, snap_b.n - 1)
    qa = np.array([index_a[i] for i in sampled], dtype=np.int64)
    qb = np.array([index_b[i] for i in sampled], dtype=np.int64)
    neigh_a, _ = topk_cosine(snap_a.vectors, qa, k_eff, unit=snap_a.unit_vectors())
    neigh_b, _ = topk_cosine(snap_b.vectors, qb, k_eff, unit=snap_b.unit_vectors())

    ids_a = snap_a.ids
    ids_b = snap_b.ids
    jaccards: list[float] = []
    inversion_abs: list[int] = []
    ids_with_big_inversion = 0
    degraded = 0
    heavy: list[str] = []

    for r in range(len(sampled)):
        set_a = {ids_a[j] for j in neigh_a[r]}
        set_b = {ids_b[j] for j in neigh_b[r]}
        inter = set_a & set_b
        union = set_a | set_b
        jac = len(inter) / len(union) if union else 1.0
        jaccards.append(jac)
        if jac < N1_LOW_JACCARD:
            degraded += 1
        if jac <= N1_HEAVY_LOSS_JACCARD:
            heavy.append(sampled[r])
        rank_a = {id_: pos for pos, id_ in enumerate(ids_a[j] for j in neigh_a[r])}
        rank_b = {id_: pos for pos, id_ in enumerate(ids_b[j] for j in neigh_b[r])}
        diffs = [abs(rank_a[cid] - rank_b[cid]) for cid in inter]
        inversion_abs.extend(diffs)
        if diffs and max(diffs) >= N1_RANK_INVERSION_BIG:
            ids_with_big_inversion += 1

    mean_j = float(np.mean(jaccards)) if jaccards else 1.0
    heavy_frac = len(heavy) / len(sampled)

    stats.update(
        {
            "mode": mode,
            "k_effective": int(k_eff),
            "sampled_ids": len(sampled),
            "sampled_ids_examples": sampled[:20],
            "mean_jaccard": mean_j,
            "jaccard_percentiles": _percentiles(jaccards),
            "degraded_ids_jaccard_lt_0.5": degraded,
            "heavy_loss_count": len(heavy),
            "heavy_loss_ids": heavy[:N1_HEAVY_EXAMPLE_CAP],
            "heavy_loss_ids_truncated": len(heavy) > N1_HEAVY_EXAMPLE_CAP,
            "heavy_loss_fraction": heavy_frac,
            "rank_inversion": {
                "mean_abs_delta": float(np.mean(inversion_abs)) if inversion_abs else 0.0,
                "max_abs_delta": int(max(inversion_abs)) if inversion_abs else 0,
                "ids_with_inversion_ge_5_fraction": (
                    ids_with_big_inversion / len(sampled)
                ),
            },
        }
    )

    # concentration of heavy-loss ids by directory
    stats["heavy_loss_concentration"] = _concentration(
        heavy, snap_a, snap_b, index_a, index_b
    )

    if mean_j >= N1_MEAN_GREEN:
        sev = "green"
        tail = "neighbor graph stable"
    elif mean_j >= N1_MEAN_YELLOW:
        sev = "yellow"
        tail = "neighborhood drift — inspect heavy-loss ids before cutover"
    else:
        sev = "red"
        tail = "migration regression suspected; inspect heavy-loss ids"
    findings.append(
        Finding(
            check="N1",
            severity=sev,
            message=(
                f"mean neighbor Jaccard {mean_j:.2f} "
                f"(k={k_eff}, {mode}, n={len(sampled)}; thresholds: green >= "
                f"{N1_MEAN_GREEN}, yellow >= {N1_MEAN_YELLOW}) — {tail}"
            ),
            details={"mean_jaccard": mean_j, "mode": mode},
        )
    )

    conc = stats["heavy_loss_concentration"]
    if heavy:
        top_dirs = ", ".join(
            f"{d} ({c})" for d, c in list(conc["by_directory"].items())[:3]
        ) or "no path metadata"
        if heavy_frac >= N1_HEAVY_YELLOW:
            sev = "red"
        elif heavy_frac >= N1_HEAVY_GREEN:
            sev = "yellow"
        else:
            sev = "green"
        findings.append(
            Finding(
                check="N1",
                severity=sev,
                message=(
                    f"{len(heavy)} chunks lost >= 70% of their top-{k_eff} "
                    f"neighbors (Jaccard <= {N1_HEAVY_LOSS_JACCARD}; thresholds: "
                    f"green < {N1_HEAVY_GREEN:.0%}, yellow < {N1_HEAVY_YELLOW:.0%} "
                    f"of queried ids), concentrated in: {top_dirs}"
                ),
                details={"heavy_ids": heavy[:50], "concentration": conc},
            )
        )

    return stats, findings


def _concentration(
    heavy_ids: list[str],
    snap_a: Snapshot,
    snap_b: Snapshot,
    index_a: dict[str, int],
    index_b: dict[str, int],
) -> dict:
    """Group heavy-loss ids by directory of their chunk path (side A paths
    preferred, side B as fallback)."""
    counts: dict[str, int] = {}
    paths_known = snap_a.paths is not None or snap_b.paths is not None
    for cid in heavy_ids:
        path = None
        if snap_a.paths is not None and cid in index_a:
            path = snap_a.paths[index_a[cid]]
        elif snap_b.paths is not None and cid in index_b:
            path = snap_b.paths[index_b[cid]]
        key = dir_of(path) if path else "unknown"
        counts[key] = counts.get(key, 0) + 1
    ordered = dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
    return {
        "paths_available": paths_known,
        "by_directory": ordered,
    }


# ---------------------------------------------------------------------------
# N2 — population stats
# ---------------------------------------------------------------------------


def _pop_stats(snap: Snapshot, side: str) -> dict:
    stats: dict = {
        "side": side,
        "n": snap.n,
        "dim": snap.dim,
        "model": snap.model,
        "norm": {},
        "outliers": {"z_threshold": N2_OUTLIER_Z, "count": 0, "fraction": 0.0, "examples": []},
    }
    if snap.n == 0:
        stats["norm"] = {"note": "no vectors"}
        return stats
    norms = np.linalg.norm(snap.vectors, axis=1)
    mean = float(np.mean(norms))
    std = float(np.std(norms))  # population std, deterministic
    stats["norm"] = {
        "mean": mean,
        "std": std,
        "min": float(np.min(norms)),
        "max": float(np.max(norms)),
    }
    # Constant-norm snapshots (embedders that pre-normalize) make z-scores
    # meaningless: a std of float rounding noise turns 1e-7 wobble into
    # |z| > 3. Skip with an explicit reason instead of reporting noise.
    constant_norms = mean > 0.0 and (std / mean) < N2_NORM_CV_EPS
    if constant_norms:
        z = np.zeros(snap.n)
        out_mask = np.zeros(snap.n, dtype=bool)
    elif std > 0.0:
        z = (norms - mean) / std
        out_mask = np.abs(z) > N2_OUTLIER_Z
    else:
        z = np.zeros(snap.n)
        out_mask = np.zeros(snap.n, dtype=bool)
    count = int(np.count_nonzero(out_mask))
    examples: list[dict] = []
    if count:
        order = np.argsort(-np.abs(z[out_mask]))
        out_ids = np.flatnonzero(out_mask)
        for pos in order[:5]:
            i = int(out_ids[pos])
            examples.append(
                {
                    "id": snap.ids[i],
                    "norm": float(norms[i]),
                    "z": float(z[i]),
                }
            )
    stats["outliers"] = {
        "z_threshold": N2_OUTLIER_Z,
        "count": count,
        "fraction": count / snap.n,
        "examples": examples,
        **(
            {"skipped_reason": "norm variance ~0 (pre-normalized embedder?); "
                               "z-scores would be float noise"}
            if constant_norms
            else {}
        ),
    }
    return stats


def check_n2(snap_a: Snapshot, snap_b: Snapshot) -> tuple[dict, list[Finding]]:
    """Per-side norm stats, dimension consistency (hard error on mismatch),
    and extreme-norm outlier flags (|z| > 3)."""
    stats = {"a": _pop_stats(snap_a, "A"), "b": _pop_stats(snap_b, "B")}
    findings: list[Finding] = []

    if snap_a.n > 0 and snap_b.n > 0 and snap_a.dim != snap_b.dim:
        raise DimensionMismatchError(
            f"dimension mismatch: A dim={snap_a.dim} (model {snap_a.model!r}) vs "
            f"B dim={snap_b.dim} (model {snap_b.model!r}). Cross-dimension "
            "comparison is a hard error for vecdiff v0.1 — re-export both "
            "snapshots with a common dimension."
        )

    findings.append(
        Finding(
            check="N2",
            severity="green",
            message=(
                f"dim consistent across snapshots (dim={snap_a.dim})"
                if snap_a.n > 0 and snap_b.n > 0
                else "dim check skipped (an empty snapshot has no dimension)"
            ),
            details={"dim_a": snap_a.dim, "dim_b": snap_b.dim},
        )
    )

    for side_snap, side_stats in ((snap_a, stats["a"]), (snap_b, stats["b"])):
        if side_snap.n == 0:
            findings.append(
                Finding(
                    check="N2",
                    severity="yellow",
                    message=f"snapshot {side_stats['side']} is empty (0 vectors)",
                    details={},
                )
            )
            continue
        out = side_stats["outliers"]
        if out["count"] == 0:
            reason = out.get("skipped_reason")
            findings.append(
                Finding(
                    check="N2",
                    severity="green",
                    message=(
                        f"{side_stats['side']}: no extreme-norm outliers "
                        f"(|z| > {N2_OUTLIER_Z})"
                        + (f" — {reason}" if reason else "")
                    ),
                    details={},
                )
            )
        else:
            frac = out["fraction"]
            if frac <= N2_OUTLIER_GREEN:
                sev = "green"
            elif frac < N2_OUTLIER_YELLOW:
                sev = "yellow"
            else:
                sev = "red"
            ex = out["examples"][0]
            findings.append(
                Finding(
                    check="N2",
                    severity=sev,
                    message=(
                        f"{side_stats['side']}: {out['count']} extreme-norm "
                        f"outlier(s) ({frac:.2%} of vectors, |z| > {N2_OUTLIER_Z}; "
                        f"thresholds: green <= {N2_OUTLIER_GREEN:.0%}, yellow < "
                        f"{N2_OUTLIER_YELLOW:.0%}), e.g. id {ex['id']!r} "
                        f"(norm {ex['norm']:.3f}, z {ex['z']:+.1f})"
                    ),
                    details=out,
                )
            )

    if snap_a.n > 0 and snap_b.n > 0:
        mean_a = stats["a"]["norm"]["mean"]
        mean_b = stats["b"]["norm"]["mean"]
        shift = abs(mean_b - mean_a) / max(abs(mean_a), 1e-12)
        stats["norm_mean_shift"] = {
            "relative": shift,
            "mean_a": mean_a,
            "mean_b": mean_b,
        }
        if shift <= N2_NORM_SHIFT_GREEN:
            sev = "green"
            tail = "norm distribution stable"
        elif shift <= N2_NORM_SHIFT_YELLOW:
            sev = "yellow"
            tail = "norm distribution shifted — early warning"
        else:
            sev = "red"
            tail = "large norm shift — check embedding pipeline scaling"
        findings.append(
            Finding(
                check="N2",
                severity=sev,
                message=(
                    f"norm mean shift A->B {shift:.1%} "
                    f"(A mu={mean_a:.3f}, B mu={mean_b:.3f}; thresholds: green <= "
                    f"{N2_NORM_SHIFT_GREEN:.0%}, yellow <= {N2_NORM_SHIFT_YELLOW:.0%})"
                    f" — {tail}"
                ),
                details=stats["norm_mean_shift"],
            )
        )

    return stats, findings


# ---------------------------------------------------------------------------
# N4 — duplicates
# ---------------------------------------------------------------------------


def check_n4(
    snap: Snapshot,
    side: str,
    *,
    threshold: float = 0.999,
) -> tuple[dict, list[Finding]]:
    """Pairs with cosine >= threshold *within* one snapshot.

    Exact scan (upper triangle only, each pair counted once), computed in
    blocked matrix multiplications. Time is O(n^2 * dim); for very large
    indexes budget accordingly. Catches re-chunking accidents and
    boilerplate explosions; near-cosine-1.0 pairs between distinct chunks
    also catch pipeline bugs (constant/cached vectors).
    """
    stats: dict = {
        "side": side,
        "n": snap.n,
        "threshold": float(threshold),
        "pairs": 0,
        "affected_ids": 0,
        "examples": [],
        "examples_truncated": False,
    }
    findings: list[Finding] = []
    n = snap.n
    if n < 2:
        findings.append(
            Finding(
                check="N4",
                severity="yellow",
                message=f"N4 skipped for {side}: needs >= 2 vectors (n={n})",
                details=stats,
            )
        )
        return stats, findings

    v = snap.unit_vectors()
    ids_arr = np.asarray(snap.ids)

    pair_count = 0
    examples_seen = 0
    affected: set[str] = set()
    examples: list[dict] = []
    step = block_rows(n)
    for start in range(0, n, step):
        end = min(n, start + step)
        sims = v[start:end] @ v.T  # (b, n)
        upper = np.arange(n)[None, :] > np.arange(start, end)[:, None]
        hits = (sims >= threshold) & upper
        count = int(np.count_nonzero(hits))
        if not count:
            continue
        pair_count += count
        examples_seen += count
        coords = np.argwhere(hits)
        # vectorized affected-id collection: a duplicate explosion is n^2/2
        # pairs, and a Python-level loop over it would dominate the run
        hit_rows = coords[:, 0] + start
        hit_cols = coords[:, 1]
        affected.update(
            np.unique(np.concatenate((ids_arr[hit_rows], ids_arr[hit_cols]))).tolist()
        )
        room = N4_EXAMPLE_CAP - len(examples)
        if room > 0:
            for local_i, j in coords[:room]:
                i = start + int(local_i)
                j = int(j)
                examples.append(
                    {
                        "id_a": snap.ids[i],
                        "id_b": snap.ids[j],
                        "cosine": float(sims[int(local_i), j]),
                        "path_a": (snap.paths[i] if snap.paths else None),
                        "path_b": (snap.paths[j] if snap.paths else None),
                    }
                )
    truncated = examples_seen > len(examples)

    examples.sort(key=lambda ex: (-ex["cosine"], ex["id_a"], ex["id_b"]))
    ratio = pair_count / n
    stats.update(
        {
            "pairs": pair_count,
            "pair_ratio": ratio,
            "affected_ids": len(affected),
            "examples": examples[:10],
            "examples_truncated": truncated,
        }
    )

    if pair_count == 0:
        sev = "green"
        tail = "no duplicates"
    elif ratio < N4_YELLOW:
        sev = "yellow"
        tail = "a few duplicates — usually local accidents"
    else:
        sev = "red"
        tail = "duplicate explosion — re-chunking accident or boilerplate flood"
    top = (
        f", top: {examples[0]['id_a']} ~ {examples[0]['id_b']} "
        f"(cosine {examples[0]['cosine']:.6f})"
        if examples
        else ""
    )
    findings.append(
        Finding(
            check="N4",
            severity=sev,
            message=(
                f"{side}: {pair_count} duplicate pair(s) at cosine >= {threshold} "
                f"({ratio:.2%} of {n} vectors; thresholds: green == 0, yellow < "
                f"{N4_YELLOW:.0%}, red >= {N4_YELLOW:.0%}){top} — {tail}"
            ),
            details={k: stats[k] for k in ("pairs", "pair_ratio", "affected_ids")},
        )
    )
    return stats, findings


# ---------------------------------------------------------------------------
# N5 — constant vectors (pipeline-bug semantics)
# ---------------------------------------------------------------------------


def check_n5(
    snap: Snapshot,
    side: str,
) -> tuple[dict, list[Finding]]:
    """Bit-identical vectors reused across different chunk ids.

    Complements N4: near-duplicate *content* (cosine >= threshold) usually
    means a re-chunking accident, but one embedding reused wholesale —
    a cached API response, a constant fallback vector, a broken batch —
    is a pipeline bug and wants different remediation. Exact float32
    equality via ``np.unique`` is O(n log n * d), no O(n^2) scan needed.
    """
    n = snap.n
    stats: dict = {
        "side": side,
        "n": n,
        "flood_group_min": N5_FLOOD_GROUP,
        "flood_fraction": N5_FLOOD_FRACTION,
        "groups": 0,
        "ids_in_groups": 0,
        "largest_group": 0,
        "largest_group_fraction": 0.0,
        "examples": [],
    }
    findings: list[Finding] = []
    if n < 2:
        findings.append(
            Finding(
                check="N5",
                severity="green",
                message=f"N5 skipped for {side}: needs >= 2 vectors (n={n})",
                details=stats,
            )
        )
        return stats, findings

    _, inverse = np.unique(snap.vectors, axis=0, return_inverse=True)
    counts = np.bincount(inverse.reshape(-1))
    group_sizes = counts[counts >= 2]
    stats["groups"] = int(len(group_sizes))
    stats["ids_in_groups"] = int(group_sizes.sum()) if len(group_sizes) else 0
    largest = int(group_sizes.max()) if len(group_sizes) else 0
    stats["largest_group"] = largest
    stats["largest_group_fraction"] = largest / n
    if largest:
        order = np.argsort(-counts)
        shown = 0
        for group_idx in order:
            if counts[group_idx] < 2 or shown >= 5:
                break
            members = np.flatnonzero(inverse.reshape(-1) == group_idx)
            stats["examples"].append(
                {
                    "size": int(counts[group_idx]),
                    "ids": [snap.ids[i] for i in members[:5]],
                }
            )
            shown += 1

    frac = stats["largest_group_fraction"]
    if largest == 0:
        sev = "green"
        message = (
            f"{side}: no bit-identical vectors (constant-embedding check clean)"
        )
    else:
        if frac >= N5_FLOOD_FRACTION:
            sev = "red"
        elif largest >= N5_FLOOD_GROUP:
            sev = "yellow"
        else:
            sev = "green"
        ex = stats["examples"][0]["ids"][:3] if stats["examples"] else []
        message = (
            f"{side}: {stats['groups']} bit-identical group(s), largest "
            f"{largest} of {n} vectors ({frac:.1%}; thresholds: yellow >= "
            f"{N5_FLOOD_GROUP} members, red >= {N5_FLOOD_FRACTION:.0%} of "
            f"index)"
            + (
                f", e.g. ids {ex} — constant/cached embedding suspected"
                if sev != "green"
                else " — small groups are re-chunking duplicates (see N4)"
            )
        )
    findings.append(
        Finding(check="N5", severity=sev, message=message, details=stats)
    )
    return stats, findings


# ---------------------------------------------------------------------------
# Q1 — supervised canonical queries
# ---------------------------------------------------------------------------

# Thresholds mirror N1's by design (they measure the same quantity — top-k
# neighborhood survival — just over real queries instead of chunk ids).
Q1_MEAN_GREEN = 0.90
Q1_MEAN_YELLOW = 0.70
Q1_HEAVY_LOSS_JACCARD = 0.30
Q1_HEAVY_GREEN = 0.02
Q1_HEAVY_YELLOW = 0.10
Q1_RANK_INVERSION_BIG = 5
Q1_WORST_LISTED = 10


def check_queries(
    snap_a: Snapshot,
    snap_b: Snapshot,
    query_ids: list[str],
    queries_a: np.ndarray,
    queries_b: np.ndarray,
    *,
    k: int = 10,
) -> tuple[dict, list[Finding]]:
    """Run the same canonical queries through both indexes and compare.

    Queries arrive per side, each embedded with that side's model — the
    supervised complement to N1: it measures what real retrieval traffic
    would see, including queries whose neighborhoods were invisible in the
    chunk-id comparison.
    """
    if int(k) < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    stats: dict = {
        "k_requested": int(k),
        "mode": "supervised",
        "queries_given": len(query_ids),
    }
    findings: list[Finding] = []

    for side, snap, queries in (("A", snap_a, queries_a), ("B", snap_b, queries_b)):
        if queries.shape[0] and queries.shape[1] != snap.dim:
            raise DimensionMismatchError(
                f"query vector dim {queries.shape[1]} does not match snapshot "
                f"{side} dim {snap.dim} (model {snap.model!r}) — each side's "
                "queries must be embedded with that side's model."
            )

    # query ids are independent of chunk ids: every supplied query runs
    stats["queries"] = len(query_ids)
    if snap_a.n == 0 or snap_b.n == 0 or len(query_ids) == 0:
        findings.append(
            Finding(
                check="Q1",
                severity="yellow",
                message=(
                    f"Q1 skipped: needs >= 1 query and non-empty snapshots "
                    f"(queries={len(query_ids)}, A n={snap_a.n}, B n={snap_b.n})"
                ),
                details=stats,
            )
        )
        return stats, findings

    k_eff = min(int(k), snap_a.n, snap_b.n)
    neigh_a, _ = topk_cosine_queries(
        snap_a.vectors, queries_a, k_eff, unit=snap_a.unit_vectors()
    )
    neigh_b, _ = topk_cosine_queries(
        snap_b.vectors, queries_b, k_eff, unit=snap_b.unit_vectors()
    )
    ids_a, ids_b = snap_a.ids, snap_b.ids

    jaccards: list[float] = []
    inversion_abs: list[int] = []
    per_query: list[dict] = []
    ids_with_big_inversion = 0
    heavy: list[str] = []
    for qi, qid in enumerate(query_ids):
        set_a = {ids_a[j] for j in neigh_a[qi]}
        set_b = {ids_b[j] for j in neigh_b[qi]}
        inter = set_a & set_b
        union = set_a | set_b
        jac = len(inter) / len(union) if union else 1.0
        jaccards.append(jac)
        if jac <= Q1_HEAVY_LOSS_JACCARD:
            heavy.append(qid)
        rank_a = {id_: pos for pos, id_ in enumerate(ids_a[j] for j in neigh_a[qi])}
        rank_b = {id_: pos for pos, id_ in enumerate(ids_b[j] for j in neigh_b[qi])}
        diffs = [abs(rank_a[cid] - rank_b[cid]) for cid in inter]
        inversion_abs.extend(diffs)
        big = bool(diffs) and max(diffs) >= Q1_RANK_INVERSION_BIG
        ids_with_big_inversion += big
        per_query.append(
            {"id": qid, "jaccard": jac, "max_rank_delta": max(diffs) if diffs else 0}
        )

    mean_j = float(np.mean(jaccards)) if jaccards else 1.0
    heavy_frac = len(heavy) / len(query_ids)
    per_query.sort(key=lambda d: d["jaccard"])
    stats.update(
        {
            "k_effective": int(k_eff),
            "mean_jaccard": mean_j,
            "jaccard_percentiles": _percentiles(jaccards),
            "heavy_loss_queries": heavy[:Q1_WORST_LISTED],
            "heavy_loss_count": len(heavy),
            "heavy_loss_fraction": heavy_frac,
            "rank_inversion": {
                "mean_abs_delta": float(np.mean(inversion_abs)) if inversion_abs else 0.0,
                "max_abs_delta": int(max(inversion_abs)) if inversion_abs else 0,
                "queries_with_inversion_ge_5_fraction": (
                    ids_with_big_inversion / len(query_ids)
                ),
            },
            "worst_queries": per_query[:Q1_WORST_LISTED],
        }
    )

    if mean_j >= Q1_MEAN_GREEN:
        sev = "green"
        tail = "query results stable across indexes"
    elif mean_j >= Q1_MEAN_YELLOW:
        sev = "yellow"
        tail = "query drift — spot-check the worst queries before cutover"
    else:
        sev = "red"
        tail = "query results rebuilt — treat as a cutover-review signal"
    findings.append(
        Finding(
            check="Q1",
            severity=sev,
            message=(
                f"canonical-query mean Jaccard {mean_j:.2f} (k={k_eff}, "
                f"n={len(query_ids)}; thresholds: green >= {Q1_MEAN_GREEN}, "
                f"yellow >= {Q1_MEAN_YELLOW}) — {tail}"
            ),
            details={"mean_jaccard": mean_j},
        )
    )
    if heavy:
        if heavy_frac >= Q1_HEAVY_YELLOW:
            sev = "red"
        elif heavy_frac >= Q1_HEAVY_GREEN:
            sev = "yellow"
        else:
            sev = "green"
        findings.append(
            Finding(
                check="Q1",
                severity=sev,
                message=(
                    f"{len(heavy)} quer{'y' if len(heavy) == 1 else 'ies'} lost "
                    f">= 70% of their top-{k_eff} results (Jaccard <= "
                    f"{Q1_HEAVY_LOSS_JACCARD}; thresholds: green < "
                    f"{Q1_HEAVY_GREEN:.0%}, yellow < {Q1_HEAVY_YELLOW:.0%} of "
                    f"queries), e.g. {heavy[:3]}"
                ),
                details={"heavy_queries": heavy[:50]},
            )
        )
    return stats, findings


# ---------------------------------------------------------------------------
# N3 — orphans (rot audit against a path manifest)
# ---------------------------------------------------------------------------

N3_ORPHAN_YELLOW = 0.05  # orphan fraction: 0 -> green, < 5% -> yellow, >= 5% -> red


def check_orphans(
    snap: Snapshot,
    side: str,
    existing_paths: set[str],
    live_symbols: dict[str, set[str]] | None = None,
) -> tuple[dict, list[Finding]]:
    """Rot audit: orphan chunks (source path gone) and ghost chunks (path
    exists but some of the chunk's declared symbols are gone).

    The manifest is the Snapshot<->symbol-graph join interface (paths are
    the join key). Two levels, same interface:

    * plain text manifest — one existing source path per line (file-level);
    * jsonl manifest — ``{"path", "symbols"}`` records from a symbol-graph
      extractor (cartograph for Swift/iOS emits ``graph --format json`` with
      per-symbol ``location.path`` + ``usr``). Chunks that carry a
      ``symbols`` list are then checked symbol-level: a chunk whose symbols
      disappeared while its file survives is a *ghost* — invisible to the
      file-level check.
    """
    stats: dict = {
        "side": side,
        "n": snap.n,
        "manifest_paths": len(existing_paths),
        "with_path_metadata": 0,
        "symbol_level": bool(live_symbols) and snap.symbols is not None,
        "orphans": 0,
        "ghosts": 0,
        "rot_fraction": 0.0,
        "examples": [],
        "ghost_examples": [],
    }
    findings: list[Finding] = []
    if snap.paths is None:
        findings.append(
            Finding(
                check="N3",
                severity="yellow",
                message=(
                    f"N3 skipped for {side}: no chunk_paths metadata (export "
                    "with per-chunk paths to enable the rot audit)"
                ),
                details=stats,
            )
        )
        return stats, findings

    def norm(p: str) -> str:
        return posixpath.normpath(p.replace("\\", "/"))

    existing = {norm(x) for x in existing_paths}
    live: dict[str, set[str]] = (
        {norm(k): set(v) for k, v in (live_symbols or {}).items()}
        if live_symbols
        else {}
    )
    use_symbols = live and snap.symbols is not None

    orphans: list[str] = []
    ghosts: list[tuple[str, str, list[str]]] = []  # (id, path, dead symbols)
    for pos, (cid, raw) in enumerate(zip(snap.ids, snap.paths)):
        stats["with_path_metadata"] += 1
        path = norm(raw)
        if path not in existing:
            orphans.append(cid)
            continue
        if use_symbols:
            declared = snap.symbols[pos]
            if not declared:
                continue
            dead = sorted(set(declared) - live.get(path, set()))
            if dead:
                ghosts.append((cid, raw, dead))

    rot = len(orphans) + len(ghosts)
    frac = rot / snap.n if snap.n else 0.0
    stats["orphans"] = len(orphans)
    stats["ghosts"] = len(ghosts)
    stats["rot_fraction"] = frac
    stats["examples"] = orphans[:50]
    stats["ghost_examples"] = [
        {"id": cid, "path": path, "dead_symbols": dead[:5]}
        for cid, path, dead in ghosts[:50]
    ]
    if rot == 0:
        sev = "green"
        message = (
            f"{side}: no rot — all {snap.n} chunk paths exist"
            + (
                " and all declared symbols are live "
                f"({len(existing)} manifest paths, symbol-level)"
                if use_symbols
                else f" (file-level, {len(existing)} manifest paths)"
            )
        )
    else:
        if frac >= N3_ORPHAN_YELLOW:
            sev = "red"
        else:
            sev = "yellow"
        parts = []
        if orphans:
            parts.append(
                f"{len(orphans)} orphan chunk(s) (path gone), e.g. "
                f"{[snap.paths[snap.ids.index(c)] for c in orphans[:3]]}"
            )
        if ghosts:
            g = ghosts[0]
            parts.append(
                f"{len(ghosts)} ghost chunk(s) (file alive, symbols gone), "
                f"e.g. {g[0]} lost {g[2][:3]}"
            )
        message = (
            f"{side}: {parts[0]}"
            + (f"; {parts[1]}" if len(parts) > 1 else "")
            + f" — {frac:.1%} of {snap.n} (thresholds: green == 0, yellow < "
            f"{N3_ORPHAN_YELLOW:.0%})"
        )
    findings.append(Finding(check="N3", severity=sev, message=message, details=stats))
    return stats, findings
