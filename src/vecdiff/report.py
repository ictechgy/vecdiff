"""Report rendering: console table, JSON, Markdown, plus gate semantics."""

from __future__ import annotations

import json
from pathlib import Path

from .checks import Finding, gate_exit_code, worst_severity

_DISCLAIMER = (
    "vecdiff collects evidence for human judgment; it never claims one "
    "index or model is better."
)

_TAG = {"green": "[GREEN]", "yellow": "[YELLOW]", "red": "[RED]  "}


def build_report(
    *,
    snapshot_a_summary: dict,
    snapshot_b_summary: dict,
    params: dict,
    n1_stats: dict,
    n2_stats: dict,
    n4_stats: dict,
    n5_stats: dict,
    findings: list[Finding],
    q1_stats: dict | None = None,
    n3_stats: dict | None = None,
    notes: list[str],
    gate: bool,
) -> dict:
    """Assemble the plain-dict report (JSON-safe) from check outputs."""
    counts = {"red": 0, "yellow": 0, "green": 0}
    for f in findings:
        counts[f.severity] += 1
    worst = worst_severity(findings)
    code = gate_exit_code(findings)
    shared = n1_stats.get("shared_ids", 0)
    return {
        "tool": "vecdiff",
        "version": params["version"],
        "snapshots": {"a": snapshot_a_summary, "b": snapshot_b_summary},
        "params": {k: v for k, v in params.items() if k != "version"},
        "shared_ids": shared,
        "checks": {
            "N1": n1_stats,
            "N2": n2_stats,
            "N4": n4_stats,
            "N5": n5_stats,
            **({"Q1": q1_stats} if q1_stats is not None else {}),
            **({"N3": n3_stats} if n3_stats is not None else {}),
        },
        "findings": [
            {
                "check": f.check,
                "severity": f.severity,
                "message": f.message,
                "details": f.details,
            }
            for f in findings
        ],
        "notes": notes,
        "verdict": {
            "worst": worst,
            "counts": counts,
            "statement": (
                f"{worst.upper()}: {counts['red']} red, {counts['yellow']} yellow, "
                f"{counts['green']} green finding(s)"
            ),
            "disclaimer": _DISCLAIMER,
        },
        "gate": {
            "enabled": bool(gate),
            "exit_code": code if gate else None,
            "semantics": "2 = any RED, 1 = any YELLOW, 0 = all green",
        },
    }


# ---------------------------------------------------------------------------
# console
# ---------------------------------------------------------------------------


def _fmt(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def render_console(rep: dict) -> str:
    a = rep["snapshots"]["a"]
    b = rep["snapshots"]["b"]
    params = rep["params"]
    n1 = rep["checks"]["N1"]
    lines: list[str] = []

    lines.append(f"vecdiff {rep['version']} - embedding index diff")
    lines.append(
        f"  A {a['path']}  (adapter={a['adapter']}, model={a['model']}, "
        f"dim={a['dim']}, n={a['n']}, created={a['created_at'] or 'unknown'})"
    )
    lines.append(
        f"  B {b['path']}  (adapter={b['adapter']}, model={b['model']}, "
        f"dim={b['dim']}, n={b['n']}, created={b['created_at'] or 'unknown'})"
    )
    if "shared_ids" in n1:
        lines.append(
            f"  shared ids: {n1['shared_ids']} "
            f"(only in A: {n1['only_in_a']}, only in B: {n1['only_in_b']})"
        )

    lines.append("")
    lines.append(
        f"N1 neighbor stability - cosine kNN k={n1.get('k_effective', params['k'])}, "
        f"queries={n1.get('sampled_ids', 0)} "
        + (
            "(full)"
            if n1.get("mode") == "full"
            else f"(sample {params['sample']:.0%}, seed {params['seed']})"
        )
    )
    if "mean_jaccard" in n1:
        pct = n1["jaccard_percentiles"]
        inv = n1["rank_inversion"]
        lines.append(
            f"  mean Jaccard {_fmt(n1['mean_jaccard'])} | "
            f"p10 {_fmt(pct.get('p10', 0))} p25 {_fmt(pct.get('p25', 0))} "
            f"p50 {_fmt(pct.get('p50', 0))} p75 {_fmt(pct.get('p75', 0))} "
            f"p90 {_fmt(pct.get('p90', 0))}"
        )
        lines.append(
            f"  rank inversion: mean |dRank| {_fmt(inv['mean_abs_delta'], 2)}, "
            f"max {inv['max_abs_delta']}, ids with inversion >= 5: "
            f"{inv['ids_with_inversion_ge_5_fraction']:.1%}"
        )
        lines.append(
            f"  heavy-loss ids (Jaccard <= 0.30): "
            f"{n1.get('heavy_loss_count', len(n1['heavy_loss_ids']))} "
            f"({n1['heavy_loss_fraction']:.1%} of queried)"
        )
    else:
        lines.append("  (skipped - see findings)")

    lines.append("")
    n2a, n2b = rep["checks"]["N2"]["a"], rep["checks"]["N2"]["b"]
    lines.append("N2 population stats")
    lines.append(f"  {'':>14} {'A':>12} {'B':>12}")
    lines.append(f"  {'n':>14} {n2a['n']:>12} {n2b['n']:>12}")
    lines.append(f"  {'dim':>14} {n2a['dim']:>12} {n2b['dim']:>12}")
    if n2a["n"] and n2b["n"]:
        na, nb = n2a["norm"], n2b["norm"]
        lines.append(
            f"  {'norm mean':>14} {_fmt(na['mean']):>12} {_fmt(nb['mean']):>12}"
        )
        lines.append(
            f"  {'norm std':>14} {_fmt(na['std']):>12} {_fmt(nb['std']):>12}"
        )
        lines.append(
            f"  {'norm min/max':>14} "
            f"{_fmt(na['min']) + '/' + _fmt(na['max']):>12} "
            f"{_fmt(nb['min']) + '/' + _fmt(nb['max']):>12}"
        )
        for side_stats in (n2a, n2b):
            out = side_stats["outliers"]
            if out["count"]:
                ex = out["examples"][0]
                lines.append(
                    f"  outliers {side_stats['side']}: {out['count']} "
                    f"({out['fraction']:.2%}, |z| > {out['z_threshold']}), e.g. "
                    f"{ex['id']} (norm {ex['norm']:.3f}, z {ex['z']:+.1f})"
                )
            else:
                lines.append(
                    f"  outliers {side_stats['side']}: 0 (|z| > {out['z_threshold']})"
                )

    lines.append("")
    n4a, n4b = rep["checks"]["N4"]["A"], rep["checks"]["N4"]["B"]
    lines.append(f"N4 duplicates (cosine >= {n4a['threshold']})")
    for side_stats in (n4a, n4b):
        top = ""
        if side_stats["examples"]:
            ex = side_stats["examples"][0]
            top = f", top: {ex['id_a']} ~ {ex['id_b']} (cosine {ex['cosine']:.6f})"
        lines.append(
            f"  {side_stats['side']}: {side_stats['pairs']} pair(s) "
            f"({side_stats.get('pair_ratio', 0):.2%} of {side_stats['n']} vectors), "
            f"{side_stats['affected_ids']} affected ids{top}"
        )

    lines.append("")
    n5a, n5b = rep["checks"]["N5"]["A"], rep["checks"]["N5"]["B"]
    lines.append("N5 constant vectors (bit-identical reuse)")
    for side_stats in (n5a, n5b):
        lines.append(
            f"  {side_stats['side']}: {side_stats['groups']} group(s), "
            f"{side_stats['ids_in_groups']} ids, largest "
            f"{side_stats['largest_group']} "
            f"({side_stats['largest_group_fraction']:.1%} of {side_stats['n']})"
        )

    if "Q1" in rep["checks"]:
        q1 = rep["checks"]["Q1"]
        lines.append("")
        lines.append(
            f"Q1 canonical queries (supervised, k={q1.get('k_effective', '?')}, "
            f"n={q1.get('queries', 0)})"
        )
        if "mean_jaccard" in q1:
            pct = q1["jaccard_percentiles"]
            inv = q1["rank_inversion"]
            lines.append(
                f"  mean Jaccard {_fmt(q1['mean_jaccard'])} | "
                f"p10 {_fmt(pct.get('p10', 0))} p50 {_fmt(pct.get('p50', 0))} "
                f"p90 {_fmt(pct.get('p90', 0))}"
            )
            lines.append(
                f"  rank inversion: mean |dRank| {_fmt(inv['mean_abs_delta'], 2)}, "
                f"max {inv['max_abs_delta']}, queries with inversion >= 5: "
                f"{inv['queries_with_inversion_ge_5_fraction']:.1%}"
            )
            lines.append(
                f"  heavy-loss queries (Jaccard <= 0.30): {q1['heavy_loss_count']} "
                f"({q1['heavy_loss_fraction']:.1%} of queries)"
            )
        else:
            lines.append("  (skipped - see findings)")

    if "N3" in rep["checks"]:
        lines.append("")
        lines.append("N3 rot audit (vs paths manifest)")
        for side_stats in rep["checks"]["N3"].values():
            if "rot_fraction" in side_stats:
                level = "symbol-level" if side_stats.get("symbol_level") else "file-level"
                lines.append(
                    f"  {side_stats['side']}: {side_stats['orphans']} orphan(s), "
                    f"{side_stats.get('ghosts', 0)} ghost(s) "
                    f"({side_stats['rot_fraction']:.1%} of {side_stats['n']}, "
                    f"{level}, manifest {side_stats['manifest_paths']} paths)"
                )
            else:
                lines.append(f"  {side_stats['side']}: skipped (no path metadata)")

    lines.append("")
    lines.append("Findings (graded signals; thresholds inline)")
    for f in rep["findings"]:
        lines.append(f"  {_TAG[f['severity']]} {f['check']}  {f['message']}")

    lines.append("")
    verdict = rep["verdict"]
    lines.append(f"Verdict: {verdict['statement']}")
    lines.append(f"  {_DISCLAIMER}")
    if rep["gate"]["enabled"]:
        lines.append(
            f"  gate: exit code {rep['gate']['exit_code']} ({rep['gate']['semantics']})"
        )

    if rep["notes"]:
        lines.append("")
        lines.append("Notes (method honesty):")
        for note in rep["notes"]:
            lines.append(f"  - {note}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|")


def render_markdown(rep: dict) -> str:
    a = rep["snapshots"]["a"]
    b = rep["snapshots"]["b"]
    n1 = rep["checks"]["N1"]
    n2a, n2b = rep["checks"]["N2"]["a"], rep["checks"]["N2"]["b"]
    n4a, n4b = rep["checks"]["N4"]["A"], rep["checks"]["N4"]["B"]
    params = rep["params"]
    out: list[str] = []

    out.append(f"# vecdiff {rep['version']} report")
    out.append("")
    out.append("## Snapshots")
    out.append("")
    out.append("| side | path | adapter | model | dim | n | created_at |")
    out.append("|---|---|---|---|---|---|---|")
    for side, s in (("A", a), ("B", b)):
        out.append(
            f"| {side} | {_md_escape(s['path'])} | {s['adapter']} | "
            f"{_md_escape(str(s['model']))} | {s['dim']} | {s['n']} | "
            f"{s['created_at'] or 'unknown'} |"
        )
    if "shared_ids" in n1:
        out.append("")
        out.append(
            f"Shared ids: **{n1['shared_ids']}** "
            f"(only in A: {n1['only_in_a']}, only in B: {n1['only_in_b']})"
        )

    out.append("")
    out.append("## Checks")
    out.append("")
    out.append("### N1 neighbor stability (kNN Jaccard, Vectory-style)")
    out.append("")
    if "mean_jaccard" in n1:
        mode = n1.get("mode", "full")
        pct = n1["jaccard_percentiles"]
        inv = n1["rank_inversion"]
        if mode == "full":
            queries_desc = "full"
        else:
            queries_desc = (
                f"sample {params['sample']:.0%}, seed {params['seed']}"
            )
        out.append(
            f"- queries: {n1['sampled_ids']} shared ids ({queries_desc})"
        )
        out.append(f"- mean Jaccard: **{n1['mean_jaccard']:.3f}**")
        out.append(
            "- Jaccard percentiles: "
            + ", ".join(f"{k} {v:.3f}" for k, v in pct.items())
        )
        out.append(
            f"- rank inversion: mean |dRank| {inv['mean_abs_delta']:.2f}, max "
            f"{inv['max_abs_delta']}, ids with inversion >= 5: "
            f"{inv['ids_with_inversion_ge_5_fraction']:.1%}"
        )
        heavy_count = n1.get("heavy_loss_count", len(n1["heavy_loss_ids"]))
        if heavy_count:
            out.append(
                f"- heavy-loss ids (Jaccard <= 0.30): {heavy_count} "
                f"({n1['heavy_loss_fraction']:.1%})"
            )
            conc = n1.get("heavy_loss_concentration", {})
            if conc.get("by_directory"):
                out.append(
                    "- concentrated in: "
                    + ", ".join(
                        f"`{d}` ({c})"
                        for d, c in list(conc["by_directory"].items())[:5]
                    )
                )
    else:
        out.append("- skipped (see findings)")

    out.append("")
    out.append("### N2 population stats")
    out.append("")
    out.append("| stat | A | B |")
    out.append("|---|---|---|")
    out.append(f"| n | {n2a['n']} | {n2b['n']} |")
    out.append(f"| dim | {n2a['dim']} | {n2b['dim']} |")
    if n2a["n"] and n2b["n"]:
        na, nb = n2a["norm"], n2b["norm"]
        out.append(f"| norm mean | {na['mean']:.3f} | {nb['mean']:.3f} |")
        out.append(f"| norm std | {na['std']:.3f} | {nb['std']:.3f} |")
        out.append(f"| norm min | {na['min']:.3f} | {nb['min']:.3f} |")
        out.append(f"| norm max | {na['max']:.3f} | {nb['max']:.3f} |")
        out.append(
            f"| outliers (|z| > {n2a['outliers']['z_threshold']}) "
            f"| {n2a['outliers']['count']} | {n2b['outliers']['count']} |"
        )

    out.append("")
    out.append(f"### N4 duplicates (cosine >= {n4a['threshold']})")
    out.append("")
    out.append("| side | pairs | ratio | affected ids |")
    out.append("|---|---|---|---|")
    for side_stats in (n4a, n4b):
        out.append(
            f"| {side_stats['side']} | {side_stats['pairs']} | "
            f"{side_stats.get('pair_ratio', 0):.2%} | {side_stats['affected_ids']} |"
        )

    out.append("")
    n5a, n5b = rep["checks"]["N5"]["A"], rep["checks"]["N5"]["B"]
    out.append("### N5 constant vectors (bit-identical reuse)")
    out.append("")
    out.append("| side | groups | ids in groups | largest group | largest / n |")
    out.append("|---|---|---|---|---|")
    for side_stats in (n5a, n5b):
        out.append(
            f"| {side_stats['side']} | {side_stats['groups']} | "
            f"{side_stats['ids_in_groups']} | {side_stats['largest_group']} | "
            f"{side_stats['largest_group_fraction']:.1%} |"
        )

    if "Q1" in rep["checks"]:
        q1 = rep["checks"]["Q1"]
        out.append("")
        out.append("### Q1 canonical queries (supervised)")
        out.append("")
        if "mean_jaccard" in q1:
            inv = q1["rank_inversion"]
            out.append(f"- queries: {q1['queries']} (k={q1['k_effective']})")
            out.append(f"- mean Jaccard: **{q1['mean_jaccard']:.3f}**")
            out.append(
                f"- rank inversion: mean |dRank| {inv['mean_abs_delta']:.2f}, max "
                f"{inv['max_abs_delta']}, queries with inversion >= 5: "
                f"{inv['queries_with_inversion_ge_5_fraction']:.1%}"
            )
            if q1["heavy_loss_count"]:
                out.append(
                    f"- heavy-loss queries (Jaccard <= 0.30): "
                    f"{q1['heavy_loss_count']} ({q1['heavy_loss_fraction']:.1%})"
                )
        else:
            out.append("- skipped (see findings)")

    if "N3" in rep["checks"]:
        out.append("")
        out.append("### N3 rot audit (vs paths manifest)")
        out.append("")
        out.append("| side | orphans | ghosts | rot fraction | manifest paths |")
        out.append("|---|---|---|---|---|")
        for side_stats in rep["checks"]["N3"].values():
            if "rot_fraction" in side_stats:
                out.append(
                    f"| {side_stats['side']} | {side_stats['orphans']} | "
                    f"{side_stats.get('ghosts', 0)} | "
                    f"{side_stats['rot_fraction']:.1%} | "
                    f"{side_stats['manifest_paths']} |"
                )

    out.append("")
    out.append("## Findings")
    out.append("")
    out.append("| severity | check | finding |")
    out.append("|---|---|---|")
    for f in rep["findings"]:
        out.append(
            f"| {f['severity'].upper()} | {f['check']} | {_md_escape(f['message'])} |"
        )
    verdict = rep["verdict"]
    out.append("")
    out.append(f"**Verdict: {verdict['statement']}**")
    out.append("")
    out.append(f"> {_DISCLAIMER}")
    if rep["gate"]["enabled"]:
        out.append("")
        out.append(
            f"Gate: exit code **{rep['gate']['exit_code']}** "
            f"({rep['gate']['semantics']})."
        )

    if rep["notes"]:
        out.append("")
        out.append("## Notes (method honesty)")
        out.append("")
        for note in rep["notes"]:
            out.append(f"- {note}")

    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# output helpers
# ---------------------------------------------------------------------------


def write_json(rep: dict, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(rep, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def write_markdown(rep: dict, path: str | Path) -> None:
    Path(path).write_text(render_markdown(rep), encoding="utf-8")
