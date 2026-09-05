"""vecdiff command-line interface.

Exit codes
----------
0  success (with --gate: all signals green)
1  with --gate: at least one YELLOW finding
2  with --gate: at least one RED finding (blocks cutover);
   also the argparse usage-error convention
3  hard error (bad snapshot, dimension mismatch, I/O failure)
"""

from __future__ import annotations

import argparse
import sys

from . import __version__
from . import checks
from . import report as rpt
from .errors import VecdiffError
from .snapshot import load_snapshot


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vecdiff",
        description=(
            "Diff two embedding-index snapshots and report graded "
            "findings: N1 neighbor stability, N2 population stats, "
            "N4 duplicates. Fully local, deterministic, numpy-only. "
            "vecdiff collects evidence for human judgment; it never "
            "claims one index or model is better."
        ),
    )
    p.add_argument("snapshot_a", help="snapshot A (blue / before): dir, .npz, .db/.sqlite, or .index")
    p.add_argument("snapshot_b", help="snapshot B (green / after)")
    p.add_argument(
        "--k",
        type=int,
        default=10,
        help="neighborhood size for N1 (default: 10)",
    )
    p.add_argument(
        "--sample",
        type=float,
        default=0.2,
        help="fraction of shared ids sampled for N1 (default: 0.2)",
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="use all shared ids for N1 (exact, slower)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed for N1 sampling (default: 0; same seed = same sample)",
    )
    p.add_argument(
        "--dup-threshold",
        type=float,
        default=0.999,
        help="cosine threshold for N4 duplicate pairs (default: 0.999)",
    )
    p.add_argument(
        "--format",
        choices=("auto", "native", "sqlite", "faiss"),
        default="auto",
        help="input format for both snapshots (default: auto-detected)",
    )
    p.add_argument("--json", metavar="OUT", help="also write the report as JSON to OUT")
    p.add_argument("--markdown", metavar="OUT", help="also write the report as Markdown to OUT")
    p.add_argument(
        "--gate",
        action="store_true",
        help="CI gate: exit 2 if any RED finding, 1 if any YELLOW, 0 otherwise",
    )
    p.add_argument("--version", action="version", version=f"vecdiff {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.k < 1:
        parser.error("--k must be >= 1")
    if not (0.0 < args.sample <= 1.0):
        parser.error("--sample must be in (0, 1]")
    if not (0.0 < args.dup_threshold <= 1.0):
        parser.error("--dup-threshold must be in (0, 1]")

    try:
        snap_a = load_snapshot(args.snapshot_a, args.format)
        snap_b = load_snapshot(args.snapshot_b, args.format)

        # N2 first: dimension mismatch is a hard error and must stop before
        # any kNN matrix math.
        n2_stats, n2_findings = checks.check_n2(snap_a, snap_b)
        n1_stats, n1_findings = checks.check_n1(
            snap_a,
            snap_b,
            k=args.k,
            sample=args.sample,
            full=args.full,
            seed=args.seed,
        )
        n4_a_stats, n4_a_findings = checks.check_n4(
            snap_a, "A", threshold=args.dup_threshold
        )
        n4_b_stats, n4_b_findings = checks.check_n4(
            snap_b, "B", threshold=args.dup_threshold
        )

        findings = n1_findings + n2_findings + n4_a_findings + n4_b_findings

        notes: list[str] = []
        if not args.full and n1_stats.get("mode") == "sampled":
            notes.append(
                f"N1 ran on a sample of shared ids ({n1_stats['sampled_ids']}/"
                f"{n1_stats['shared_ids']}, seed {args.seed}); its statistics are "
                "estimates. Rerun with --full for exact values."
            )
        notes.append(
            "N1 neighbors are exact brute-force cosine over each snapshot's "
            "full index (not ANN), computed in blocked matrix multiplications "
            "with bounded memory."
        )
        notes.append(
            "N4 is an exact O(n^2) scan (blocked); for very large indexes "
            "budget time accordingly."
        )
        notes.extend([f"A: {n}" for n in snap_a.notes])
        notes.extend([f"B: {n}" for n in snap_b.notes])

        rep = rpt.build_report(
            snapshot_a_summary=snap_a.summary(),
            snapshot_b_summary=snap_b.summary(),
            params={
                "version": __version__,
                "k": args.k,
                "sample": args.sample,
                "full": bool(args.full),
                "seed": args.seed,
                "dup_threshold": args.dup_threshold,
                "format": args.format,
            },
            n1_stats=n1_stats,
            n2_stats=n2_stats,
            n4_stats={"A": n4_a_stats, "B": n4_b_stats},
            findings=findings,
            notes=notes,
            gate=args.gate,
        )

        if args.json:
            rpt.write_json(rep, args.json)
        if args.markdown:
            rpt.write_markdown(rep, args.markdown)

        print(rpt.render_console(rep))
        if args.json:
            print(f"wrote JSON report: {args.json}")
        if args.markdown:
            print(f"wrote Markdown report: {args.markdown}")
    except VecdiffError as exc:
        print(f"vecdiff: error: {exc}", file=sys.stderr)
        return 3
    except OSError as exc:
        print(f"vecdiff: error: {exc}", file=sys.stderr)
        return 3

    if args.gate:
        return checks.gate_exit_code(findings)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
