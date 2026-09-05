#!/usr/bin/env python3
"""Build the case-study snapshots: one real corpus, two real embedding models.

This script is documentation-grade: it runs in *your* environment and needs
``fastembed`` (``pip install fastembed``) — it is NOT a vecdiff dependency
(vecdiff itself stays numpy-only). The models are fixed to two widely used
384-dim embedders so the snapshots are comparable:

  A (blue)  BAAI/bge-small-en-v1.5
  B (green) sentence-transformers/all-MiniLM-L6-v2

Same chunker over the same files on both sides -> identical chunk ids, so N1
measures pure model-swap neighbor drift.

Usage:
    python export_snapshots.py OUT_DIR SRC_LABEL [SRC_LABEL ...]

where SRC_LABEL is ``/path/to/source-dir=label`` (the label becomes the top
path component in chunk metadata, e.g. ``py_myapp/src/main.py``).

Output: OUT_DIR/{A,B}.jsonl + {A,B}.meta.json — the universal jsonl format
vecdiff reads directly.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

CHUNK_LINES = 36  # lines per chunk; no overlap
MODEL_A = "BAAI/bge-small-en-v1.5"
MODEL_B = "sentence-transformers/all-MiniLM-L6-v2"
CODE_SUFFIXES = {".py", ".swift"}


def iter_chunks(sources: list[tuple[Path, str]]):
    """Yield (chunk_id, path, text) over all labeled source dirs."""
    for root, label in sources:
        for f in sorted(root.rglob("*")):
            if f.suffix.lower() not in CODE_SUFFIXES or not f.is_file():
                continue
            rel = f.relative_to(root).as_posix()
            lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
            for start in range(0, len(lines), CHUNK_LINES):
                window = lines[start : start + CHUNK_LINES]
                text = "\n".join(window).strip()
                if not text:
                    continue
                cid = f"{label}:{rel}:{start + 1:05d}"
                yield cid, f"{label}/{rel}", text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument(
        "src_labels",
        nargs="+",
        help="source dirs as /path/to/dir=label (label prefixes chunk paths)",
    )
    args = parser.parse_args()

    sources: list[tuple[Path, str]] = []
    for spec in args.src_labels:
        if "=" not in spec:
            parser.error(f"bad SRC_LABEL {spec!r}: expected /path/to/dir=label")
        raw, label = spec.rsplit("=", 1)
        path = Path(raw).expanduser().resolve()
        if not path.is_dir():
            parser.error(f"not a directory: {path}")
        sources.append((path, label))

    chunks = list(iter_chunks(sources))
    if not chunks:
        print("no code chunks found (expected .py/.swift files)", file=sys.stderr)
        return 1
    ids = [c[0] for c in chunks]
    if len(set(ids)) != len(ids):
        print("internal error: duplicate chunk ids", file=sys.stderr)
        return 1
    texts = [c[2] for c in chunks]
    print(f"{len(chunks)} chunks from {len(sources)} source dirs")

    from fastembed import TextEmbedding  # heavy import, after arg validation

    args.out.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    for tag, model_name in (("A", MODEL_A), ("B", MODEL_B)):
        print(f"embedding with {model_name} ...")
        model = TextEmbedding(model_name)
        vecs = list(model.embed(texts))
        dim = len(vecs[0])
        with open(args.out / f"{tag}.jsonl", "w", encoding="utf-8") as fh:
            for (cid, path, _), vec in zip(chunks, vecs):
                fh.write(
                    json.dumps(
                        {"id": cid, "vector": [float(x) for x in vec], "path": path}
                    )
                    + "\n"
                )
        (args.out / f"{tag}.meta.json").write_text(
            json.dumps(
                {
                    "model": model_name,
                    "created_at": now,
                    "chunking": f"{CHUNK_LINES}-line windows, no overlap",
                    "corpus": [f"{label} <- {root}" for root, label in sources],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"  wrote {tag}.jsonl ({len(chunks)} x {dim})")
    print(f"\nrun:  vecdiff {args.out/'A.jsonl'} {args.out/'B.jsonl'} --full --gate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
