"""Tests for scripts/make_demo_snapshots.py."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from vecdiff.snapshot import load_snapshot

SRC = Path(__file__).resolve().parents[1] / "src"
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_demo_snapshots.py"


def test_demo_script_creates_loadable_snapshots(tmp_path):
    target = tmp_path / "demo"
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), str(target)],
        capture_output=True, text=True, env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert (target / "snapA").is_dir()
    assert (target / "snapB").is_dir()

    a = load_snapshot(target / "snapA")
    b = load_snapshot(target / "snapB")
    assert a.n == 300 and a.dim == 32
    assert b.n == 300 and b.dim == 32
    assert len(set(a.ids) & set(b.ids)) == 295  # 5 renamed ids in B
    assert a.model == "demo-model-v1"
    assert b.model == "demo-model-v2"

    # determinism: same seed -> identical bytes of metadata ids
    target2 = tmp_path / "demo2"
    subprocess.run(
        [sys.executable, str(SCRIPT), str(target2)],
        capture_output=True, text=True, env=env, check=True,
    )
    a2 = load_snapshot(target2 / "snapA")
    assert a.ids == a2.ids
    assert (a.vectors == a2.vectors).all()
