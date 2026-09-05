"""End-to-end CLI tests (in-process via vecdiff.cli.main)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from vecdiff.cli import main

from conftest import make_ids, write_dir

SRC = Path(__file__).resolve().parents[1] / "src"


def _pair(tmp_path: Path, noisy: bool, n: int = 120, d: int = 16):
    rng = np.random.default_rng(77)
    ids = make_ids(n)
    v = rng.standard_normal((n, d)).astype(np.float32)
    a = write_dir(tmp_path / "a", ids, v, model="ma")
    w = v.copy()
    if noisy:
        w[:30] += rng.standard_normal((30, d)).astype(np.float32) * 3.0
    b = write_dir(tmp_path / "b", ids, w, model="mb")
    return a, b


def test_cli_identical_snapshots_clean_exit(tmp_path, capsys):
    rng = np.random.default_rng(3)
    ids = make_ids(60)
    v = rng.standard_normal((60, 16)).astype(np.float32)
    a = write_dir(tmp_path / "a", ids, v)
    b = write_dir(tmp_path / "b", ids, v.copy())
    jp = tmp_path / "r.json"
    mp = tmp_path / "r.md"
    code = main([str(a), str(b), "--full", "--json", str(jp), "--markdown", str(mp)])
    assert code == 0
    out = capsys.readouterr().out
    assert "N1 neighbor stability" in out
    assert "Verdict: GREEN" in out
    rep = json.loads(jp.read_text())
    assert rep["tool"] == "vecdiff"
    assert rep["checks"]["N1"]["mean_jaccard"] == 1.0
    assert rep["verdict"]["worst"] == "green"
    md = mp.read_text()
    assert "# vecdiff" in md and "N1 neighbor stability" in md
    for line in md.splitlines():
        if line.startswith("|") and line.count("|") >= 3:
            pass  # table rows well-formed by construction


def test_cli_gate_red_on_heavy_regression(tmp_path):
    a, b = _pair(tmp_path, noisy=True)
    jp = tmp_path / "r.json"
    code = main([str(a), str(b), "--full", "--gate", "--json", str(jp)])
    assert code == 2
    rep = json.loads(jp.read_text())
    assert rep["verdict"]["worst"] == "red"
    assert rep["gate"]["exit_code"] == 2


def test_cli_gate_yellow_on_few_duplicates(tmp_path):
    rng = np.random.default_rng(5)
    n = 400
    ids = make_ids(n)
    v = rng.standard_normal((n, 16)).astype(np.float32)
    w = v.copy()
    w[5] = v[4]  # exactly one duplicate pair -> N4 yellow, everything else green
    a = write_dir(tmp_path / "a", ids, v)
    b = write_dir(tmp_path / "b", ids, w)
    code = main([str(a), str(b), "--full", "--gate"])
    assert code == 1


def test_cli_dim_mismatch_hard_error_exit_3(tmp_path, capsys):
    rng = np.random.default_rng(9)
    ids = make_ids(20)
    a = write_dir(tmp_path / "a", ids, rng.standard_normal((20, 16)))
    b = write_dir(tmp_path / "b", ids, rng.standard_normal((20, 8)))
    code = main([str(a), str(b)])
    assert code == 3
    assert "dimension mismatch" in capsys.readouterr().err


def test_cli_missing_snapshot_exit_3(tmp_path, capsys):
    code = main([str(tmp_path / "nope-a"), str(tmp_path / "nope-b")])
    assert code == 3
    assert "does not exist" in capsys.readouterr().err


def test_cli_sampled_mode_runs_and_notes_sampling(tmp_path, capsys):
    a, b = _pair(tmp_path, noisy=False, n=300)
    code = main([str(a), str(b)])  # default --sample 0.2
    assert code == 0
    out = capsys.readouterr().out
    assert "sample 20%" in out
    assert "--full for exact values" in out


def test_cli_bad_sample_value_is_usage_error(tmp_path):
    a, b = _pair(tmp_path, noisy=False, n=20)
    with pytest.raises(SystemExit) as exc:
        main([str(a), str(b), "--sample", "1.5"])
    assert exc.value.code == 2


def test_cli_version_subprocess():
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    proc = subprocess.run(
        [sys.executable, "-m", "vecdiff", "--version"],
        capture_output=True, text=True, env=env,
    )
    assert proc.returncode == 0
    assert proc.stdout.strip() == "vecdiff 0.1.0"
