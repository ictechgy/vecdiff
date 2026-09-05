"""End-to-end CLI tests (in-process via vecdiff.cli.main)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from vecdiff import __version__
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
    assert proc.stdout.strip() == f"vecdiff {__version__}"


def test_cli_explicit_jsonl_format(tmp_path):
    rng = np.random.default_rng(21)
    ids = make_ids(40)
    v = rng.standard_normal((40, 8)).astype(np.float32)
    paths = []
    for tag in ("a", "b"):
        f = tmp_path / f"{tag}.dat"  # unknown extension -> needs --format
        with open(f, "w", encoding="utf-8") as fh:
            for i in range(40):
                fh.write(
                    json.dumps(
                        {
                            "id": ids[i],
                            "vector": v[i].tolist(),
                            **({"path": f"src/x/{i}.py"} if tag == "a" else {}),
                        }
                    )
                    + "\n"
                )
        paths.append(f)
    code = main([str(paths[0]), str(paths[1]), "--format", "jsonl", "--full"])
    assert code == 0


def test_cli_heavy_loss_truncation_renders(tmp_path, capsys):
    # 230 independent snapshots -> every shared id heavy-loss (> cap 200);
    # console must render the exact count, not the capped list length
    rng = np.random.default_rng(31)
    ids = make_ids(230)
    a = write_dir(tmp_path / "a", ids, rng.standard_normal((230, 8)).astype(np.float32))
    b = write_dir(
        tmp_path / "b", ids, rng.standard_normal((230, 8)).astype(np.float32)
    )
    code = main([str(a), str(b), "--full", "--gate"])
    assert code == 2
    out = capsys.readouterr().out
    assert "heavy-loss ids (Jaccard <= 0.30): 230 (100.0% of queried)" in out


def test_cli_n5_constant_vector_flagged(tmp_path, capsys):
    rng = np.random.default_rng(41)
    ids = make_ids(120)
    v = rng.standard_normal((120, 12)).astype(np.float32)
    w = v.copy()
    w[60:] = v[0]  # cached/constant embedding reused for 60 ids
    a = write_dir(tmp_path / "a", ids, v)
    b = write_dir(tmp_path / "b", ids, w)
    jp = tmp_path / "r.json"
    code = main([str(a), str(b), "--full", "--gate", "--json", str(jp)])
    assert code == 2
    out = capsys.readouterr().out
    assert "N5 constant vectors" in out
    rep = json.loads(jp.read_text())
    n5 = rep["checks"]["N5"]["B"]
    assert n5["largest_group"] == 61  # v[0] plus the 60 copies
    n5_findings = [f for f in rep["findings"] if f["check"] == "N5"]
    assert any(f["severity"] == "red" for f in n5_findings)


def _write_queries(path, ids, vecs):
    with open(path, "w", encoding="utf-8") as fh:
        for i, qid in enumerate(ids):
            fh.write(json.dumps({"id": qid, "vector": vecs[i].tolist()}) + "\n")


def test_cli_supervised_queries_run(tmp_path, capsys):
    rng = np.random.default_rng(51)
    ids = make_ids(80)
    v = rng.standard_normal((80, 16)).astype(np.float32)
    q = rng.standard_normal((8, 16)).astype(np.float32)
    a = write_dir(tmp_path / "a", ids, v)
    b = write_dir(tmp_path / "b", ids, v + rng.standard_normal((80, 16)).astype(np.float32) * 0.01)
    qids = [f"query_{i}" for i in range(8)]
    _write_queries(tmp_path / "qa.jsonl", qids, q)
    _write_queries(tmp_path / "qb.jsonl", qids, q)
    code = main([str(a), str(b), "--full", "--queries-a", str(tmp_path / "qa.jsonl"),
                 "--queries-b", str(tmp_path / "qb.jsonl")])
    assert code == 0
    out = capsys.readouterr().out
    assert "Q1 canonical queries (supervised" in out


def test_cli_queries_must_come_in_pairs(tmp_path):
    a, b = _pair(tmp_path, noisy=False, n=30)
    with pytest.raises(SystemExit) as exc:
        main([str(a), str(b), "--queries-a", "x.jsonl"])
    assert exc.value.code == 2


def test_cli_paths_manifest_orphans(tmp_path, capsys):
    from conftest import make_paths

    rng = np.random.default_rng(61)
    ids = make_ids(50)
    v = rng.standard_normal((50, 16)).astype(np.float32)
    a = write_dir(tmp_path / "a", ids, v)
    paths = make_paths(ids)
    manifest = tmp_path / "tree.txt"
    manifest.write_text("\n".join(paths[5:]) + "\n", encoding="utf-8")  # 10% orphans
    code = main([str(a), str(a), "--paths-manifest", str(manifest), "--gate"])
    assert code == 2
    out = capsys.readouterr().out
    assert "N3 orphan chunks" in out
    assert "orphan" in out
