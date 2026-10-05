"""Verify the cluster wrapper preserves failures and publishes useful snapshots."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

SCRIPT = Path(__file__).resolve().parents[2] / "cluster" / "diagnose_run.py"
spec = importlib.util.spec_from_file_location("cluster_diagnostics", SCRIPT)
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


def test_tail_snapshot_is_bounded_and_replaced(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "snapshot"
    source.write_bytes(b"abcdefgh")
    diagnostics.tail_snapshot(source, destination, limit=4)
    assert destination.read_bytes() == b"efgh"
    source.write_bytes(b"new")
    diagnostics.tail_snapshot(source, destination, limit=4)
    assert destination.read_bytes() == b"new"


def test_process_sample_reports_cpu_and_memory():
    sample = diagnostics.process_sample(os.getpid())
    assert sample["cpu_seconds"] >= 0
    assert "VmRSS:" in sample["status"]


def test_wrapper_preserves_exit_status_and_copies_logs_and_config(tmp_path):
    fake = tmp_path / "simulation"
    fake.write_text("""import os, sys
from pathlib import Path
print("simulation progress", flush=True)
print("simulation warning", file=sys.stderr, flush=True)
p = Path(os.environ["STEELO_HOME"]) / "output" / "sim_test"
p.mkdir()
(p / "simulation_config.json").write_text('{"year":2025}')
sys.exit(7)
""")
    env = os.environ.copy()
    env.pop("MASTER_EXCEL", None)
    env["STEELO_HOME"] = str(tmp_path / "home")
    live = tmp_path / "live"
    env["STEELO_LIVE_DIAGNOSTICS"] = str(live)
    result = subprocess.run([sys.executable, str(SCRIPT), str(fake)], env=env, capture_output=True, timeout=20)
    assert result.returncode == 7, result.stderr
    assert b"simulation progress" in result.stdout
    assert b"exited with status 7" in result.stdout
    assert b"simulation warning" not in result.stdout
    assert "simulation progress" in (live / "run.stdout.log").read_text()
    assert "simulation warning" in (live / "run.stderr.log").read_text()
    assert (live / "exit_status").read_text() == "7\n"
    assert json.loads((live / "sim_test/simulation_config.json").read_text()) == {"year": 2025}
    assert json.loads((live / "provenance.json").read_text())["source_sha256"]
    assert (live / "resources.jsonl").exists()


def test_progress_forwards_stages_without_replaying_or_noisy_stderr(tmp_path, capsys):
    stdout = tmp_path / "run.stdout.log"
    stderr = tmp_path / "run.stderr.log"
    stdout.write_text("Preparing data...\npartial")
    stderr.write_text(
        "INFO operation=year_start year=2025\n"
        "INFO operation=allocation_model year=2025 duration_s=12\n"
        "INFO geography fallback\n"
        "INFO operation=memory_checkpoint rss_mb=100\n"
    )
    reporter = diagnostics.ProgressReporter(tmp_path)
    reporter.report()
    result = capsys.readouterr().out
    assert "Preparing data" in result
    assert "year_start" in result
    assert "allocation_model" in result
    assert "partial" not in result
    assert "geography" not in result
    assert "memory_checkpoint" not in result
    reporter.report()
    assert capsys.readouterr().out == ""
    with stdout.open("a") as stream:
        stream.write(" completed\nlast line")
    reporter.report()
    assert capsys.readouterr().out == "partial completed\n"
    reporter.report(final=True)
    assert capsys.readouterr().out == "last line\n"


def test_batch_uses_checkout_python_even_when_activation_has_stale_path(tmp_path):
    project = tmp_path / "project"
    bin_dir = project / ".venv/bin"
    bin_dir.mkdir(parents=True)
    (project / "cluster").mkdir()
    (project / "cluster/diagnose_run.py").write_text("# diagnostic wrapper placeholder\n")
    (bin_dir / "activate").write_text("export VIRTUAL_ENV=/missing/old/project/.venv\n")
    python = bin_dir / "python"
    python.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$SLURM_SUBMIT_DIR/launched_args"\n')
    python.chmod(0o755)
    (bin_dir / "run_simulation").write_text("# console entrypoint placeholder\n")
    # Avoid copying empty results during this launcher-only test.
    (bin_dir / "rsync").write_text("#!/bin/bash\nexit 0\n")
    (bin_dir / "rsync").chmod(0o755)
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("STEELO_", "SLURM_")) or key in {"MASTER_EXCEL", "RUN_NAME"}:
            env.pop(key)
    env.update(
        SLURM_SUBMIT_DIR=str(project),
        SLURM_JOB_ID="test",
        STEELO_SCRATCH_BASE=str(tmp_path / "scratch"),
        PATH=str(bin_dir) + ":" + env["PATH"],
    )
    result = subprocess.run(
        ["bash", str(SCRIPT.with_name("run_simulation.sbatch"))],
        env=env,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr.decode()
    args = (project / "launched_args").read_text().splitlines()
    assert args[:2] == [str(project / "cluster/diagnose_run.py"), str(bin_dir / "run_simulation")]


def test_worker_keeps_crash_handler_without_periodic_sampling(tmp_path, monkeypatch):
    from unittest.mock import patch

    monkeypatch.setenv("STEELO_HOME", str(tmp_path))
    monkeypatch.delenv("STEELO_STACK_SAMPLE_SECONDS", raising=False)
    monkeypatch.setattr(sys, "argv", ["test"])
    (tmp_path / "output").mkdir()
    with (
        patch.object(diagnostics.faulthandler, "enable") as enable,
        patch.object(diagnostics.faulthandler, "disable") as disable,
        patch.object(diagnostics.faulthandler, "dump_traceback_later") as sample,
        patch.object(diagnostics.runpy, "run_path") as run,
    ):
        diagnostics.worker(["simulation", "--end-year", "2027"])
        enable.assert_called_once()
        sample.assert_not_called()
        run.assert_called_once_with("simulation", run_name="__main__")
        disable.assert_called_once()
