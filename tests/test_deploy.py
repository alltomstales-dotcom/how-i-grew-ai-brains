import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_dash_refuses_when_tailscale_ip_is_empty(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "tailscale").write_text("#!/bin/sh\nexit 0\n")
    (bin_dir / "docker").write_text('#!/bin/sh\necho "docker $*" >> "$DOCKER_LOG"\n')
    for f in bin_dir.iterdir():
        f.chmod(0o755)
    log = tmp_path / "docker.log"
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "DOCKER_LOG": str(log)}
    r = subprocess.run(["bash", str(ROOT / "deploy" / "remote.sh"), "dash"], env=env,
                       capture_output=True, text=True)
    assert r.returncode != 0
    assert "docker run" not in (log.read_text() if log.exists() else "")


def test_dash_binds_the_address_it_is_given(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "docker").write_text('#!/bin/sh\necho "docker $*" >> "$DOCKER_LOG"\n')
    (bin_dir / "docker").chmod(0o755)
    log = tmp_path / "docker.log"
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "DOCKER_LOG": str(log),
           "AUTOLEARN_DASH_IP": "10.1.2.3"}
    r = subprocess.run(["bash", str(ROOT / "deploy" / "remote.sh"), "dash"], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "-p 10.1.2.3:9001:9001" in log.read_text()


def fake_ssh(tmp_path, **extra):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "ssh").write_text('#!/bin/sh\necho "ssh $*" >> "$SSH_LOG"\ncat >/dev/null\n')
    (bin_dir / "ssh").chmod(0o755)
    log = tmp_path / "ssh.log"
    env = {k: v for k, v in os.environ.items() if not k.startswith("AUTOLEARN_")}
    env.update({"PATH": f"{bin_dir}:{os.environ['PATH']}", "SSH_LOG": str(log), **extra})

    def run(*args):
        r = subprocess.run(["bash", str(ROOT / "deploy" / "spark.sh"), *args], env=env, input="",
                           capture_output=True, text=True)
        return r, (log.read_text() if log.exists() else "")
    return run


def test_spark_sh_needs_a_host(tmp_path):
    r, log = fake_ssh(tmp_path)("status")
    assert r.returncode != 0 and "AUTOLEARN_HOST" in r.stderr
    assert log == ""


def test_spark_sh_uses_the_configured_host_and_folder(tmp_path):
    run = fake_ssh(tmp_path, AUTOLEARN_HOST="gpu-box", AUTOLEARN_REMOTE_DIR="work/al")
    r, log = run("train", "--run-id", "x")
    assert r.returncode == 0, r.stderr
    assert log.startswith("ssh gpu-box bash work/al/deploy/remote.sh train --run-id x")
    r, log = run("sync")
    assert r.returncode == 0, r.stderr
    assert "ssh gpu-box mkdir -p work/al && tar -C work/al -xzf -" in log


GIB_KB = 1048576


def fake_host(tmp_path, avail_gib, total_gib=121, running="", cap="", used=""):
    """Fake docker and /proc/meminfo; running = names of trainers holding a memory cap."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "docker").write_text(
        '#!/bin/sh\ncase "$1" in\n'
        '  ps) case "$*" in *label=*) [ -n "$FAKE_RUNNING" ] && printf "%s\\n" $FAKE_RUNNING ;; esac ;;\n'
        '  inspect) echo "$FAKE_CAP" ;;\n'
        '  stats) echo "$FAKE_USED / 121GiB" ;;\n'
        '  *) echo "docker $*" >> "$DOCKER_LOG" ;;\nesac\n')
    (bin_dir / "docker").chmod(0o755)
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(f"MemTotal: {total_gib * GIB_KB} kB\nMemFree: 1 kB\nMemAvailable: {avail_gib * GIB_KB} kB\n")
    log = tmp_path / "docker.log"
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "DOCKER_LOG": str(log),
           "MEMINFO": str(meminfo), "FAKE_RUNNING": running, "FAKE_CAP": cap, "FAKE_USED": used}

    def run(*args):
        r = subprocess.run(["bash", str(ROOT / "deploy" / "remote.sh"), *args], env=env,
                           capture_output=True, text=True)
        return r, (log.read_text() if log.exists() else "")
    return run


def test_budget_keeps_eight_gib_headroom(tmp_path):
    r, _ = fake_host(tmp_path, avail_gib=33)("budget")
    assert r.returncode == 0
    assert "budget_gb=25.0" in r.stdout and "headroom_gb=8" in r.stdout


def test_budget_holds_back_what_running_trainers_may_still_claim(tmp_path):
    # A trainer capped at 12 GiB already using 7 GiB may still grow by 5.
    r, _ = fake_host(tmp_path, avail_gib=26, running="autolearn-train", cap="12", used="7GiB")("budget")
    assert "budget_gb=13.0" in r.stdout and "reserved_gb=5.0" in r.stdout


def test_train_slot_gets_its_own_container_and_memory_cap(tmp_path):
    r, log = fake_host(tmp_path, avail_gib=33)("train", "--slot", "ctl", "--mem-gb", "10", "--run-id", "x")
    assert r.returncode == 0, r.stderr
    line = [l for l in log.splitlines() if l.startswith("docker run")][0]
    assert "--name autolearn-train-ctl" in line and "--label autolearn.mem_gb=10" in line
    assert "XLA_PYTHON_CLIENT_MEM_FRACTION=0.0826" in line
    assert "--mem-limit-gb 10.74" in line and line.endswith("--run-id x")


def test_train_refuses_a_cap_beyond_the_budget(tmp_path):
    r, log = fake_host(tmp_path, avail_gib=15)("train", "--mem-gb", "10", "--run-id", "x")
    assert r.returncode != 0 and "budget" in r.stderr
    assert "docker run" not in log


def test_train_refuses_a_bad_slot_name(tmp_path):
    r, log = fake_host(tmp_path, avail_gib=33)("train", "--slot", "../x", "--run-id", "x")
    assert r.returncode != 0 and "docker run" not in log


def test_analysis_commands_are_capped_and_budget_checked(tmp_path):
    run = fake_host(tmp_path, avail_gib=33)
    for cmd in ("eval", "league", "compare", "species", "chase", "signals"):
        r, log = run(cmd, "--run-dir", "runs/x")
        assert r.returncode == 0, r.stderr
        line = [l for l in log.splitlines() if l.startswith("docker run") and f"scripts.{cmd}" in l][-1]
        assert "XLA_PYTHON_CLIENT_MEM_FRACTION=0.0661" in line  # 8 of 121 GiB


def test_eval_refused_when_it_would_eat_the_headroom(tmp_path):
    r, log = fake_host(tmp_path, avail_gib=15)("eval", "--run-dir", "runs/x")
    assert r.returncode != 0 and "docker run" not in log
