"""The public site, the video list and the private-to-public sync (no JAX needed)."""
import json
import subprocess
from pathlib import Path

import pytest

from tools import build_pages, sync_from_private as sync, videos

ROOT = Path(__file__).resolve().parents[1]
WATCH = ROOT / "site" / "watch"


def listing():
    return json.loads((WATCH / "runs.json").read_text())


def test_every_listed_run_has_a_replay():
    for run in listing()["runs"]:
        assert run["has_3d"] and (WATCH / run["id"] / "episode3d.json").is_file()


def test_every_moment_names_a_real_animal_alive_at_its_step():
    for m in listing()["moments"]:
        episode = json.loads((WATCH / m["run"] / "episode3d.json").read_text())
        animal = episode["animals"][str(m["follow"])]
        assert animal["born"] <= m["step"] < animal["died"]
        rows = episode["steps"][m["step"]][animal["kind"]]
        assert m["follow"] in [r[0] for r in rows]


def test_build_puts_the_viewer_beside_the_replays(tmp_path):
    out = build_pages.build(tmp_path / "site")
    for name in ("index.html", "videos.json", "watch/index.html", "watch/runs.json", "watch/vendor/three.module.js",
                 "watch/vendor/LICENSE", "watch/pp-ambush-001/episode3d.json"):
        assert (out / name).is_file(), name


def test_readme_video_table_matches_the_file(tmp_path):
    (tmp_path / "videos.json").write_text((ROOT / "videos.json").read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "README.md").write_text((ROOT / "README.md").read_text(encoding="utf-8"), encoding="utf-8")
    videos.render(tmp_path)
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == (ROOT / "README.md").read_text(encoding="utf-8")


def test_go_live_links_only_what_is_public(tmp_path, monkeypatch):
    (tmp_path / "videos.json").write_text(json.dumps({"series": "s", "videos": [
        {"part": 1, "title": "a", "date": "Oct 1", "url": "https://youtu.be/one"},
        {"part": 2, "title": "b", "date": "Oct 2", "url": None},
        {"part": 3, "title": "c", "date": "Oct 3", "url": None}]}))
    (tmp_path / "README.md").write_text(f"x\n{videos.START}\n{videos.END}\ny\n")
    monkeypatch.setenv("SCHEDULED_VIDEOS", json.dumps({"2": "https://youtu.be/two", "3": "https://youtu.be/three"}))
    assert videos.go_live(tmp_path, check=lambda url: url.endswith("two")) == 1
    data = json.loads((tmp_path / "videos.json").read_text())
    assert [v["url"] for v in data["videos"]] == ["https://youtu.be/one", "https://youtu.be/two", None]
    readme = (tmp_path / "README.md").read_text()
    assert "[Watch](https://youtu.be/two)" in readme and "coming Oct 3" in readme and "three" not in readme


RULES = {"replace": [[r"secret-box", "a GPU machine"]], "deny": [r"secret-box"]}


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@users.noreply.github.com", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repos(tmp_path):
    private, public = tmp_path / "private", tmp_path / "public"
    for r in (private, public):
        r.mkdir()
        git(r, "init", "-q")
    (private / "code.py").write_text("one\ntwo on secret-box\nthree\nfour\nfive\nsix\n")
    (private / "gone.py").write_text("bye\n")
    (private / "README.md").write_text("private readme\n")
    git(private, "add", "-A")
    git(private, "commit", "-qm", "first")
    (public / "code.py").write_text("one, edited in public\ntwo on a GPU machine\nthree\nfour\nfive\nsix\n")
    (public / "gone.py").write_text("bye\n")
    (public / "README.md").write_text("public readme\n")
    commit_file = public / "commit.txt"
    commit_file.write_text(git(private, "rev-parse", "HEAD") + "\n")
    return private, public, commit_file, {**RULES, "private_repo": str(private), "private_branch": "HEAD"}


def test_sync_merges_scrubbed_changes_and_keeps_public_edits(repos):
    private, public, commit_file, rules = repos
    (private / "code.py").write_text("one\ntwo on secret-box\nthree\nfour\nfive\nsix, changed on secret-box\n")
    (private / "new.py").write_text("made on secret-box\n")
    (private / "docs/superpowers/specs").mkdir(parents=True)
    (private / "docs/superpowers/specs/s.md").write_text("spec\n")
    (private / "README.md").write_text("changed private readme\n")
    (private / "gone.py").unlink()
    git(private, "add", "-A")
    git(private, "commit", "-qm", "second")

    assert sync.sync(rules, None, public, commit_file) == 0
    code = (public / "code.py").read_text()
    assert "one, edited in public" in code and "six, changed on a GPU machine" in code and "secret-box" not in code
    assert (public / "new.py").read_text() == "made on a GPU machine\n"
    assert (public / "docs/specs/s.md").read_text() == "spec\n"
    assert (public / "README.md").read_text() == "public readme\n"
    assert not (public / "gone.py").exists()
    assert commit_file.read_text().strip() == git(private, "rev-parse", "HEAD")


def test_sync_reports_a_conflict(repos):
    private, public, commit_file, rules = repos
    (private / "code.py").write_text("one, edited in private\ntwo on secret-box\nthree\nfour\nfive\nsix\n")
    git(private, "add", "-A")
    git(private, "commit", "-qm", "second")
    assert sync.sync(rules, None, public, commit_file) != 0
    assert "<<<<<<< public" in (public / "code.py").read_text()


def test_check_catches_private_terms_secrets_and_identities(repos):
    _, public, _, rules = repos
    assert sync.check(rules, public) == 0
    (public / "leak.md").write_text("trained on Secret-Box\n")
    assert sync.check(rules, public) == 1
    (public / "leak.md").write_text("key = 'sk-" + "a" * 24 + "'\n")
    assert sync.check(rules, public) == 1
    (public / "leak.md").unlink()
    subprocess.run(["git", "-C", str(public), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(public), "-c", "user.name=me", "-c", "user.email=me@example.com", "commit", "-qm", "x"], check=True)
    assert sync.check(rules, public) == 1
