#!/usr/bin/env python3
"""Bring new work from the private autolearn repo into this public one, through the scrub.

    python tools/sync_from_private.py sync [--to COMMIT]   # three-way merge of everything new, then check
    python tools/sync_from_private.py check                # scan this repo for private terms and secrets

The word list is not in this repo, or it would publish what it removes. It lives in a private JSON
file (AUTOLEARN_SCRUB_RULES, default ~/.config/autolearn-public/scrub.json):

    {"private_repo": path, "private_branch": name,
     "replace": [[regex, replacement], ...],   # applied in order, multi-line mode
     "deny": [regex, ...],                     # case-insensitive; check fails on any hit
     "deny_not_vendor": [regex, ...]}          # deny entries that upstream vendored code may contain

sync never commits or pushes. Read the diff, run the tests, then commit (docs/publishing.md).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMIT_FILE = ROOT / "tools" / "private_source_commit.txt"
# These are written for the public and only ever change by hand.
PUBLIC_ONLY = {"README.md"}
PATH_MAP = [("docs/superpowers/", "docs/"), ("docs/README.md", "docs/reading-list.md")]
SKIP_SCAN = {"uv.lock", "tools/sync_from_private.py"}
VENDOR = "viz/dashboard/vendor/"
SECRETS = [
    r"\bsk-[A-Za-z0-9_-]{16,}", r"\bgh[oprsu]_[A-Za-z0-9]{20,}", r"\bgithub_pat_[A-Za-z0-9_]{20,}",
    r"\bAKIA[0-9A-Z]{16}\b", r"\bxox[abprs]-[A-Za-z0-9-]{10,}", r"-----BEGIN [A-Z ]*PRIVATE KEY",
    r"(?i)(api[_-]?key|secret|token|passw(or)?d)[\"']?\s*[:=]\s*[\"'][^\"'\s]{8,}",
    r"https?://[^/\s:@]+:[^/\s@]+@",
]


def load_rules(path: str | None = None) -> dict:
    path = path or os.environ.get("AUTOLEARN_SCRUB_RULES") or "~/.config/autolearn-public/scrub.json"
    return json.loads(Path(path).expanduser().read_text())


def public_path(path: str) -> str:
    for old, new in PATH_MAP:
        if path.startswith(old):
            return new + path[len(old):]
    return path


def scrub(data: bytes, rules: dict) -> bytes:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return data
    for pattern, new in rules["replace"]:
        text = re.sub(pattern, new, text, flags=re.MULTILINE)
    return text.encode("utf-8")


def git(repo, *args, check=True) -> bytes:
    return subprocess.run(["git", "-C", str(repo), *args], check=check, capture_output=True).stdout


def show(repo, commit: str, path: str) -> bytes | None:
    r = subprocess.run(["git", "-C", str(repo), "show", f"{commit}:{path}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def merge3(ours: bytes, base: bytes, theirs: bytes) -> tuple[bytes, bool]:
    """git merge-file on the public file, the scrubbed old private file and the scrubbed new one."""
    with tempfile.TemporaryDirectory() as tmp:
        files = [Path(tmp) / n for n in ("ours", "base", "theirs")]
        for f, data in zip(files, (ours, base, theirs)):
            f.write_bytes(data)
        r = subprocess.run(["git", "merge-file", "-p", "-L", "public", "-L", "old private", "-L", "new private",
                            *map(str, files)], capture_output=True)
    return r.stdout, r.returncode != 0


def sync(rules: dict, to: str | None, root: Path = ROOT, commit_file: Path = COMMIT_FILE) -> int:
    repo = rules["private_repo"]
    new = git(repo, "rev-parse", to or rules.get("private_branch", "HEAD")).decode().strip()
    old = commit_file.read_text().strip()
    names = git(repo, "diff", "--name-only", "--no-renames", old, new).decode().splitlines()
    conflicts, manual = [], []
    for path in names:
        if path in PUBLIC_ONLY:
            manual.append(path)
            continue
        target = root / public_path(path)
        before, after = show(repo, old, path), show(repo, new, path)
        if after is None:
            target.unlink(missing_ok=True)
            print(f"removed  {target.relative_to(root)}")
            continue
        after = scrub(after, rules)
        if before is None or not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(after)
            print(f"added    {target.relative_to(root)}")
            continue
        merged, conflict = merge3(target.read_bytes(), scrub(before, rules), after)
        target.write_bytes(merged)
        print(f"{'CONFLICT' if conflict else 'merged  '} {target.relative_to(root)}")
        if conflict:
            conflicts.append(str(target.relative_to(root)))
    commit_file.write_text(new + "\n")
    print(f"\nprivate source now at {new[:10]} ({len(names)} files changed since {old[:10]})")
    for path in manual:
        print(f"written by hand here, so review the private change yourself: {path}")
    if conflicts:
        print("fix the conflict markers in: " + ", ".join(conflicts))
    return check(rules, root) or (1 if conflicts else 0)


def tracked(root: Path) -> list[str]:
    out = git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
    return [p for p in out.decode().split("\0") if p and (root / p).is_file()]


def scan(text: str, patterns: list[str], flags=0):
    for pattern in patterns:
        for m in re.finditer(pattern, text, flags):
            line = text.count("\n", 0, m.start()) + 1
            yield line, pattern, text[max(0, m.start() - 40):m.end() + 40].replace("\n", " ")


def check(rules: dict, root: Path = ROOT) -> int:
    """Non-zero when any tracked file, or the history's author lines, holds a private term or a secret."""
    files, hits = tracked(root), 0
    for path in files:
        data = (root / path).read_bytes()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        deny = rules["deny"]
        if path.startswith(VENDOR):
            deny = [d for d in deny if d not in rules.get("deny_not_vendor", [])]
        found = [] if path == "tools/sync_from_private.py" else list(scan(text, deny, re.IGNORECASE))
        if path not in SKIP_SCAN and not path.startswith(VENDOR):
            found += list(scan(text, SECRETS))
        for line, pattern, context in found:
            print(f"HIT {path}:{line} [{pattern}] {context[:160]}")
            hits += 1
    authors = git(root, "log", "--all", "--format=%an <%ae>%n%cn <%ce>", check=False).decode().splitlines()
    for who in sorted(set(authors)):
        if not who.endswith(("@users.noreply.github.com>", "<noreply@github.com>")):
            print(f"HIT git history identity: {who}")
            hits += 1
    print(f"check: {len(files)} files, {hits} hits" + ("" if hits else ", clean"))
    return 1 if hits else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["sync", "check"])
    p.add_argument("--to", help="private commit to sync to (default: the private branch's tip)")
    args = p.parse_args(argv)
    rules = load_rules()
    return sync(rules, args.to) if args.command == "sync" else check(rules)


if __name__ == "__main__":
    sys.exit(main())
