#!/usr/bin/env python3
"""The video list: videos.json is the only place to edit; this writes the README's table from it.

    python tools/videos.py render    # after editing videos.json by hand
    python tools/videos.py go-live   # link any scheduled part that YouTube now shows publicly

A part stays "coming <date>" with no link until it is public, because a scheduled video shows
"unavailable". go-live takes the scheduled links from SCHEDULED_VIDEOS, a JSON object of part number
to URL (a repository secret in the daily workflow), or from the file SCHEDULED_VIDEOS_FILE
(default ~/.config/autolearn-public/scheduled_videos.json). They are not kept in this repo.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START, END = "<!-- videos:start -->", "<!-- videos:end -->"


def table(videos: list[dict]) -> str:
    rows = ["| Part | Title | Watch |", "|---|---|---|"]
    for v in videos:
        watch = f"[Watch]({v['url']})" if v.get("url") else f"coming {v['date']}"
        rows.append(f"| {v['part']} | {v['title']} | {watch} |")
    return "\n".join(rows)


def render(root: Path = ROOT) -> None:
    videos = json.loads((root / "videos.json").read_text(encoding="utf-8"))["videos"]
    readme = root / "README.md"
    text = readme.read_text(encoding="utf-8")
    head, rest = text.split(START)
    _, tail = rest.split(END)
    readme.write_text(f"{head}{START}\n{table(videos)}\n{END}{tail}", encoding="utf-8")


def is_public(url: str) -> bool:
    """YouTube's oEmbed answers 200 for a public video and 401 or 403 for a private one."""
    query = urllib.parse.urlencode({"url": url, "format": "json"})
    try:
        with urllib.request.urlopen(f"https://www.youtube.com/oembed?{query}", timeout=20) as r:
            return r.status == 200
    except urllib.error.HTTPError:
        return False


def scheduled() -> dict:
    if os.environ.get("SCHEDULED_VIDEOS"):
        return json.loads(os.environ["SCHEDULED_VIDEOS"])
    path = Path(os.environ.get("SCHEDULED_VIDEOS_FILE", "~/.config/autolearn-public/scheduled_videos.json")).expanduser()
    return json.loads(path.read_text()) if path.exists() else {}


def go_live(root: Path = ROOT, check=is_public) -> int:
    """Fill in the link of every part that has gone public; returns how many."""
    path = root / "videos.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    urls, changed = scheduled(), 0
    for v in data["videos"]:
        url = urls.get(str(v["part"]))
        if v.get("url") or not url:
            continue
        if check(url):
            v["url"] = url
            changed += 1
        print(f"part {v['part']}: {'now linked' if v.get('url') else 'not public yet'}")
    if changed:
        lines = ",\n".join("    " + json.dumps(v, ensure_ascii=False) for v in data.pop("videos"))
        head = json.dumps(data, indent=2, ensure_ascii=False)[:-2]
        path.write_text(f'{head},\n  "videos": [\n{lines}\n  ]\n}}\n', encoding="utf-8")
        render(root)
    return changed


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args == ["render"]:
        render()
    elif args == ["go-live"]:
        print(f"{go_live()} newly linked")
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
