# Publishing

Training happens in a private working repo. This public repo gets the same code and results through a scrub, which
swaps machine names and private paths for neutral wording. The word list is kept outside this repo, because publishing
it would publish what it removes. So these steps need that file, and only the maintainer can run them.

## A new stage

From the root of this repo:

```bash
git pull --rebase                            # a daily workflow may have added video links
python3 tools/sync_from_private.py sync      # three-way merge, then check; no commit, no push
git diff                                     # read it, and fix any conflict markers
UV_PROJECT_ENVIRONMENT=$HOME/.venvs/autolearn-public uv run pytest -q
```

Then update by hand:

- `docs/results.md` and `docs/specs/README.md`: a row for the stage, its bar, its result and its caveats.
- `README.md`: the story, if the stage changes it. `sync` lists the private README when it changes, since this one is
  written separately.
- `SUGGESTIONS.md`: the status and result, if a viewer suggested it.

Finish with `python3 tools/sync_from_private.py check`, which must print `clean`, then commit and push. `tools/` has
the merge logic: for each changed file, it merges the scrubbed old and new private versions into the public one, so
public-only edits (such as the viewer's shareable links) survive.

## A new replay

```bash
rsync <gpu-host>:<code-folder>/runs/NAME/episode3d.json site/watch/NAME/   # a few hundred KB
```

Add `{"id": "NAME", "has_3d": true, "label": "Stage N: what it shows"}` to `site/watch/runs.json`. For a moment worth
linking, add `{"run": "NAME", "follow": ID, "step": N, "caption": "..."}` to `moments` and link to
`watch/?run=NAME&follow=ID&step=N&speed=4&play=1`. A push to `main` republishes the site. To preview it locally:

```bash
python3 tools/build_pages.py && python3 -m http.server 8000 --bind 127.0.0.1 --directory _site
```

Only replays go on the public site. The training dashboard (`deploy/spark.sh dash`) has no login and stays on a
private network; the public site is a separate static copy.

## A new video

`videos.json` is the only place to edit. A scheduled part has `"url": null` and shows "coming <date>", because a private
video would show "unavailable". When a part goes public, link it either way:

- **By hand:** paste the URL into `videos.json`, run `python3 tools/videos.py render`, then commit and push. Or run
  `python3 tools/videos.py go-live`, which reads the scheduled links from a private file and links whichever YouTube
  now shows publicly.
- **Automatically:** the `Link new videos` workflow runs daily at 06:10 UTC and does the same `go-live`, but only if the
  `SCHEDULED_VIDEOS` repository secret is set (a JSON object of part number to URL). Without the secret it does nothing.
  For a new part, add its row to `videos.json` and its link to the secret or the private file.
