# Dynasties: a lineage tracer for the series (2026-11-04)

**Why.** Series Shorts hold viewers but gain few subscribers, because they report population statistics with no
characters. This tool turns runs into named family lines ("dynasties") a storyteller can follow.

**What it is.** `scripts/lineage.py` (tests `tests/test_dynasty.py`). It is a reading tool: it changes no world rule and
edits no existing file.

```
python -m scripts.lineage --run-dir runs/A [--run-dir runs/B ...] [--origin <gpu-host>:~/autolearn/runs/A ...] \
    [--out-dir runs/lineage] [--episodes 8] [--steps 1000] [--seed 12345] [--frozen] [--replays 8]
python -m scripts.lineage --rank-only --out-dir runs/lineage      # rebuild lines.md only
```

## How it works

- **Replay.** Individual-brain runs (`scripts.train_indiv`, stages 14 to 20) replay through `agents.indiv.make_step`
  with `restart=False`, under the run's own rule (mutation sigma, learning, culture, stage 19 anchoring; the anchor pool
  run must sit beside the run). Founders are drawn the way `scripts.cross_indiv` draws them: bodies from the run's
  `genes_latest.npz`, brains at random from `pop_latest.npz`. `--frozen` swaps in frozen brains. Shared-brain runs
  (`scripts.train`, e.g. continents) apply the run's single brain per species to every animal, so their lines are lines
  of bodies only. Continents replay in isolation, with no migrants.
- **uids at birth.** `StepInfo.prey_parent`/`pred_parent` mark a newborn as a slot whose parent isn't itself. Births come
  after deaths within a step, so a slot that was alive before a step and holds a newborn after it lost its animal in that
  step. Each newborn gets a fresh uid, and its parent uid is the uid then held in the parent's slot. The tracer checks
  every step that its living uids match the world's living slots.
- **Cause of death.** Exact. A thin wrapper around the world's `advance` watches `_predation` as it runs and adds the
  per-slot `eaten` and per-hunter `caught` masks to the step's info (no change to `simulations/evo.py`). Grazers are
  "eaten" or "starved", and hunters always "starved". Each step's eaten count is checked against `StepInfo.catches`;
  `lineage.json` records `cause_of_death: exact` only when every step matched.
- **Lines and moments.** A line is a founder alive at step 0 plus all its descendants. Share is the percentage of the
  line's own species that is alive at each step. The moments are:
  - **rise:** first passes 25, 50 or 80%, counted only if the line started below that level;
  - **takeover:** 100%, with at least two members alive;
  - **fall:** drops to half its peak after the peak;
  - **near_extinction** then **comeback:** down to 2 or fewer members after having had at least 3, then the share it
    regains with at least 3 members alive again;
  - **last_survivor**, then **extinction**, which is flagged `with_species` when the whole species dies at that step;
  - **longest_lived:** the member with the longest life.

  **Population floor.** Shares count toward peaks, rises, takeovers, falls, comebacks and drama only while the species
  has at least half its founders' number alive (4 hunters or 12 grazers in the tiny worlds). Extinction and last
  survivor still use raw counts. Two failures in the first full pass made this necessary:
  - in hunter collapses, the last line was "100% of the hunters" when only two were left, and was scored as falling
    when the species died;
  - a line stuck at one member showed a "comeback" because its rivals died around it.

  Each `series` point also carries the species' living count, so a collapse is visible.
- **Drama** is measured in percentage points of the species and is the sum of three parts:
  - **rise_fall:** the smaller of the climb and the fall after the peak;
  - **comeback:** the smaller of the drop and the recovery;
  - **takeover:** half of the final share gained, counted only when the line ends with at least 50% of its species.
  Takeover counts half because in 1,000 steps one grazer line nearly always takes over, so on its own a takeover is
  the least surprising arc.

## Outputs (`runs/lineage/<run>/`)

- `lineage.json`: run metadata (path, world, mode, seed, episodes, steps, cause check), then `episodes_traced[]`. Each
  episode has:
  - `animals[]`: `{uid, parent_uid, species, born, died, cause}`; `died` and `cause` are null if the animal is alive at
    the end;
  - `species_extinct`;
  - `founders[]`, the founder table. Each row holds `founder` (uid), `name`, `color`, `born`, `slot`, `genes`,
    `descendants`, `children`, `generations`, `share_floor`, `peak_pct`, `peak_step`, `start_pct`, `final_pct`, `final_members`,
    `extinct_step`, `drama`, `drama_parts`, `arc`, `moments[]` (`{kind, step, text, ...}`), `fell_with_species` and
    `series[]` as `[step, percent of species, members, species alive]`;
  - `replay`: the file name of the episode's replay.
- `episode3d_ep<k>.json` for each replayed episode, and the most dramatic one again as `episode3d.json`, in the format that
  `viz/dashboard/world3d.html` already reads. Two additions:
  - every `animals[uid]` gains `parent`, `line` (its founder's uid), `cause` and `alive_at_end`. As in
    `scripts.episode3d`, a survivor's `died` equals the number of frames.
  - top-level `lines` is keyed by founder uid and holds `name`, `color`, `species`, `founder`, `born`, `descendants`,
    `peak_pct`, `peak_step`, `extinct_step`, `drama`, `arc`, `moments` and `series`. Also new at top level: `episode`
    and `lineage`.

  Frame `t` is the world after `t` steps (frame 0 is the start), and `ate` lists hunters that caught something that
  step (exact). To highlight a line, draw every row `[uid, x, y, energy]` of `steps[t].prey`/`.pred` where
  `animals[uid].line == <founder uid>`. Open it with `world3d.html?run=lineage/<run>&file=episode3d.json`.
- `runs/lineage/lines.md`: an overall ranking (at most 3 lines per run), the best three of each arc, and details for
  every line listed (run path, seed, episode, founder, share, moments, replay file).

## Limits

- An episode is not training. The replay starts from the run's final population, drawn at random, so each replayed
  world starts from fresh founders. Lines in a replay are not the lines the run evolved through.
- In anchored worlds (stages 19 and 20 A/B) every grazer born takes a brain from the frozen pool, so grazer lines there
  inherit bodies, not brains. Stage 20 freezes bodies (`gene_sigma=0`), so its lines inherit brains only.
- Founder names are a fixed list that rotates by episode, so storytellers may rename freely.

## First traces (2026-11-04)

Outputs are in `runs/lineage/` in the WSL repo (gitignored). Run copies (config and checkpoints only) are in
`~/autolearn-lineage-runs/`. Tiny runs: 8 episodes of 1,000 steps each, seed 12345, under each run's own rule; the
continent: 2 episodes of 400 steps. Cause of death was exact in every run.

| run | stage | animals traced |
|---|---|---|
| `pp-vis-D-0` | 20 D | 11,505 |
| `pp-vis-A-1` | 20 A, anchored | 1,308 |
| `pp-hof-a50-0` | 19, half anchored | 2,028 |
| `pp-evo-full-0` | 17 | 10,618 |
| `pp-ltw-lamarck-1` | 14c winter, Lamarckian | 16,839 |
| `pp-lts-darwin-1` | 14c stealth, Darwinian | 11,273 |
| `pp-camo_tundra-L` | 13c continent, shared brain | 44,415 |

Top lines:

1. **Flint and Slate**, rival hunter families in `pp-hof-a50-0` episode 6 (replay `episode3d_ep6.json`, founders #27
   and #28).
   - Slate passes 50% at step 111 and is down to 1 living at 353.
   - Slate is back to 75% at 452, the step Flint drops to 1.
   - Flint is back to 75% at 465, and Slate's last member dies at 468.
   - Flint dies out with the last hunter at 492.
2. **Basalt** hunters, `pp-lts-darwin-1` episode 1, founder #24.
   - Passes 25% at step 84 and 50% at 204, peaking at 67% at 283.
   - Falls to half by 308 and is down to 1 living at 451.
   - Back to 38% at 516, then dies out at 532, while the species lasts until 871.

Hunter numbers here are small: 3 to 6 alive around these moments. A share of 75% can mean 3 of 4.

Grazer takeovers are near certain: in most 1,000-step episodes one founder's line becomes 95 to 100% of the grazers.
Continent lines start at 0.15% each (682 founders) and none reaches 25% in 400 steps, so continents need much longer
replays to tell line stories.
