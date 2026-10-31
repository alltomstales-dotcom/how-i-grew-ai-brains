# Stage 8b: ambush (a viewer's suggestion)

Date: 2026-10-30. Written before any run.

## Why

On the first Short, a viewer (@rhetkuehne8613) wrote: "In nature, most predators have the ability to hide, allowing
them to spend less energy chasing down prey." Checking the code, the world could not reward that. Moving cost
the same as standing still, and only hills gave any cover. So hiding could never pay, however smart the hunter.

## World

`ambush` (EvoConfig plus):

- **Moving costs energy.** `move_cost` 0.5 per cell moved, for both species. Standing still is free. A predator's base
  upkeep is also 0.5, so a waiting predator burns about half what a walking one does.
- **Thickets.** `thickets` 0.25: a quarter of the cells, in fixed seamless patches (`thicket_seed`). An animal inside
  one shows only from the next cell, to both species and to its own kind. Plants grow in them as anywhere.
- **Observation.** A new channel shows thicket cells within sight, after the height channel and before the brain gene.

`cover_hides_pred` and `cover_hides_prey` (both on) exist for the test, not for training.

## Success test (can fail)

`pp-ambush-001` (seed 0) and `pp-ambush-002` (seed 1), 30 minutes each. `python -m scripts.ambush --run-dir runs/NAME --evolved`
on 256 fixed seeds. All three must hold, on both seeds:

1. **They use cover:** trained hunters' time in thickets is at least 1.3x the map's share.
2. **They wait:** trained hunters stand still at least 1.5x as often as random ones.
3. **Hiding pays:** the same brains on the same seeds, with thickets no longer hiding predators, catch at least 10% less.

The third is the one that matters. Waiting alone could just mean saving energy, and sitting in thickets could be an accident.

Reported: `chase.py --evolved`, standing still inside against outside thickets, gene means, and a 3-D replay.

## Result (2026-10-30): PASS on both seeds

Evolved bodies, 256 fixed seeds, 30 minutes of training each.

| run | time in thickets (map 25%) | standing still (random 20%) | catches per predator-step | with thickets not hiding predators | drop |
|---|---|---|---|---|---|
| pp-ambush-001 | 65% (2.6x, bar 1.3x) | 90% (4.4x, bar 1.5x) | 0.0069 | 0.0019 | 72% (bar 10%) |
| pp-ambush-002 | 64% (2.5x) | 93% (4.5x) | 0.0056 | 0.0018 | 68% |

**PASS.** In the 3-D replays of both runs, all 17 catches were made by a hunter inside a thicket. In pp-ambush-001, hunter 36 settled on cell (1, 26)
and made four catches in one episode, each by stepping one cell and returning to that same cell on the next step, once after waiting 52 steps.

**But ambush is not stronger than chasing.** `scripts.ambush --chaser`, 96 episodes, same trained prey and evolved bodies:

| run | hunter | catches per step | energy burned per step | energy per catch | episode length |
|---|---|---|---|---|---|
| pp-ambush-001 | trained | 0.0066 | 0.65 | 99 | 108 |
| | one-line chaser | 0.0093 | 1.07 | 115 | 62 |
| pp-ambush-002 | trained | 0.0051 | 0.63 | 123 | 103 |
| | one-line chaser | 0.0088 | 1.05 | 120 | 64 |

The chaser catches 1.4-1.7x as much and burns about two thirds more energy per step, so per meal the two are about even. Learning found a second
way to make the same living, not a better one. Episodes with chasers end sooner, but which species dies out first was not measured.
