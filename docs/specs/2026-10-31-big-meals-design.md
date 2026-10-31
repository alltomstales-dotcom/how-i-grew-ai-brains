# Stage 10c: does a big enough meal let trained hunters last?

Date: 2026-10-31. Written before any of these runs.

## Why

Stage 10b's world keeps plenty of grazers, but trained hunters die out: they catch 0.0062 per hunter-step, about 0.16 energy per step at 25 a meal, and burn about 1. A rough sum says they break even at a meal of about 160 (1 / 0.0062), if the catch rate holds. It may not: more hunters may mean fewer grazers or warier ones, and ecology warns that richer predators can crash their own prey.

## World

`lasting` with bigger meals, and a hunter's energy cap raised so a meal isn't clipped: `lasting_m100` (`catch_energy` 100, `pred_energy_cap` 200) and `lasting_m250` (`catch_energy` 250, `pred_energy_cap` 500). Nothing else changes. `pp-m100-{0,1}`, `pp-m250-{0,1}`, seeds 0 and 1, 150 updates each.

## Success test (can fail)

Prediction: 100 falls short and 250 clears it. Scored with `python -m scripts.persist --run-dir runs/NAME --evolved --steps 2000`, 256 fixed seeds:

1. **Big enough lasts:** at 250, both species alive at step 1,999 in at least 0.5 of worlds, on both seeds.
2. **The sum is about right:** at 100, under 0.5 on both seeds.
3. **They still hunt:** at 250, a fixed chaser catches at most 1.0x what learned hunters do (`chase.py --evolved`), on both seeds.

Reported, not bars: hunter and grazer numbers against the caps (32 and 96), whether numbers cycle over 2,000 steps, the catch rate against the 0.0062 the sum assumed.

## Result (2026-10-31): FAIL

| 256 seeds, evolved | seed 0 | seed 1 | bar |
|---|---|---|---|
| both alive at step 1,999, meal 250 | 0.066 | 0.098 | at least 0.5: missed |
| both alive at step 1,999, meal 100 | 0.000 | 0.000 | under 0.5: met |
| fixed chaser against learned hunters, meal 250 | 1.10x | 0.74x | at most 1.0: missed on seed 0 |

A bigger meal helps a lot and doesn't get there. At 250 both species share 0.301 and 0.359 of worlds at step 200, falling to 0.066 and 0.098, against none from step 600 to 1,000 at 100. The sum's break-even of 160 was too hopeful, and its assumed catch rate didn't hold: learned hunters catch 0.0083 at a meal of 100 and 0.0054 and 0.0055 at 250, so they catch less as meals grow, against the 0.0062 assumed.

**Hunters look all-or-nothing.** At 250, hunters per world over all worlds are 2.12 and 3.01 at step 1,999, in 0.066 and 0.098 of worlds, which is about 32 per surviving world, right at the slot cap. So a world either fills up or loses them. Those full worlds keep dying, too, which isn't explained.

Grazers sit at the cap of 96 in every run, with plants about 0.92 of the map. No cycles are visible in these averages, though averaging over worlds would hide them in any one world.

## Probe after the result: which side kills the hunters?

Not a pre-set test. `scripts.persist --evolved --steps 2000`, 256 seeds, swapping a hand-written animal in for one side, on `pp-last-0` (meal 25) and `pp-m250-0` (meal 250). Both species alive at step 1,999:

| grazers | hunters | meal 25 | meal 250 |
|---|---|---|---|
| hand-written | trained | 1.000, 32 hunters | 1.000, 32 hunters |
| trained | hand-written | 0.000 | 0.035 |
| trained | trained | 0.000 | 0.066 |

Trained hunters last in every world against hand-written grazers, and hand-written hunters die against trained ones. So trained grazers are what kill the hunters, on these two runs. That confirms the reading in stage 10b. Part 1 of this series ended on grazers who learned to dodge while hunters starved, and it's the same problem.

