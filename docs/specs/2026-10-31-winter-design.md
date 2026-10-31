# Stage 9c: do winters make real omnivores?

Date: 2026-10-31. Written before any training run.

## Why

In stages 9a and 9b hunters ended at nearly all meat or nearly all plants, with a sharp switch between guts of 0.08 and 0.20, and mixed diets only on the way there. In ecology, a usual explanation for generalists is food that changes: a specialist has bad seasons. So give this world seasons where plants are worth eating half the time.

## World

`winter_len` 60: every 60 steps the whole map swaps between summer and winter, so an episode of 500 steps has about four of each. In winter plants feed everyone, grazers and hunters, at `winter_value` 0.25 of normal.

- `omni_winter`: `plant_digest` 0.224. For a hunter with diet 0 that is 0.224 in summer and 0.056 in winter, on either side of the switch stage 9b found.
- `omni_d14`, the control: `plant_digest` 0.14 all year, the same yearly average as `omni_winter` (0.224 x 0.625), no winter.

A first probe with random actions tried winters that kill plants (no seeding, plants dying 1 to 3x as fast). Plant cover fell and never recovered, since plants' net growth here is about zero even in summer, so winter thins the food instead and leaves plants alone. No training run informed that choice or these numbers.

## Success test (can fail)

`pp-winter-{0,1}`, `pp-omnid14-{0,1}`, seeds 0 and 1, 150 updates each. `python -m scripts.omnivore --winter runs/W --control runs/C --evolved`, 256 fixed seeds. All must hold on both seeds:

1. **A mixed diet in the winter world:** diet between 0.30 and 0.70, plants between 0.20 and 0.80 of hunter energy.
2. **One mixed diet, not two camps:** spread of the diet gene (sd) at most 0.20.
3. **Winter is why:** diet in `omni_d14` is outside 0.30 to 0.70.

Reported, not bars: hunter and grazer numbers, catches per hunter-step, whether hunters and grazers survive winters at all, `chase.py --evolved`.

## Result (2026-10-31): FAIL on both seeds

| 256 seeds, evolved | seed 0 | seed 1 | bar |
|---|---|---|---|
| diet, winter | 0.109 | 0.116 | 0.30 to 0.70: missed |
| plants' share of hunter energy, winter | 0.936 | 0.929 | 0.20 to 0.80: missed |
| diet spread (sd), winter | 0.102 | 0.113 | at most 0.20: met |
| diet, control (no winter, same yearly gut) | 0.210 | 0.183 | outside 0.30 to 0.70: met |

Winters did not make omnivores. Hunters went further toward plants than the control did, with a plant share of 0.93 against 0.73 and 0.77. A likely reason is that winter hits grazers too: they average 10.8 and 10.6 in the winter world against 31.6 and 31.1 in the control, so meat got scarce as well. That is my reading; meat's value over the seasons wasn't measured. A fixed chaser catches 1.6x and 1.5x what these hunters do.

**The surprise is in the control.** At a gut of 0.14 its diet spread is 0.29 and 0.24, the widest of any stage 9 world. The snapshot from seed 0 at update 150 (4,096 genomes in tenths from 0 to 1) reads 1975, 872, 415, 164, 70, 23, 13, 27, 138, 399: two groups, with a gap in between, and the meat group doubled from update 120 (198 in the top tenth) to 150. Seed 1 reads 2068, 835, 415, 215, 109, 70, 66, 98, 120, 100, with much less of a gap. Those are the two seeds this was noticed on, and neither has been shown to last. Stage 9d tests it on fresh seeds.

