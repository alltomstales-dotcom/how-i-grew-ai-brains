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
