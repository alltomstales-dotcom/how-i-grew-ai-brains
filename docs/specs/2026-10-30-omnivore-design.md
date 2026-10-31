# Stage 9a: hunters who can eat plants

Date: 2026-10-30. Written before any run.

## Why

A viewer asked on the first Short whether a predator's edge is its meal, and Tom answered about omnivores. Stage 5b then showed hunters catching about once per 140 predator-steps, about 0.19 energy per step at 25 a meal, well under what they burn. So hunting barely pays here, and plants are everywhere. If a hunter's gut can change, does it keep hunting?

## World

A sixth gene, `diet`, used only by hunters and only in these worlds. A hunter with diet d keeps d of every catch, and on an unbitten plant left by the grazers it takes (1 - d) x `plant_digest` of what a grazer would. Grazers eat first. Hunters see their own diet gene. Founders start at 0.5.

- `omni`: `plant_digest` 1.0.
- `omni_tough`: `plant_digest` 0.3, a gut built badly for plants.
- `omni_neutral`, the control: `plant_digest` 1.0, but every hunter gets 0.5 of each whatever its gene says, so the gene only drifts.

## Success test (can fail)

`pp-omni-{0,1}`, `pp-omnit-{0,1}`, `pp-omnin-{0,1}`, seeds 0 and 1, 150 updates each. `python -m scripts.omnivore --omni runs/A --tough runs/B --neutral runs/C --evolved`, 256 fixed seeds, measures the mean diet of living hunters. Both must hold on both seeds:

1. **Hunters leave meat:** diet in `omni` is at least 0.15 below `omni_neutral`.
2. **A poor gut keeps them hunting:** diet in `omni_tough` is at least 0.10 above `omni`.

Reported, as predictions and not bars: catches per hunter-step (I expect them to fall by half or more in `omni`), share of hunter energy from plants, hunter and grazer numbers, plant cover, the spread of the diet gene (one mixed diet, or two camps), `chase.py --evolved`.

## Result (2026-10-30): FAIL, on the second bar

| 256 seeds, evolved | seed 0 | seed 1 | bar |
|---|---|---|---|
| diet, `omni` | 0.107 | 0.098 | |
| diet, `omni_neutral` | 0.475 | 0.445 | |
| below neutral | 0.369 | 0.347 | at least 0.15: met |
| diet, `omni_tough` | 0.050 | 0.049 | |
| tough above `omni` | -0.056 | -0.050 | at least 0.10: missed, wrong way |

**Hunters left meat, fast and almost completely.** Diet fell from 0.5 to about 0.10 by update 38 and stayed there. Plants give them all but a rounding error of their energy (1.00 on both seeds, rounded). Hunters sit at about 31.6 of 32 slots. Catches per hunter-step are 0.0010, against 0.0073 and 0.0067 for hunters with no plants in stage 5b's control, a different world with its own settings. A fixed chaser catches 9.7x and 10.2x what these trained hunters do, where in stage 5b it caught about 0.7x. They have stopped being hunters.

**The poor gut did the opposite of my prediction.** I expected it to keep them hunting. Diet ended lower, at 0.05, and hunters paid in numbers: 5.1 and 5.2 of them, against 31.6. Grazers rose to about 29 from about 21. Those hunters do catch more (0.0026 and 0.0025), but plants are still 0.99 of their energy. The path differs too: at update 38 their diet was still 0.47 and 0.45 with a wide spread (0.23 and 0.19) and plants were 0.57 and 0.64 of their energy, so they lived as real omnivores for a while before going to plants.

**The control is informative.** In `omni_neutral` the gene does nothing and every hunter gets half of each, and plants are still 0.97 of their energy, with 0.0014 catches per hunter-step. So the brains stop hunting even when evolution can't change the gut. Its diet stays at about the founders' 0.5 with a wide spread (0.30 and 0.28), which looks like drift, as designed.

One camp, not two: `omni` sits piled against 0 (sd about 0.10), which is what a floor plus mutation would give. There's no sign of a split, though the spread alone can't rule one out, and `species.py` wasn't run on this gene.

Inference, not measured: hunting paid about 0.18 energy per step in stage 5b, and plants pay 2.45 x `plant_digest` per step at the tough runs' 0.49 bites per hunter-step, so even a gut of 0.3 pays about 0.7. That would explain why 0.3 wasn't poor enough, and sets up stage 9b.

