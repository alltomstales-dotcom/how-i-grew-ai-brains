# Stage 9b: how bad must a gut be to keep a hunter hunting?

Date: 2026-10-31. Written before any run.

## Why

Stage 9a's hunters left meat for plants even with a gut at 0.3, against my prediction. A rough sum says why it might be: hunting paid about 0.18 energy per step (0.0073 catches x 25), and plants pay about 2.45 x `plant_digest` per step (0.49 bites x 5). Those are equal at `plant_digest` of about 0.07. This stage tests that sum: it predicts the switch before seeing it.

## World

`omni` with `plant_digest` 0.03 (`omni_d03`), 0.08 (`omni_d08`), 0.20 (`omni_d20`). Nothing else changes. `pp-omnid03-{0,1}`, `pp-omnid08-{0,1}`, `pp-omnid20-{0,1}`, seeds 0 and 1, 150 updates each.

## Success test (can fail)

`python -m scripts.omnivore --ladder runs/D03 runs/D08 runs/D20 --evolved`, 256 fixed seeds. Both must hold on both seeds:

1. **A bad enough gut keeps them hunting:** diet at 0.03 is at least 0.50.
2. **The switch is below 0.20:** diet at 0.20 is at most 0.25.

Reported, not bars: diet at 0.08, which is near the predicted switch and could go either way; catches per hunter-step, against stage 9a and a fixed chaser; hunter numbers, since hunters who can do neither may die out, and if they die out at 0.03 the diet number means little; share of energy from plants; `chase.py --evolved` at 0.03.

## Result (2026-10-31): PASS on both seeds

| 256 seeds, evolved | seed 0 | seed 1 | bar |
|---|---|---|---|
| diet, gut 0.03 | 0.903 | 0.906 | at least 0.50 |
| diet, gut 0.08 | 0.863 | 0.878 | reported |
| diet, gut 0.20 | 0.067 | 0.069 | at most 0.25 |
| plant share of energy, 0.03 / 0.08 / 0.20 | 0.016 / 0.059 / 0.970 | 0.015 / 0.051 / 0.972 | reported |
| catches per hunter-step, 0.03 / 0.08 / 0.20 | 0.0068 / 0.0072 / 0.0043 | 0.0069 / 0.0070 / 0.0037 | reported |

A bad enough gut does keep hunters hunting, and they hunt well: at gut 0.03 a fixed chaser catches 0.67x and 0.64x what they do, back near stage 5b's level, against about 10x at gut 1.0.

**The switch is sharp, and the rough sum was only roughly right.** It said about 0.07. At 0.08 hunters still get about 95% of their energy from meat, so the switch lies somewhere between 0.08 and 0.20. The sum's bite rate is uncertain too, since hunters take 0.33 bites per step at gut 0.08 and 0.54 at 0.20. These three points don't locate the switch, and none shows a mixed diet.

**The price is numbers.** Hunters average 2.0 to 3.3 across an evaluation episode in all three worlds, against about 31.6 at gut 1.0. This measure is lower than the 7.5 to 12 in the training metrics, probably because evaluation worlds keep running after hunters die out while training worlds reset. That is my reading and I didn't check it directly. The same applies to stage 9a's tough-gut figure (5 in evaluation, about 11 in training). Compare evaluation with evaluation.

Hunters end at nearly all meat or nearly all plants, and the mixed diets seen were on the way there (gut 0.20 at update 61: diet 0.25, sd 0.20 and 0.24). That leaves open whether a stable mixed diet is possible here, which is stage 9c's question.

