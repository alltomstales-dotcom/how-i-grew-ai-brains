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
