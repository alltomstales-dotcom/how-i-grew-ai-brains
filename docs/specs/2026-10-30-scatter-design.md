# Stage 7d: is the map the memory?

Date: 2026-10-30. Written before the test ran. No new training; it uses the Stage 7c brains.

## Why

In the fickle world, brains with no memory recover after a flip, and the good kind climbs to about 0.8 of what surrounds them while the whole map stays at half. That suggests the population remembers by where it lives. It is only a correlation.

## Test (can fail)

`python -m scripts.scatter --run-dir runs/NAME --evolved`, 256 fixed seeds, for pp-life-ff-{0,1} and pp-life-gru-{0,1}. Each world plays until 40 steps after the clock last restarted, then every living grazer is moved to a random open cell, once. Good-bite share over the next 10 steps is compared with the same worlds and seeds without the scatter.

- **The map is the memory (no-memory brains):** scattering drops their good-bite share by at least 0.15, on both seeds.
- **The head is the memory (memory brains):** the drop is less than half the no-memory brain's drop on the same seed, on both seeds.

Both must hold. Reported: both shares, and the good kind's share within 2 cells of grazers, before and after.

## Result (2026-10-30): FAIL. The map is not the memory

| run | good bites, unscattered | scattered | drop (bar 0.15 for no memory) | good kind near grazers, unscattered / scattered |
|---|---|---|---|---|
| pp-life-ff-0 (no memory) | 0.700 | 0.665 | 0.035 | 0.734 / 0.724 |
| pp-life-ff-1 (no memory) | 0.703 | 0.627 | 0.076 | 0.703 / 0.676 |
| pp-life-gru-0 (memory) | 0.860 | 0.810 | 0.050 | 0.457 / 0.467 |
| pp-life-gru-1 (memory) | 0.760 | 0.724 | 0.036 | 0.507 / 0.506 |

Scattering barely hurts no-memory grazers, so where they live does not hold the memory. The hypothesis in the 7c spec was wrong. Also, right after a random scatter the good kind is still about 0.7 of what surrounds them, so the 0.8 could not have come from sorting in space. It suggests a selection among whole worlds instead, tested next with `scripts.scatter --worlds`.

## What it was instead: dead worlds (2026-10-30, `scripts.scatter --worlds`)

The no-memory runs' evolved plant pools are about 85% shrubs (0.848 / 0.811), so each round is rich (the common plant feeds) or poor (the rare one does). The memory run's is about half and half (0.50).

By steps since the clock restarted, pp-life-ff-0 (pp-life-ff-1 in brackets):

| | 0-10 | 10-40 | 40-80 | 80+ |
|---|---|---|---|---|
| good kind's share of edible plants, each living world counted once | 0.507 (0.508) | 0.504 (0.501) | 0.613 (0.586) | 0.800 (0.754) |
| bites in poor worlds | 25,599 (26,209) | 21,477 (26,371) | 4,454 (6,787) | 684 (939) |
| bites in rich worlds | 25,195 (27,272) | 56,124 (63,615) | 58,993 (65,221) | 92,220 (63,472) |
| good-bite share, poor worlds | 0.366 (0.402) | 0.513 (0.533) | 0.676 (0.660) | 0.706 (0.714) |
| good-bite share, rich worlds | 0.645 (0.635) | 0.701 (0.681) | 0.732 (0.715) | 0.770 (0.757) |

**Finding (corrected below).** Most of the no-memory "recovery" is survivorship. Poor worlds end, so later averages come from lucky worlds: about nine in ten at 80+ steps. A rise inside each kind of world remains, about 0.2 of the 0.27 at 40-80 when poor and rich are weighted equally. Grazers dying off inside a world could explain it, but that was not separated here, and this is not evidence of learning.

**What this changes in 7c.** The no-memory curve in that spec should not be read as recovery, and the two brains did not face the same plants when evolved bodies were used. The memory result stands on the cleaner comparison: the wiped control used the same brain and the same plants (0.853 against 0.658, and 0.759 against 0.603), and memory grazers keep taking 0.78-0.82 good bites even in poor worlds. Why the no-memory plant pool tilts to shrubs was not measured.

## Correction (2026-10-30, while making the Part 9 Short)

"Most of the recovery is survivorship" claimed too much. It holds for the plants around grazers: the grazer-weighted plant share tracks the share of bites taken in lucky worlds (0.72, 0.95, 1.0 against 0.72, 0.93, 0.99 on seed 0). It does not hold for good bites. Splitting the rise from 0-10 to 80+ steps into world selection and change inside each kind of world, selection accounts for 12-52% on seed 0 and 9-47% on seed 1, depending on the order of the split. So the rise inside each kind of world may be the larger part, and it remains unexplained. The test's FAIL and the warning about the 7c no-memory curve both stand.
