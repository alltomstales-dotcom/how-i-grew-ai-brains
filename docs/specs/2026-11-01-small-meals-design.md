# Stage 12i: do winter's worlds fade because small grazers are small meals?

Date: 2026-11-01. Written before any of these runs.

## Why

Stage 12h: with winter, grazers shrink to about 0.12 and worlds last in 0.098 of cases. With grazers' size free, they stay at 0.93 and worlds last in 0.934 and 0.844. That change does two things: more grazers get through winter, and a meal is worth 71 instead of 31. So does the meal alone carry it?

## World

One world, `bulk_winter_flat`: `bulk_winter`, with every meal a flat 71 whatever the grazer's size, which is what a meal is worth in `bulk_winter_free`. Grazers pay upkeep as usual. No crossing. Seeds 0 and 1, 150 updates, with one seed on each machine.

## Success test (can fail)

`python -m scripts.persist --run-dir runs/NAME --evolved --steps 2000`, 256 fixed seeds:

1. **The meal carries it:** both species alive at step 1,999 in at least 0.5 of worlds, on both seeds.
2. **Grazers are still small:** grazer size at most 0.4 on both seeds, over the last 15 updates. Otherwise something else changed and the test can't separate them.

Prediction: both bars met. A hunter then eats well from a small grazer, and a small grazer's worth no longer limits it. Reported, not bars: hunter size and numbers, grazers, and the chaser ratio.

## Result (2026-11-01): PASS on both bars, and hunters lose their lead

| | grazer size | hunter size | both alive at step 1,999 | hunters | chaser against learned hunters |
|---|---|---|---|---|---|
| `bulk_winter`, meals 50 x (0.5 + size) | 0.119, 0.120 | 0.68, 0.70 | 0.098, 0.098 | 16.1, 16.9 | 0.70x, 0.66x |
| `bulk_winter_flat`, meals of 71 | **0.123, 0.122** | 0.77, 0.76 | **1.000, 1.000** | 31.6, 31.6 | **1.15x, 1.05x** |

Both bars are met, and the prediction held: with grazers just as small, worlds go from lasting in about a tenth of cases to all of them. So in this small world, it's small meals that make winter's worlds fade, and not winter's cost to hunters directly or too few grazers.

**The cost is hunters' skill.** A fixed chaser now catches more than learned hunters, 1.15x and 1.05x, on both seeds. It was not a bar here, but it would fail stage 12g's third bar. Hunters sit at their cap of 32, and with food this generous, hunting well may not pay. That reading isn't tested. So this world isn't a fix to carry over as it stands, and it's a sign of where the middle lies: meals of about 31 are too small to carry a world, and 71 is enough that skill stops mattering. Stage 12f saw the same drift, with the lead narrowing from 0.74x at meals of 50 to 0.89x at 70.

Grazers also fall to about 50 of 96 at step 600, and they return to 96 at step 1,999, which is the season the evaluation lands on.

