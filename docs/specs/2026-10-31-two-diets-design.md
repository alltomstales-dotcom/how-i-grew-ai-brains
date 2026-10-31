# Stage 9d: do hunters split into two kinds?

Date: 2026-10-31. Written before any of these runs.

## Why

Stage 5a tried six times to get one founder pool to split into two species, and failed. Stage 9c's control then showed something unplanned: at a gut of 0.14 on seed 0, hunters' diet genes formed two groups, near all plants and near all meat, with a gap between. Seed 1 showed much less of a gap. I noticed this by looking, so it counts as a question and not a result. Everything below runs on seeds not yet used, for twice as long.

## World

No new rules. `omni_d14` (gut 0.14) and `omni_d11` (gut 0.11, between 0.08, where hunters stay on meat, and 0.14). `pp-diet14-{2,3}` and `pp-diet11-{2,3}`, seeds 2 and 3, 300 updates each, against 150 before, so a split has time to fade.

## Success test (can fail)

For at least one gut, on both seeds, all of:

1. **Two groups, by a bar tighter than stage 5a's (see below):** `python -m scripts.species --run-dir runs/NAME --gene diet`, hunters only, in each of the last 3 snapshots: separation D at least 4, smaller group at least 0.2 of hunters, gap between group means at least 0.6.
2. **Living together, not just in different worlds:** `python -m scripts.omnivore --camps runs/NAME --evolved`, 256 fixed seeds. Of the worlds with hunters alive at step 400 of 500, at least 0.25 hold both a hunter with diet under 0.4 and one over 0.6. The pool is shuffled at reset, so both are usually there at the start; this asks whether both are still there 400 steps later.

The bar allows two guts, so it has two chances, and that is stated here on purpose.

**Validity check, run before training:** the same `species --gene diet` on `pp-omnin-{0,1}`, where the gene does nothing, must not pass. If drift alone can pass test 1, the test means nothing and I'd have to change it before training.

It came close, so I changed it. With stage 5a's bar (D 2, minority 0.2, gap 0.3), `pp-omnin` read D 2.61, 2.61, 2.66, gap 0.47 to 0.49, on seed 0, and D 2.58, 2.40, 2.38, gap 0.42 to 0.44, on seed 1, failing only because the minority dipped to 0.17 and 0.13 in one snapshot each. So drift clears that D and gap every time. The bar above sits clear of those numbers. For comparison, `pp-omnid14-0`, where the idea came from, read D 2.05, 7.36, 8.51, gap 0.44, 0.79, 0.81, minority 0.23, 0.29, 0.14, and `pp-omnid14-1` read D 2.01, 1.77, 1.94. Both fail the new bar, so it can fail. I set it having seen those numbers, which is one more reason it is scored only on fresh seeds.

Reported, not bars: the same test on `pp-omnid14-{0,1}`, where the idea came from; the histogram of diets over time; hunter numbers; the mean energy of each group.

## Result (2026-10-31): FAIL by the rule, with a real split and a test I set badly

**Test 1, two groups, passed at gut 0.11 on both fresh seeds.**

| last 5 of 300 updates | D | smaller group | gap |
|---|---|---|---|
| `pp-diet11-2` | 6.58, 7.12, 5.64, 6.03, 5.92 | 0.45, 0.38, 0.34, 0.27, 0.33 | 0.73 to 0.77 |
| `pp-diet11-3` | 5.21, 5.59, 6.47, 6.40, 6.73 | 0.39, 0.44, 0.44, 0.40, 0.39 | 0.69 to 0.75 |

Gut 0.14 failed it: D is high (up to 9.42) but the smaller group sits at 0.09 to 0.16, under the 0.2 bar, and `pp-diet14-2` dropped to D 1.98 at update 300, so that one wasn't steady.

**Test 2, living together, failed in all four runs, and told us little:** no world had any hunters at step 400. I chose that step without checking how long hunters last. Under the rule the stage fails, and I'm not moving the bar after seeing the result.

**Exploratory, not a pre-set test:** `--camp-curve` on `pp-diet11-{2,3}`, 256 evaluation worlds:

| step | worlds with hunters | with both kinds | plant eaters per world (energy) | meat eaters per world (energy) | grazers |
|---|---|---|---|---|---|
| 0 | 256, 256 | 242, 251 | 5.30, 4.88 (60) | 2.38, 3.00 (60) | 24 |
| 50 | 256, 256 | 227, 236 | 10.00, 9.31 (8.0) | 2.62, 3.72 (14.3, 13.5) | 95, 93 |
| 100 | 123, 143 | 1, 4 | 0.04, 0.05 | 0.70, 0.99 (17.9, 17.5) | 90, 84 |
| 150 | 47, 60 | 0, 0 | 0.00 | 0.24, 0.32 | 60, 55 |
| 200 | 16, 15 | 0, 0 | 0.00 | 0.07, 0.06 | 36, 33 |
| 300 | 0, 1 | 0, 0 | 0.00 | 0.00 | 12, 10 |

So the two kinds share a world for about the first 50 to 75 steps and then take turns. Plant eaters double on the starting plants and crash by step 100, low on energy. Meat eaters outlast them and then die out too. No world keeps hunters to the end, and these evaluation worlds run on after that, so the grazer decline here may not reflect training, where a world restarts when a species dies out.

**What this does and doesn't show.** The pool holds two distinct kinds, steadily, on two fresh seeds, at a separation drift doesn't reach. That is the first split in this project. They aren't two species living side by side. The hypothesis, not yet tested: every restart hands out a rich field of plants, which rewards plant eaters early, while the grazer boom that follows rewards meat eaters, and restarting from a mixed pool is what keeps both going. Stage 9e tests the plant half of that. This also questions earlier hunter results: stage 9b's few hunters look like the same die-out, which I haven't checked.

