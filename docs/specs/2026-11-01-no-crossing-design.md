# Stage 13e: does camouflage really travel between continents?

Date: 2026-11-01. Written before any of these runs.

## Why

Stage 13c: hunters' camo gene reaches 0.5 in the order meadow, forest, tundra on both sets, at updates 494, 516, 539 and 35, 53, 60, which is the direction animals walk. That looks like one invention crossing the ring. But lineages aren't traced, and meadow could simply get there first by having the most hunters. If crossing carries it, then with no crossing the continents shouldn't sweep near each other.

## World

`camo_meadow` and `camo_tundra` of stage 13c, unchanged, with the same seeds as sets K (52 and 54) and L (55 and 57), and no crossing at all. Forest isn't run: its hunters die out alone, so it would tell us nothing. Sets N (K's seeds) and O (L's seeds), 600 updates.

## Success test (can fail)

For each set, take the first update where hunters' gene reaches 0.5 on meadow and on tundra. With crossing, those were 45 updates apart on K, and 25 on L.

**Crossing matters:** on both sets, meadow and tundra are at least 100 updates apart, or one of them never reaches 0.5 in 600 updates.

If not, then continents that can't exchange a single animal sweep about as close together, and stage 13c's timing doesn't need travel, whatever the order looked like.

Prediction: the bar passes on N, where set K waited about 500 updates. L is my doubt: it swept by update 60, and if an early sweep is easy to reach on these seeds, both continents may do it on their own, which would fail the bar. Reported: gene curves, both species alive at step 1,999, and hunter numbers.

## Result (2026-11-01): PASS, and tundra never gets there alone

All four trainers reached update 600. First update where hunters' gene reaches 0.5:

| | meadow | tundra | apart |
|---|---|---|---|
| set K, with crossing | 494 | 539 | 45 |
| set L, with crossing | 35 | 60 | 25 |
| set N, K's seeds, no crossing | 83 | **never** | no sweep |
| set O, L's seeds, no crossing | 38 | **never** | no sweep |

The bar is met on both sets, including set O, where I doubted it. Final means over the last 60 updates:

| | hunters' gene | both alive at step 1,999 | hunters |
|---|---|---|---|
| meadow N, O, no crossing | 0.939, 0.940 | 1.000, 1.000 | 719, 833 |
| tundra N, O, no crossing | **0.086, 0.140** | **0.000, 0.000** | 98, 110 |
| tundra K, L, with crossing | 0.897, 0.903 | 0.000, 1.000 | 168, 433 |

**So tundra's hunters got camouflage from somewhere else.** Alone, in 600 updates, on two sets, they never reach 0.5. Connected to the other continents, they reach about 0.90, soon after forest does. What travels is the gene, carried by arrivals. That's the reading this control supports, though lineages still aren't traced.

**Connection delivers something climate alone doesn't.** This changes my answer to Tom's question about what the continents are worth: here, a continent ended up with a trait it didn't reach alone.

What this doesn't show:

- That tundra can never invent it. It's 600 updates on two sets, and set K took meadow 494.
- That crossing rescues tundra's worlds. With crossing they last in 0.000 and 1.000, so one set of two.
- Why tundra doesn't get there. Winter's thin budget is an obvious guess, not tested.
- Forest wasn't run alone, so its part in the chain is inferred from timing.

Crossing also seems to slow meadow on K's seeds: 494 against 83 with none, while L's seeds show 35 against 38. With so much chance in when a sweep starts, I wouldn't read much into one pair.

