# Stage 15d: do mutation zones really keep worlds alive?

Date: 2026-11-01. Written before any of these runs.

## Why

Stage 15c: at about the same average mutation, worlds with mutation zones last far better than worlds with even mutation (0.969, 0.656 against 0.219, 0.000). It wasn't a bar there, I noticed it afterwards, and it's two sets, so it could be chance. This tests it with its own bar on fresh seeds.

## World

Isolated tundra, 256 x 256, no crossing, camouflage gene at 0.10, as stage 15c, `mut_zone_tundra` and `mut_even_tundra`, on seeds 58 and 59, 600 updates. Sets A2 and B2.

## Success test (can fail)

`scripts.persist --evolved --steps 2000 --episodes 32`, both species alive at step 1,999, on both sets:

1. **Zones last better:** `mut_zone_tundra` at least 0.25 above `mut_even_tundra`.
2. **Zones last:** `mut_zone_tundra` at least 0.5.

Prediction: both met. I hold it loosely, since a result noticed after the fact often shrinks on repeat. To check the average, I'll report the share of hunter births in zones if the runs let me estimate it, and otherwise say that they don't. Reported, not bars: hunters' gene, hunter numbers, how widely genes spread.

## Result (2026-11-01): PASS on both bars

All four trainers reached update 600.

| isolated tundra | both alive at step 1,999, A2, B2 | hunters' gene | hunters |
|---|---|---|---|
| `mut_zone_tundra` | **1.000, 1.000** | 0.888, 0.886 | 283, 297 |
| `mut_even_tundra` | 0.719, 0.156 | 0.904, 0.904 | 256, 256 |

1. **Zones last better: met.** By 0.281 and 0.844.
2. **Zones last: met.**

The prediction held, on fresh seeds and with its own bar. Across all four sets, zones last in 0.969, 0.656, 1.000 and 1.000, and even mutation in 0.219, 0.000, 0.719 and 0.156.

**The average is what I said it was.** `scripts.zone_births` counts children by place: grazers 25.1% and 25.0% in zones, hunters 25.8% and 25.0%, on A2 and set Y. So births are even across the map, and the average is about 2x. Zone worlds don't last better by having less mutation overall.

**What differs is who gets it.** Three children in four mutate at 1x, and one in four at 5x. Even mutation gives everyone 2x. Stage 15e asks whether the place matters at all, or only that mix.

