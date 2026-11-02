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
