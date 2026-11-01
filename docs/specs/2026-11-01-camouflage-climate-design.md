# Stage 13c: does camouflage follow climate?

Date: 2026-11-01. Written before any of these runs.

## Why

Stage 13b: at a price of 0.10, hunters evolve camouflage on both seeds and small worlds last. Climate has shaped sight and size on the continents, and here's a case where a climate should matter directly: forest's thickets already hide hunters from anything but the next cell, so paying for camouflage there buys little. Tundra's winters thin the budget that pays for it.

## World

The three continents of stage 11c, with the same crossing, pace, displacing and lineage tags, with fixed camouflage switched off and the gene on at a price of 0.10, from founders at 0.1, and with no size. Worlds `camo_meadow`, `camo_forest`, `camo_tundra`. Sets K (seeds 52, 53, 54) and L (seeds 55, 56, 57). Checkpoint clips stay off, and every trainer has to reach update 600 for its set to count.

## Success test (can fail)

Means over the last 10% of training, on both sets:

1. **Camouflage follows climate:** hunters' camo gene differs by at least 0.15 between highest and lowest continents, with the same order on both sets.
2. **Worlds still last:** `scripts.persist --evolved --steps 2000 --episodes 32`, both species alive at step 1,999 in at least 0.5 of worlds, on meadow and tundra.
3. **Hunters still hunt:** a fixed chaser catches at most 1.0x what learned hunters do, on meadow and tundra.

Prediction: forest's hunters end lowest, by at least 0.15, with meadow and tundra both near 0.94, so the order between those two may flip and fail bar 1 as it did in stage 12a. Bar 2 is my other doubt, with tundra's thin budget. Hunters in forest are mostly arrivals, which may carry meadow's and tundra's gene in, and that could hide the effect, which would be a result too.

Reported, not bars: grazer sight's order, hunter numbers, and migrants' shares.

## Result (2026-11-01): FAIL on bars 1 and 2, and camouflage spreads in the same order twice

All six trainers reached update 600. Means over the last 60 updates, for sets K and L:

| | meadow | forest | tundra | bar |
|---|---|---|---|---|
| hunters' camo gene | 0.939, 0.939 | 0.931, 0.930 | 0.897, 0.903 | a gap of 0.15: **missed**, 0.042 and 0.036 |
| both alive at step 1,999 | **0.344**, 1.000 | 0.000, 0.000 | **0.000**, 1.000 | meadow and tundra at least 0.5: **missed** on K |
| chaser against learned hunters | 0.82x, 0.64x | 0.97x, 0.61x | 0.80x, 0.64x | meadow and tundra at most 1.0: met |
| grazer sight | 0.580, 0.637 | 0.387, 0.299 | 0.122, 0.093 | reported, same order as every clean set |
| hunters per world in training | 392, 803 | 77, 62 | 168, 433 | reported |

**My prediction was wrong:** forest isn't lowest. Forest's hunters all descend from arrivals, so they carry the gene in even though thickets already hide them. I named that risk beforehand, and it may be what happened, but it isn't tested.

**What I didn't expect is the order and timing of the spread.** The update where hunters' gene first reaches 0.5:

| | meadow | forest | tundra |
|---|---|---|---|
| set K | 494 | 516 | 539 |
| set L | 35 | 53 | 60 |

Meadow, then forest, then tundra, which is the direction animals walk, on both sets, though the sets start about 450 updates apart. On set K nothing moves on any continent for nearly 500 updates, then all three go in about 45. That's what I'd expect if it arises once and travels. It isn't shown yet: lineages aren't traced, and meadow is the likeliest starting place anyway, having the most hunters for much of a run, so two sets agreeing on the first step is weak evidence. The tight timing on set K is the stronger part. A control with no crossing, and tracing the gene through arrivals, would settle it.

**Timing may explain the survival gap.** Set L swept by update 60, and its meadow and tundra last in every world, with hunters at 803 and 433. That's the best continent result so far, with hiding that evolved and at a price. Set K swept with about 100 updates left, and lasts in 0.344 and 0.000. Two sets can't show that's the reason, and the bar stays as it is.

