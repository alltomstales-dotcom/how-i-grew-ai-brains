# Stage 12g: with a race that lasts, does size follow climate?

Date: 2026-11-01. Written before any of these runs.

## Why

Stage 12a asked whether a body part that matters follows climate, and couldn't answer: size wiped out the hunters. Stage 12f found a world where size decides catches and both species last, in the small map, at meals of 50. So this asks 12a's question again, with its bars unchanged.

## World

The three continents of stage 12a, with the same crossing, pace, displacing and lineage tags, plus `catch_energy` 50 and `pred_energy_cap` 170. Worlds `bulk_feast_meadow`, `bulk_feast_forest`, `bulk_feast_tundra`. Sets I (seeds 46, 47, 48) and J (seeds 49, 50, 51).

## Success test (can fail)

Stage 12a's bars, unchanged. Means over the last 10% of training, on both sets:

1. **Size follows climate:** in grazers or hunters, size differs by at least 0.15 between highest and lowest continents, with the same order on both sets.
2. **Worlds still last:** `scripts.persist --evolved --steps 2000 --episodes 32`, both species alive at step 1,999 in at least 0.5 of worlds, on meadow and tundra.
3. **Hunters still hunt:** a fixed chaser catches at most 1.0x what learned hunters do, on meadow and tundra.

Prediction: bars 2 and 3 pass, since the small map does. Bar 1 is the risk. In stage 12f both species ran to the top of the size range, which leaves little room for a difference, and meadow and tundra may swap places, which fails "same order" as it did in stage 12a. Forest's hunters shrank to about 0.05 in stage 12a; if they do again, the gap is easily met and the order between the other two decides it.

Reported, not bars: grazer sight's order, forest's hunters, hunter numbers against stage 12a's, and migrants' shares.
