# Stage 12a: does a body part that matters follow climate too?

Date: 2026-10-31. Written before any of these runs. Tom asked for more complexity in the organism.

## Why

The bodies drawn in the viewer are looks only, except for what speed and sight genes already do. Stage 11c showed climate keeps grazer sight apart even under heavy migration. So does a new body trait that really matters follow climate too, and does it change what evolves around it? Size is the cheapest real trait to switch on, since the rules exist from stage 5a: a bigger hunter catches more surely, a bigger grazer is harder to catch and digests tough plants better, a bigger meal feeds more, and size costs energy every step.

## World

The three continents of stage 11c, with the same crossing, pace, displacing and lineage tags, plus `sizes` on, with a catch succeeding with chance sigmoid((hunter size - grazer size) / 0.25). Worlds `bulk_meadow`, `bulk_forest`, `bulk_tundra`. Sets G (seeds 40, 41, 42) and H (seeds 43, 44, 45), 600 updates each.

Stage 5a's first version with this rule had a loophole: grazers grew big and hunters shrank. That was without camouflage or lasting plants, so it may or may not return.

## Success test (can fail)

Means over the last 10% of training, on both sets:

1. **Size follows climate:** in grazers or hunters, size differs by at least 0.15 between highest and lowest continents, with the same order on both sets.
2. **Worlds still last:** `scripts.persist --evolved --steps 2000 --episodes 32`, both species alive at step 1,999 in at least 0.5 of worlds, on meadow and tundra.
3. **Hunters still hunt:** a fixed chaser catches at most 1.0x what learned hunters do, on meadow and tundra.

Prediction: bar 1 passes, and tundra's grazers come out biggest, since size is a store against winter here only by digestion and catch chance, which I hold loosely. If grazers outgrow hunters everywhere, the loophole is back and bars 2 and 3 likely fail.

Reported, not bars: whether grazer sight keeps its order on all six earlier sets' pattern, any split in size (`scripts.species`, its original use), and migrants' shares.

## Result (2026-11-01): FAIL on bars 1 and 2

Means over the last 60 updates, with hunters' numbers in training:

| | meadow | forest | tundra | bar |
|---|---|---|---|---|
| both alive at step 1,999, G, H | 0.000, 0.000 | 0.000, 0.000 | 0.000, 0.000 | meadow and tundra at least 0.5: **missed** |
| grazer size, G, H | 0.291, 0.636 | 0.200, 0.265 | 0.412, 0.310 | same order: **missed** |
| hunter size, G, H | 0.801, 0.907 | 0.051, 0.055 | 0.805, 0.731 | same order: **missed**, by the meadow and tundra pair |
| chaser against learned hunters, G, H | 0.64x, 0.80x | 12.68x, 8.34x | 0.72x, 0.66x | meadow and tundra at most 1.0: met |
| hunters per world in training, G, H | 74, 80 | 179, 174 | 89, 91 | reported (about 865 and 680 without size) |
| catches per step, set H | 2.5 | 0.25 | 3.1 | reported (about 45 and 40) |

Tom spotted this on the phone viewer while it was running: hunters collapse by about update 76. Switching on size undoes what camouflage gave: meadow and tundra go from lasting in every world to none.

It is not stage 5a's loophole. Hunters grow big rather than small in meadow and tundra, at 0.73 to 0.91 against grazers' 0.29 to 0.64, and still die out. In forest they shrink to about 0.05 and all but stop hunting. One steady thing is that forest's hunters are by far the smallest on both sets. My guess was that tundra's grazers would be biggest, and that held only on G.

Grazer sight kept its order on both sets, and more narrowly: 0.292 / 0.251 / 0.114 and 0.304 / 0.272 / 0.117. Meadow's fell from about 0.7. So one more trait changed what evolves around it.

Why isn't settled. Size costs up to 0.5 energy per step, a catch is no longer certain, and a meal is worth 25 x (0.5 + grazer size). Stage 12b separates those.

