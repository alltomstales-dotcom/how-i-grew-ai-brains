# Stage 15a: can a gene that switches on in winter do what a constant one can't?

Date: 2026-11-01. Written before any of these runs. From the plan in `2026-11-01-memory-and-inheritance-review.md`.

## Why

Every gene here is expressed from birth to death, all year. In nature many aren't: Daphnia grow helmets only when they sense a predator. Stage 13e found tundra's hunters alone never reach camouflage in 600 updates (0.086, 0.140), possibly because they pay for it all year on winter's thin budget. So what if the gene is only on in winter?

## The control this needs

Expressed half the time, the gene also costs half as much on average. So if it evolves, that may just be because it's cheaper. The fair comparison is against a constant gene at half the price, with the same average cost.

## World

Isolated tundra, 256 x 256, with no crossing, as stage 13e, on its seeds (54 and 57), 600 updates, founders' gene at 0.1:

- `camo_winter_tundra`: the gene works only in winter, at a price of 0.10 per step at gene 1, paid only in winter. In summer a hunter shows from 4 cells like any other.
- `camo_half_tundra`: the gene works all year, at a price of 0.05, the same average cost.

Stage 13e's constant gene at 0.10 is the reference: it never reached 0.5.

## Success test (can fail)

Over the last 60 updates, on both seeds:

1. **It evolves when it's conditional:** `camo_winter_tundra` hunters' gene at least 0.5.
2. **Timing is what matters:** `camo_winter_tundra` at least 0.15 above `camo_half_tundra`. If the half-price gene gets there too, the reason is price, not timing.
3. **It carries the world:** `camo_winter_tundra` has both species alive at step 1,999 in at least 0.5 of worlds.

Prediction: bar 1 passes, since hiding pays most when grazers are scarce. I don't expect bar 2 to pass by much, and I think price explains a good part of it. Bar 3 is doubtful: 0.000 alone in stage 13e. Reported, not bars: when the gene first reaches 0.5, hunters by season, and grazer sight.

## Result (2026-11-01): FAIL on bars 2 and 3, and it's price, not timing

All four trainers reached update 600. Means over the last 60 updates, on sets W and X:

| isolated tundra | hunters' gene | first reaches 0.5 | both alive at step 1,999 | hunters |
|---|---|---|---|---|
| `camo_winter_tundra`, winter only, 0.10 | 0.902, 0.909 | 128, 213 | **0.188, 0.031** | 222, 198 |
| `camo_half_tundra`, all year, 0.05 | 0.898, 0.912 | 165, 51 | **0.750, 1.000** | 299, 538 |
| stage 13e, all year, 0.10 | 0.086, 0.140 | never | 0.000, 0.000 | 98, 110 |

1. **It evolves when it's conditional: met.**
2. **Timing is what matters: missed.** The two are level, within 0.004. At the same average cost, the constant gene gets there about as fast.
3. **It carries the world: missed,** at 0.188 and 0.031.

My prediction held on bars 1 and 3, and on price explaining a good part of it, though I didn't expect a tie.

**So tundra can evolve camouflage alone, and price was the barrier.** That qualifies stage 13e: "tundra never gets there alone" holds at a price of 0.10, and at 0.05 it gets there on both sets. What stage 13e showed about crossing still stands at 0.10.

**Switching on only in winter is worse here.** Worlds where the gene works all year at half the price last far better than worlds where it works half the time at the full price, with more hunters. I'd guess that's because the winter gene leaves hunters visible all summer, and it's only a guess. This one trait in one world doesn't speak to conditional expression in general: a trait that does harm out of season, which camouflage here doesn't, might show the opposite.

