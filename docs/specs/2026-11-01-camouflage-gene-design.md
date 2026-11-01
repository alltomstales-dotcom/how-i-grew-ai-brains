# Stage 13a: can camouflage evolve?

Date: 2026-11-01. Written before any of these runs.

## Why

Stage 10f gave every hunter camouflage by decree, so a grazer sees one only from the next cell, and that was the first thing that made a trained world last (0.957, 0.961, against 0.098). A rule I wrote in by hand doesn't show that hiding is something evolution would find, or pay for. Tom also asked for more complexity in the organism. So camouflage becomes a gene, with a price, starting from hunters that hide nothing.

## World

A seventh gene, `camo`. A grazer sees a hunter only within that hunter's own radius, 4 - floor(3 x camo + 0.5) cells, and within the grazer's own sight as before. So a camo gene near 0 hides nothing, and above about 0.83 a hunter shows only from the next cell, as in stage 10f. Hunters pay `camo_cost` x camo energy each step. Grazers carry the gene with no effect and no cost. Founders start at 0.1, which hides nothing.

All are 48 x 48 on `lasting`, with no size, no fixed stealth and meals of 25, with no crossing:

- `camo`: the gene works, at a cost of 0.25 per step at gene 1.
- `camo_free`: the gene works, at no cost.
- `camo_neutral`: the gene does nothing and costs nothing. The control for drift.

Seeds 0 and 1, 150 updates, with one seed on each machine.

## Success test (can fail)

Over the last 15 updates, on both seeds:

1. **The control holds:** `camo_neutral` hunters' gene at most 0.4. If drift gets it there, the first bar below can't mean much.
2. **Camouflage evolves at a price:** `camo` hunters' gene at least 0.5, which is 2 cells or fewer.
3. **It carries the world:** `python -m scripts.persist --run-dir runs/NAME --evolved --steps 2000` on `camo`, both species alive at step 1,999 in at least 0.5 of worlds.
4. **Hunters still hunt:** on `camo`, a fixed chaser catches at most 1.0x what learned hunters do.

Prediction: bars 1 and 2 pass, since a hunter that hides should catch more whatever others do. Bar 3 is my doubt: the cost comes out of the same budget stages 12c to 12e showed is thin, and hunters must reach some camouflage before they die out. Reported, not bars: `camo_free` as the upper reference, grazer sight, hunter numbers.

## Result (2026-11-01): FAIL on bars 2 and 3, and the two seeds split

| | hunters' camo gene, seeds 0, 1 | both alive at step 1,999 | hunters | chaser against learned hunters |
|---|---|---|---|---|
| `camo_neutral` | 0.289, 0.307 | 0.000, 0.000 | 7.8, 7.5 | 0.66x, 0.58x |
| `camo_free` | 0.943, 0.945 | 0.746, 0.973 | 26.5, 29.0 | 0.68x, 0.70x |
| `camo`, price 0.25 | **0.079, 0.939** | **0.000, 0.172** | 7.5, 9.6 | 0.65x, 0.63x |

1. **The control holds: met.** Drift reaches about 0.3.
2. **Camouflage evolves at a price: missed.** Seed 1 gets there, and seed 0 never starts.
3. **It carries the world: missed.** Even with the gene at 0.94, seed 1 lasts in only 0.172.
4. **Hunters still hunt: met,** though on seed 0 that's a few hunters who hardly catch anything.

My prediction was half wrong: I expected hiding to pay whatever others did, and on one seed it never got going.

**Free camouflage evolves quickly, and it does about what my rule did.** From 0.1, it's at 0.94 by about update 30 to 60. Worlds last in 0.746 and 0.973, against stage 10f's 0.957 and 0.961 with camouflage set by hand. So hiding is something evolution finds by itself, if it's free.

**The split looks like a small fitness valley.** This is a reading from after the results, tested in stage 13b. The radius moves in steps, and the first benefit comes at a gene of about 0.17. Founders sit at 0.1, where more gene costs a little and buys nothing, so selection pushes down on a gene that has to go up. Seed 0 fell to 0.08. Seed 1 stayed near 0.1 to update 30, then jumped to 0.94 by update 60.

**The price looks too high to live on.** Seed 1's hunters reach 0.94 and pay about 0.23 per step, and they number 9.6 against 29.0 when it's free. It fits stages 12c to 12e, where these hunters live close to the edge.

