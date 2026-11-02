# Stage 15c: does it matter where and when mutation happens?

Date: 2026-11-01. Written before any of these runs. Tom asked for environmental mutation in population genomes. From the plan in `2026-11-01-memory-and-inheritance-review.md`.

## Why

Today every child mutates by the same amount (`gene_sigma` 0.05), wherever and whenever it's born. In nature, whether stress raises mutation is contested, but the question here doesn't depend on that: does it matter where and when mutation happens, as distinct from how much of it there is? Stage 13e gives a clear target, since tundra alone never reaches camouflage in 600 updates at a price of 0.10.

## The control this needs

More mutation of any kind might help. So every environmental arm is compared with a control that has the same average, spread evenly.

**I'm not testing a parent's hunger,** as I'd first planned. Animals only breed when well fed, so a parent at birth is never hungry, and the test wouldn't mean anything here.

## World

Isolated tundra, 256 x 256, no crossing, camouflage gene at 0.10, as stage 13e, on its seeds (54 and 57), 600 updates. Sets Y and Z.

- `mut_winter_tundra`: children mutate at 3x in winter and 1x in summer, an average of 2x.
- `mut_zone_tundra`: children born in the map's mutagen zones, every fourth column of cells, a quarter of the map, mutate at 5x, and elsewhere at 1x, an average of about 2x if births are even across the map.
- `mut_even_tundra`: children mutate at 2x everywhere, all year. The control.

Stage 13e's 1x is the reference.

## Success test (can fail)

Hunters' camo gene over the last 60 updates, on both sets:

1. **More mutation helps at all:** `mut_even_tundra` at least 0.5. Reported as its own question.
2. **Environmental mutation gets there:** `mut_winter_tundra` or `mut_zone_tundra` at least 0.5.
3. **Where and when matters:** that same arm at least 0.15 above `mut_even_tundra`.

The stage passes on bars 2 and 3.

Prediction: bars 1 and 2 could go either way. Bar 3 fails, since I expect the amount to matter and not the pattern. Reported, not bars: both species alive at step 1,999, when the gene first reaches 0.5, and how far other genes drift, since this changes every gene and may do damage.

## Result (2026-11-01): FAIL on bar 3, and doubling mutation changes everything

All six trainers reached update 600. Hunters' gene over the last 60 updates, on sets Y and Z:

| isolated tundra, price 0.10 | hunters' gene | first reaches 0.5 | both alive at step 1,999 | hunters |
|---|---|---|---|---|
| normal mutation, stage 13e | 0.086, 0.140 | never | 0.000, 0.000 | 98, 110 |
| `mut_even_tundra`, 2x | 0.904, 0.907 | 30, 21 | 0.219, 0.000 | 245, 230 |
| `mut_winter_tundra`, 3x in winter | 0.864, 0.865 | 35, 31 | 0.906, 0.031 | 331, 294 |
| `mut_zone_tundra`, 5x in a quarter of the map | 0.882, 0.894 | 26, 32 | 0.969, 0.656 | 297, 313 |

1. **More mutation helps at all: met.**
2. **Environmental mutation gets there: met.**
3. **Where and when matters: missed.** Both patterns end a little below even mutation, by 0.01 to 0.04. The gene doesn't care about the pattern.

My prediction held on bar 3. I said bars 1 and 2 could go either way, and the size of the effect is much larger than I expected.

**So tundra was short of variation, not incapable.** At normal mutation it never got there in 600 updates. At twice that, it takes about 30. Stage 15a found the same barrier from another side, since half the price works too. So stage 13e's "tundra never gets there alone" holds only at this price and this mutation rate. It looks as though a mutant has to reach far enough up the gene before hiding pays, and more mutation or a lower price makes that likelier. That reading isn't tested, and it may bear on stage 13a's split, where two runs of four got there, and on set K's wait of nearly 500 updates.

**The surprise is in a number that wasn't a bar.** At the same average mutation, worlds last far better with zones than with even mutation: 0.969 and 0.656 against 0.219 and 0.000, with more hunters. My guess is that most animals in the zone arm are born outside the zones, so most of the population carries less mutation damage, while the zones supply variation. Winter's result is mixed, at 0.906 and 0.031. These are two sets, and I wasn't testing this, so it could be chance. Stage 15d tests it on fresh seeds.

**A check on averages.** Zones average about 2x only if births are even across the map, and I didn't measure that, so the comparison may not be exactly at equal average. Stage 15d reports it.

