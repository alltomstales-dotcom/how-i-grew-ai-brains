# Stage 11a: do different climates make different animals, while migrants keep arriving?

Date: 2026-10-31. Written before any of these runs. Tom's idea: three connected worlds on two machines, with different ecology and climate driving evolution and learning.

## Why

Stage 10f finally gave trained brains a world that lasts, with camouflaged hunters. All earlier worlds were single maps restarted from a pool. This is the first where populations have a place: separate continents, each with its own conditions, its own brains and its own gene pool, joined only by animals walking across.

## World

Three continents of 256 x 256, in a ring: meadow, then forest, then tundra, then meadow. Each uses camouflage, plants at `seed_rate` 0.2, meals of 25, and the density of the 48 x 48 world (at most 2,730 grazers and 910 hunters).

- **Meadow:** open and rich.
- **Forest:** stage 8b's thickets and walking costs.
- **Tundra:** half of every 120 steps is winter, when plants feed a quarter as much.

Each continent is its own trainer: meadow and tundra on a DGX Spark, forest on another. Each runs 8 parallel copies of its continent, with copy-level migration random. After each training update (32 steps), an animal within 2 cells of the east edge crosses with chance 0.2, arriving at the west edge of the next continent. It carries genes, energy and lineage, not its brain, and uses the brains of where it lands. If a world is full, the migrant is lost. Files move in the background, and nothing waits on them.

Two sets: A with seeds 10, 11, 12, and B with seeds 13, 14, 15. A 10-update timing run on the three continents together took about 3.5 s per update and 6.94 GB of device memory each, with nothing needing to be paused. So each trainer runs 600 updates, about 35 minutes, which is about 560M animal-steps of slots against 470M in a 150-update run of the 48 x 48 world. This was fixed before any full run.

## Success test (can fail)

All on both sets:

1. **All three last:** `scripts.persist --evolved --steps 2000 --episodes 32` on each continent, no crossing during this test, both species alive at step 1,999 in at least 0.5 of worlds.
2. **Migrants matter:** at the end of training, at least 0.05 of living animals on each continent descend from arrivals (mean of the last 10% of updates in `transit.jsonl`, grazers and hunters each).
3. **Climate shapes bodies:** for at least one gene among speed, sight and breed, in grazers or hunters, the highest and lowest continent means differ by at least 0.15, with the same ordering of continents on both sets. The repeat is the control against chance.

Prediction: bar 3 is most likely to fail, since migration mixes genes and all three share the same costs. Forest may simply die out, since stage 10d's rules failed without camouflage.

Reported, not bars: hunter and grazer numbers against the caps, whether brains differ (a continent's brains swapped into another's world), how fast arrivals' lineages spread, and whether any continent dies and is reseeded only from itself.

## Result (2026-10-31): FAIL by the rule, on bars 1 and 2, and climate clearly shaped bodies

Bar 3 passed, and it was the one I predicted would fail. Means over the last 10% of training, sets A and B:

| gene | meadow | forest | tundra | range | same order? |
|---|---|---|---|---|---|
| grazer sight | 0.759, 0.682 | 0.325, 0.323 | 0.118, 0.097 | 0.641, 0.585 | yes |
| hunter speed | 0.291, 0.311 | 0.190, 0.213 | 0.524, 0.514 | 0.334, 0.301 | yes |
| grazer speed | 0.294, 0.248 | 0.228, 0.174 | 0.133, 0.128 | 0.161, 0.120 | yes, under 0.15 on B |

| bar | set A | set B | |
|---|---|---|---|
| 1. both species alive at step 1,999, 32 seeds | meadow 1.000, forest 0.000, tundra 1.000 | meadow 1.000, forest 0.000, tundra 1.000 | **missed** |
| 2. living animals descended from arrivals, grazers and hunters | meadow 0.006, 0.005; forest 0.001, 0.018; tundra 0.000, 0.000 | meadow 0.013, 0.004; forest 0.003, 0.048; tundra 0.000, 0.000 | **missed** |
| 3. a gene differs by at least 0.15, same order | grazer sight, hunter speed | grazer sight, hunter speed | **met** |

**Forest:** hunters die out with no restarts, 0.062 of worlds at step 600 and none by step 1,200 on set A, though they beat a fixed chaser (0.71x and 0.77x). They hold about 75 per world in training, where worlds restart. So stage 10f's camouflage doesn't save hunters from walking costs and thickets.

**Tundra:** hunters stay far below their cap, and grazers swing, with the 200-step samples ranging from about 510 to 2,730. See the fine curve below, because the 200-step samples alias with the 120-step year.

**Meadow:** both species sit on their caps, 2,730 and 910, as in stage 10f, so no limit other than slots shows here.

**Why bar 2 says little:**
- Gene flow was close to nothing, so bar 3 hardly tests climate against migration. These continents were close to isolated.
- About 60 to 80% of grazer migrants were lost on arrival, since a full world has no slot: meadow received about 7,000 of about 20,000 sent.
- **A design flaw:** the second machine ran faster, so forest finished its 600 updates in 19 minutes against 36 to 41 on the Spark. For about the second half of the Spark's runs, nothing arrived at tundra, which only gets arrivals from forest, and forest got no more. This likely explains tundra's 0.000. Crossing should be tied to shared time, or the trainers paced.

**The repeat is the evidence for bar 3:** two sets of fresh seeds gave the same order, and the same sizes within about 0.08. I can't yet say why grazers see far in meadow and barely at all in tundra, or why tundra's hunters are fast. The genes are measured, and the reasons aren't.

Each continent evolved in isolation with its own brains, and nothing here says the differences would survive strong migration. Nothing here is a claim about real climates or animals.

## Tundra's cycle, at a fine grain

`scripts.persist --evolved --steps 2000 --episodes 32 --every 10` on `pp-tundra-A`, averaged over 32 worlds that share a calendar, so a seasonal swing is not averaged away. One year, from step 980:

| step | 980 | 1000 | 1020 | 1040 | 1060 | 1080 | 1100 | 1120 |
|---|---|---|---|---|---|---|---|---|
| grazers | 2730 | 2730 | 2730 | 2727 | 1507 | 659 | 2730 | 2730 |
| grazer energy | 19.3 | 30.0 | 30.2 | 13.0 | 6.0 | 5.7 | 19.2 | 29.8 |
| hunters | 296 | 325 | 403 | 518 | 625 | 461 | 304 | 338 |

Each 120-step year, grazers fill their cap and fatten in summer, run out of energy in winter, and crash to about a quarter. They rebound by the next sample. Hunters lag behind: they keep rising as grazers fall, peak near 630 and drop to about 300 as grazers recover. Grazers still hit their cap in the good part of the year, and hunters never reach theirs (910). It's only measured on this set at this grain. This is the first swing in numbers from trained brains in this project, and it's driven by the seasons rather than arising on its own.

## Correction (2026-10-31): the migrant measure was flawed, and "nearly isolated" is withdrawn

Every world in training restarts every 500 steps, and a restart wiped the lineage tags, while migrants' genes carried on through the gene pool. So `immigrant_share` only ever counted descendants from the last few hundred steps, and bar 2 could not measure what it claimed. Stage 11b's runs show the sawtooth: on `pp-meadow-C` grazers it climbs 0.005, 0.012, 0.020, 0.033, 0.030 across the five hundred-step bins of an episode, and within episodes it peaks at 0.107, 0.082, 0.128 and 0.143 on meadow and tundra.

So the statements above that migrants left almost no descendants, and that the continents were close to isolated, are not supported. How much gene flow there was is unmeasured. A rough sum: about 3% a restart, compounding over 38 restarts, could be large. The climate result in bar 3 stands as a measurement of genes, but it no longer comes with "these worlds were isolated" attached, which if anything makes it more interesting. Stage 11c measures this properly.

