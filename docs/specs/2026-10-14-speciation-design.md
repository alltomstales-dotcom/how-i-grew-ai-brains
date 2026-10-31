# Stage 5a: two ways to be a prey (sex, size and speciation)

Date: 2026-10-14. The first slice of Stage 5 (a richer ecology). Builds on the evo, terrain and
seasons worlds. Written as the record for the build loop the user asked for, not as a gate.

## Why

Stage 5's test is that at least two stable lineages split out of one founder pool. Two things stand
in the way today:

- **Asexual breeding.** Every animal copies its own genes, so any spread in a gene is just drift
  around one optimum. A "lineage split" cannot be told apart from noise.
- **One niche.** Every prey eats the same plants in the same way. Nothing rewards two different
  answers over one middle one.

This slice adds both: mates and crossover, and a size gene that trades two niches against each
other. The food web (mid-predators, scavengers) and larger maps are Stage 5b.

## World

**Size gene.** A fourth animal gene, `size` in [0, 1], for prey and predators. Founders start at
0.5, the middle, so any split has to evolve.

- *Tough plants.* A grazer gains `grass_energy * (1 - 0.5 * tough * (1 - size))`: big animals
  digest tough plants fully, small ones get half.
- *Burrows.* Only animals with `size < burrow_size` (0.5) fit into a burrow.
- *Catches.* A predator can only catch prey no bigger than `size + catch_margin` (0.25) of its own.
- *Upkeep.* `size_cost * size` (0.5) energy per step. Being big is not free.

So a small prey hides in burrows and lives on soft plants, while a big prey eats anything and is
too large for small predators. A prey in the middle is too big to hide and too small to be safe.
These are the only rules; nothing names the two niches.

**Sexual breeding** (`mating = True`, off by default so older worlds are unchanged). An animal
ready to breed needs a living mate of its own species within `mate_radius` (2) cells, wrapped
Chebyshev. With none it waits. The child takes each gene from one parent or the other at random
(uniform crossover), then mutates. Only the parent that is ready pays the cost. The mate is the
nearest candidate; mate choice is not learned.

**Observations.** Own-state channels gain size: 11 channels. `load_brains` already zero-pads extra
input channels, so every saved brain loads unchanged. Gene snapshots saved with three animal genes
load with size 0.5.

A registered world, `niches`, is the terrain world (walls and burrows) with size, sexual breeding,
and plant toughness evolving as before.

### v2, after pp-niche-001 (2026-10-14)

pp-niche-001 and its control, 30 minutes each, both found the same loophole. Prey grew big (0.81
and 0.88), predators shrank (0.12 and 0.06), and catches fell to zero. Under the hard cut-off a big
prey was simply immune, and a predator gained nothing by growing to reach it. Predators survived
without eating only because a world in which they die out resets with fresh founders. Three rule
changes, each a flag, so the first rules stay reproducible:

- **No immunity** (`catch_scale = 0.1`). A catch succeeds with chance
  `sigmoid((predator size - prey size) / 0.1)`: 50% at equal size, 27% for a prey 0.1 bigger and
  under 1% for one 0.5 bigger. `catch_scale = 0` keeps the old cut-off.
- **Big prey, big meal.** With sizes on, a catch is worth `catch_energy * (0.5 + prey size)`.
- **Size-matched mating** (`mate_tolerance = 0.2`). A mate must be within 0.2 in size. Resets
  scatter animals across the map, so geography can never isolate two groups, and this is the
  only rule that can. Many real animals choose mates of similar size, and it is the textbook
  route to one species splitting in two within a single place.

The success test is unchanged. The control keeps the v2 rules with `mating = false`.

### v3, after pp-niche-002 (2026-10-14)

v2 failed the same way as v1. Prey reached 0.86–0.89, predators 0.03–0.06, and catches fell to
zero. Every pressure pushed prey toward big:

- Plants evolve to toughness 0.99, and big grazers digest tough plants best.
- Big prey are hard to catch.
- The small-prey niche relied on burrows, which learned prey never use.

That is purely directional selection, which cannot split a species. v3 builds the textbook route
instead, competition for two foods:

- **Two fixed plant kinds** (`plant_kinds`, `plant_sigma = 0`). Founder plants are half soft
  grass (toughness 0) and half tough shrubs (toughness 1). Plant genes no longer mutate, so both
  kinds persist. Grass is eaten whole and must reseed. Shrubs survive bites and regrow, but feed
  small grazers only half.
- **Size only affects feeding** (`size_catch = False`). Predation ignores size, and meals are
  flat again.

Size still sets digestion, upkeep and burrow fit. A population of small grazers leaves the shrubs
unused, so a big mutant gains food that nobody else is eating. A population of big grazers pays
upkeep that small grass-eaters avoid. If competition for food drives a split, size-matched mating
(kept from v2) can hold the two groups apart.

### v4, after pp-niche-003 (2026-10-14)

v3 failed as well (pp-niche-003 and pp-niche-ctl3, 30 minutes each). Prey ended big again (0.90
and 0.89), and predators small (0.03 and 0.06). In pp-niche-003 prey D reached 2.0 with a
minority of 0.38, but the size gap was 0.13, well under 0.3: one wide group, not two. The
predator D readings above 2 came from a minority of 2–5% piled at size 0. The control looked
the same, so mating made no difference. Mating did slow predators: 16 generations against 78
without it, because a mate must be within 2 cells and predators are few.

The cause was in the v3 feeding rule. Grass fed every size fully, so a big grazer was never worse
at feeding than a small one. Only upkeep favoured small bodies, and that was too weak. v4 adds a
real trade-off (`size_tradeoff`, on in `niches`). A grazer's shortfall is
`tough * (1 - size) + (1 - tough) * size`, so:

| size | on grass | on shrubs |
|---|---|---|
| 0 (small) | full | half |
| 0.5 | 3/4 | 3/4 |
| 1 (big) | half | full |

No size is best at both, and middle sizes are second best at each. When most grazers are big,
they leave grass uneaten, and small mutants gain. The pressure depends on what everyone else is
doing, which is the condition under which one species can split in two.

Test: `pp-niche-004` (v4) and `pp-niche-ctl4` (`mating = false`), 30 minutes each. 15 minutes
gives too few generations for a split.

### v5, after pp-niche-004 (2026-10-14)

v4 went the same way as v3: prey grew to 0.80 with mating and 0.89 without. The plant metrics
show why, for v3 and v4 alike. Plant toughness reached 1.0 everywhere within about 20 updates, so
**the grass had died out** and there were never two foods. Grass was killed by every bite, and
the gaps were refilled by seeding and germination from whatever survived, which was shrubs. In a
small test world with few grazers the mix collapsed the other way, to all grass. Either way, it
does not hold.

v5 (`kinds_persist`, on in `niches`) keeps both kinds:

- A bite never kills a plant. Grass, like shrubs, is bitten and then regrows.
- Seeding ignores toughness, so neither kind spreads faster.
- New plants that sprout on bare ground are fresh founders, half of each kind, not copies of
  the survivors.

A test checks that under grazing the mix stays between 25% and 75% shrubs for 400 steps.

Lesson: check that the conditions a test relies on actually hold during the run. Two-food
competition was the whole premise of v3 and v4, and the logged plant toughness showed from the
first minutes that it was missing.

### v5 result (2026-10-14): FAIL, with the first real divergence

Both foods survived: plant toughness stayed bimodal (SD 0.44–0.47), though the mix drifted to
about 70% shrubs by the end, as the code review predicted (bitten plants cannot seed, so the
kind that is grazed more spreads less).

| run | prey size at end | size SD | last 3 snapshots: D / minority / gap |
|---|---|---|---|
| pp-niche-005 (mating) | 0.14 | 0.10 | 2.09/0.25/0.16, 1.95/0.30/0.14, 1.85/0.35/0.14 |
| pp-niche-ctl5 (no mating) | 0.23 | 0.22 | 1.78/0.48/0.25, 1.79/0.44/0.25, 1.81/0.46/0.30 |

With mating, prey moved as one group toward small bodies. Without mating, two groups pulled apart
steadily (gap 0.22 at update 105, 0.30 at update 164, evenly split), still widening when the
30-minute cap stopped it, but with D under 2. Sexual mixing with a 0.2 size tolerance seems to
hold the population together, while asexual lineages are free to drift apart.

Next (v6): the control rerun for 60 minutes on a new seed, to see whether the gap keeps growing,
alongside a mating run with a stricter size tolerance (0.1), so hybrids between the groups are rarer.

### v6 result (2026-10-14): FAIL, a near miss without mating

| run | prey size at end | size SD | snapshots from update 240 to 322: D / minority / gap |
|---|---|---|---|
| pp-niche-ctl6 (no mating, 60 min, seed 1) | 0.26 | 0.26 | D 1.92–2.09, minority 0.35–0.48, gap 0.35–0.40 at every snapshot |
| pp-niche-006 (mates within 0.1 in size, 60 min) | 0.09 | 0.08 | D 1.44–1.90, gap 0.11–0.13 |

Without mating, the v5 divergence repeated on a new seed and then held. Two size groups, evenly
shared, stayed 0.35–0.40 apart for the last 80 updates (about 25 minutes), and size was the gene
that most separated them. The test fails only because one of the last three snapshots (update
300) had D 1.95 against the bar of 2.0. The gap stopped growing around update 240, so this looks
like a stable state rather than an unfinished one.

With mating, prey shrank together, more so with the stricter tolerance. In this world two-parent
breeding blocks the split that asexual lineages find.

**Where Stage 5a stands:** the bar is not met, and it is not moved after the fact. The finding:
competition for two foods holds two body sizes apart (a stable polymorphism) when lineages breed
alone, and two-parent breeding prevents it, even when mates must match in size. Whether a
learned or evolved mate choice can protect the split is left for later.

## Measuring a split

`scripts/species.py --run-dir R` reads the run's gene snapshots (`genes_*.npz`). For each snapshot
and each species, it fits a two-Gaussian mixture to the size gene and reports:

- **separation**: Ashman's D, `D = sqrt(2) |m1 - m2| / sqrt(s1^2 + s2^2)`. D > 2 is the usual bar
  for two clean modes.
- **minority share**: the smaller group's share of animals.
- **size gap**: the distance between the two group means.

It also reports which gene differs most between two 2-means clusters on all four genes.

*Amended after a probe (2026-10-14).* The first version (2-means on all genes, then D along the
axis between the centres) scored D > 2 on a 4-generation run where no split was possible. Resets
copy genomes from the pool, so a snapshot holds clumps of clones, and a curve fitted to one clump
is needle-thin. Genes are also clipped to [0, 1], so a shrinking species piles up at 0 and looks
like a second group. Now each curve's spread is floored at one mutation step (0.05), and the size
gap must reach 0.3. The probe's chance gap was 0.24, so 0.3 sits above it and above the 0.25 catch
margin. D alone is not trusted.

## Success test (can fail)

`pp-niche-001` on `niches`, learning on, evolution on. Run for 30 minutes, extended if the
separation is still climbing.

- **Pass:** in each of the last three snapshots, some species has D >= 2 on the size gene, a
  minority share of at least 0.2, and a size gap of at least 0.3.
- **Control:** the same world with `mating = False`. If it splits as well, sexual breeding is not
  what made the difference; say so.
- **Behaviour to look for in clips:** small prey running for burrows when a predator is near, and
  big prey standing their ground or grazing tough plants.

## Out of scope

Learned mate choice, kin recognition and signalling (Stage 6). New species and the food web, and
maps above 48 cells (Stage 5b).
