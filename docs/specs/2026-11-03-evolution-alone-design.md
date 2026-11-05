# Stage 17: can evolution alone build a hunter, and how small can its genome be?

Date: 2026-11-03. Written before any of these runs.

## Why

Every hunter so far started from brains trained by gradient descent (the shared trainer), and nothing since has beaten those founders. Evolution has only polished what training built. Two questions are left that the series hasn't asked:

1. With no training at all, only mutation and selection, can evolution build a hunter from random brains?
2. A real genome is far smaller than the brain it builds. If mutation can only move a brain within a small random subspace (16 or 128 directions instead of 4,702 weights), does evolution do worse, or better? Searching fewer directions can be easier.

## World and arms

`tiny_stealth`, 4,702 weights a brain, 64 worlds, 1,000 updates (about 350 generations), seeds 0 and 1. **No founders:** every animal starts from a random brain. No learning in life (`darwin`). A child gets its parent's birth brain plus mutation.

- **full:** mutation on all 4,702 weights.
- **genome 128:** mutation only along 128 fixed random directions in weight space, the same 128 for every animal. Per-weight mutation size is the same as in full.
- **genome 16:** the same with 16 directions.

The same random directions are used in every run. Grazers evolve the same way in each arm.

## Settings fixed by a pilot

Mutation size (sigma) is set by a short pilot on seed 2, before the bars are applied, so that worlds don't die out at once and mutation is big enough to matter. Its value goes in an addendum here before the full runs.

## Test

`scripts.cross_indiv`, hunters frozen, children copy the brain. Each arm's hunters face **stage 16's genes-arm grazers** (`pp-cul-genes-*`, inherited trained brains, same seed), 64 worlds of 500 steps. Random hunters catch .0134 and .0132 against those grazers, and trained genes hunters .0365 and .0344.

## Success test (can fail)

On both seeds, catches per hunter-step:

1. **Evolution alone builds something:** full catches at least 1.5x random (about .020).
2. **A small genome is enough:** genome 128 catches at least as much as full (1.0x).
3. **A tiny genome is enough:** genome 16 catches at least as much as full.

Reported, not bars: each arm against its own grazers, both species alive at step 1,999, catches over training, and how far the evolved brains are from trained ones.

## Prediction

Bar 1 fails. In 350 generations, random mutation over 4,702 weights finds little, so I expect full to reach about 1.2x random. Bar 2 passes, because searching 128 directions concentrates selection's effort. Bar 3 is my least confident call; I lean pass, but 16 directions may simply not contain a hunter. Nothing here beats trained brains.

## Addendum, pilot (2026-11-03, before any full run)

Seed 2, 150 updates, random brains, catches per step in each arm's own world (against grazers evolving alongside, so this says nothing about skill against trained grazers), first 15 updates → last 15:

| arm | sigma | hunters | catches |
|---|---|---|---|
| full | 0.003 | 8.4 → 15.0 | .171 → .375 |
| full | 0.01 | 8.1 → 15.6 | .173 → .385 |
| full | 0.03 | 8.4 → 29.6 | .182 → .951 |
| genome 128 | 0.01 | 8.2 → 15.4 | .174 → .381 |
| genome 128 | 0.03 | 7.9 → 28.6 | .162 → .899 |
| genome 16 | 0.03 | 7.9 → 29.5 | .164 → .967 |

**Fixed for the full runs: sigma 0.03 for every arm.** That's where worlds fill up and mutation visibly matters. At 0.003 and 0.01 the three arms look the same and nothing much happens. Already striking, and not a bar: at sigma 0.03, 16 directions do as well in their own world as all 4,702 weights. Whether any of it is skill shows only against trained grazers.

## Result (2026-11-03): FAIL on all three bars. Evolution alone built hunters that thrive at home and are nearly useless against trained prey

Against stage 16's genes-arm grazers (inherited trained brains), 64 worlds of 500 steps, catches per hunter-step, seeds 0 and 1 (random .0134, .0132; trained genes hunters .0365, .0344):

| hunters | catches | against random | against full |
|---|---|---|---|
| full (4,702 weights) | .0159, .0163 | 1.19x, 1.23x | |
| genome 128 | .0153, .0157 | 1.14x, 1.19x | 0.96x, 0.96x |
| genome 16 | .0169, .0119 | 1.26x, 0.90x | 1.06x, 0.73x |

1. **Evolution alone builds something: fail.** Full reached 1.19x and 1.23x random (bar 1.5x).
2. **A small genome is enough: fail, narrowly.** Genome 128 got 0.96x and 0.96x of full (bar 1.0x).
3. **A tiny genome is enough: fail.** Genome 16 got 1.06x and 0.73x, so the seeds disagree.

**In their own worlds every arm looks like a success:** catches .0344 to .0363 per hunter-step, about 31 hunters a world, and both species alive at step 1,999 in 0.97 to 1.00 of worlds. That's about level with trained genes hunters against their own grazers (.0365, .0344). In training, catches rose from about .17 to about 1.0 a step. **All of that was hunters and grazers co-evolving into an easy world.** Against grazers that inherited trained brains, these hunters die out within the test (0.0 to 0.6 left per world).

**My prediction:** bar 1 right, and I even guessed about 1.2x random. Bar 2 wrong: I expected 128 directions to beat full, and they did slightly worse. Bar 3 split by seed, where I'd leaned pass.

**What it adds to the series:** this is the fourth time an own-world number has hidden that nothing was learned that works against a competent opponent (14d, 16, 16b, and now 17). Gradient training built the founders; in about 350 generations, mutation and selection alone, at any genome size, didn't come close. Compressing the genome to 16 numbers didn't help, so with selection this weak the search wasn't limited by the number of directions.

**Limits:** one mutation size from the pilot (0.03), 350 generations, a 4,702-weight brain, and two seeds. Real evolution had far longer.

## Addendum, correction (2026-11-04, from the stage 20 reviews)

"With selection this weak the search wasn't limited by the number of directions" doesn't follow. Every arm sat at about 1.2x random against trained grazers, and the own-world fitness that drove selection rewarded catching easy co-evolved grazers. That is a floor: the test had no room to show a difference between genome sizes, so the equal results say nothing about whether 16 or 128 directions limit the search. Also, equal mutation size per weight isn't equal change in behaviour, and one fixed random subspace may simply exclude useful brains (Fable's and Astra's reviews, `docs/reviews/2026-11-04-next-stage-*.md`). The bar results stand; that sentence of the interpretation is withdrawn. The result above is left as written.
