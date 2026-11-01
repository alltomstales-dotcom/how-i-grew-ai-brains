# Stage 14b: does evolution keep learning?

Date: 2026-11-01. Written before any of these runs.

## Why

Stage 14a: inheriting what was learned does harm, and learning in life barely helps. Either the learning rule is too weak, or what it passes on goes stale. Rather than guess, let evolution decide how much to learn: a heritable learning rate. If learning is doing harm, selection should turn it down. This is the second rung of what Tom asked for: not just inheriting what's known, but evolving how much an animal learns.

## World

As stage 14a, with one change: each animal's learning rate is 0.04 x its own gene, so founders at 0.5 start at stage 14a's 0.02, and the rate can evolve between 0 and 0.04. These worlds don't use the existing brain gene for anything else, so it takes this job (`--gene-lr`), and it's named `brain` in the metrics. It's inherited with mutation like every gene, and in Lamarckian runs the child gets the parent's gene and the parent's weights as they stand.

Arms: Lamarckian and Baldwinian, on `tiny_stealth` and `tiny_winter`, seeds 0 and 1. The control for drift is stage 14a's Darwinian runs, where this gene has no effect on anything.

## Success test (can fail)

Over the last 15 updates, on `tiny_winter`, on both seeds, with hunters:

1. **Evolution turns learning down:** Lamarckian hunters' gene at least 0.15 below the Darwinian control's.
2. **That rescues the worlds:** Lamarckian worlds last in at least 0.8 of cases at step 1,999, up from 0.516 and 0.547.

Prediction: bar 1 passes, with the gene falling well under the control, and bar 2 is less sure, since 150 updates may be too few for it to fall far enough. Baldwinian should sit about at the control, since learned weights aren't passed on there, so the gene is under little pressure either way.

Reported, not bars: everything on `tiny_stealth`, grazers' gene, and how the gene's spread changes.

## Result (2026-11-01): FAIL on bar 2, and evolution does turn learning down

Hunters' gene over the last 15 updates, and both species alive at step 1,999:

| | hunters' gene, seeds 0, 1 | grazers' gene | both alive | hunters at the end |
|---|---|---|---|---|
| `tiny_winter` Darwinian control | 0.425, 0.563 | 0.522, 0.560 | 0.812, 0.938 | 21.2, 27.4 |
| `tiny_winter` Lamarckian | **0.264, 0.256** | 0.380, 0.372 | **0.531, 0.797** | 11.0, 20.0 |
| `tiny_winter` Baldwinian | 0.337, 0.398 | 0.480, 0.399 | 0.984, 0.969 | 31.1, 28.7 |
| `tiny_stealth` Darwinian control | 0.489, 0.537 | 0.598, 0.429 | 0.938, 0.688 | 29.0, 19.6 |
| `tiny_stealth` Lamarckian | 0.409, 0.388 | 0.481, 0.382 | 0.844, 0.703 | 26.0, 21.2 |
| `tiny_stealth` Baldwinian | 0.361, 0.405 | 0.559, 0.351 | 0.969, 0.969 | 30.6, 29.6 |

1. **Evolution turns learning down: met.** 0.161 and 0.307 under the control.
2. **That rescues the worlds: missed.** 0.531 and 0.797 against a bar of 0.8. Lamarckian worlds without the gene were 0.516 and 0.547, so seed 1 recovers a good deal and seed 0 doesn't.

My prediction held on both.

**It isn't only a Lamarckian thing, though I expected it to be.** I said Baldwinian's gene would sit at the control. It falls too, by 0.09 and 0.17 in winter, and on `tiny_stealth` it's as low as Lamarckian's. So selection seems to push learning down wherever learning changes weights, which fits this rule being costly to have at all, more than it fits only stale inheritance. The gene's spread is wide, about 0.2 across hunters, and its drift differs a good deal between seeds, so read these as rough.

**Tunable learning helps Baldwinian worlds:** in winter they go from 0.953 and 0.922 to 0.984 and 0.969, and on `tiny_stealth` from 0.953 and 1.000 to 0.969 and 0.969. That's small against sampling noise at 64 worlds.

These runs are about 50 generations. Stage 14c gives them more than six times as long.

