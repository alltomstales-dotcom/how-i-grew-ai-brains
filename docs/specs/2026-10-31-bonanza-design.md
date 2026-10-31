# Stage 9e: is the split made by the rich start?

Date: 2026-10-31. Written before any of these runs.

## Why

Stage 9d's hunters split into plant eaters and meat eaters at a gut of 0.11, but they take turns: plant eaters boom and crash in the first 100 steps, and meat eaters outlast them. Each world starts with half its cells in plants (`init_grass_frac` 0.5), and cover then falls to about 0.19. So the hypothesis is that the rich start is what plant eaters live on, and the split depends on it.

## World

`omni_d11_lean`: as `omni_d11`, starting at `init_grass_frac` 0.2, about the cover worlds settle at. Nothing else changes. That also leaves less food for grazers early on, which is part of what changes.

`pp-lean-{4,5}` and a replicate of the original, `pp-rich-{4,5}` (`omni_d11`), seeds 4 and 5, 300 updates each.

## Success test (can fail)

Scored with `python -m scripts.species --run-dir runs/NAME --gene diet`, stage 9d's bar (last 3 snapshots: D at least 4, smaller group at least 0.2, gap at least 0.6). Both must hold on both seeds:

1. **The split repeats:** `pp-rich` passes that bar.
2. **Without the rich start it goes:** `pp-lean` does not pass it.

If 1 fails, stage 9d's split didn't replicate, which matters more than the rest. If 2 fails, the rich start isn't needed and the hypothesis is wrong.

Reported, not bars: `--camp-curve` on all four, diet histograms, plants' share of hunter energy, and how long hunters last.

## Result (2026-10-31): FAIL by the rule, on the first bar, and the mechanism was not the one I guessed

Last 3 of 300 updates, bar D at least 4, smaller group at least 0.2, gap at least 0.6:

| | D | smaller group | gap | bar result |
|---|---|---|---|---|
| `pp-rich-4` | 7.58, 5.62, 5.86 | 0.32, 0.22, 0.16 | 0.74 to 0.78 | fail, narrowly, on the last snapshot |
| `pp-rich-5` | 6.22, 7.00, 5.60 | 0.37, 0.39, 0.36 | 0.74 to 0.76 | pass |
| `pp-lean-4` | 1.81, 1.80, 1.70 | 0.46, 0.49, 0.49 | 0.17 | no split |
| `pp-lean-5` | 1.72, 1.77, 1.62 | 0.46, 0.45, 0.43 | 0.16 to 0.18 | no split |

Bar 1 needed both rich seeds, so this fails by the rule. The split itself did repeat in kind: four rich runs across stages 9d and 9e all hold D above 5 and a gap of about 0.75 at the end. What varies is how big the meat group gets, 0.16 to 0.39, so three of the four clear the size bar. Top tenth of 4,096 genomes at update 300: 316 and 722 in the rich runs, none in either lean run.

Bar 2 held clearly. Starting at 0.2 plants, no meat group forms: diet 0.135 and 0.127, plants 0.95 and 0.94 of hunter energy.

**My mechanism was backwards.** I thought the rich start fed plant eaters. Plant eaters keep up without it only to step 50 (13.5 and 14.0 per world, against 12.3 and 9.8), and then crash harder: 0.15 and 0.12 at step 75 against 3.45 and 2.68. (Corrected: I first wrote that they "do fine".) But it's the meat group that fails to form at all. With a lean start grazers stay at about 24 at step 50, against about 95 with a rich one, so there's no boom to hunt. The likely story is a chain: rich plants, then a grazer boom, then enough meat for a meat-eating kind. That's a reading of these curves; I didn't test it directly. Hunters also die out faster on a lean start, gone by step 100.

**The wider point.** In every stage 9 world with real hunters, worlds lose them well before the end of an episode, and restarting from the pool is what keeps the kinds going. Stage 10a asks whether hunters can last at all.

