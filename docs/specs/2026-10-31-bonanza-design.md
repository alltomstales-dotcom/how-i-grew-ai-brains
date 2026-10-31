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
