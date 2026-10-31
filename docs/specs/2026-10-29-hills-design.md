# Stage 8a: hills on the grid

Date: 2026-10-29. The first step of Stage 8 (toward realism), kept inside the cheap JAX grid. Written
before any run.

## Why

Before moving to continuous physics, test whether a third dimension changes behaviour at all. Hills
give the grid three things a flat world lacks: a cost to moving (climbing), a reason to go somewhere
that is not food (high ground sees further), and cover (a ridge hides what is behind it). Cover is
the interesting one. A predator can close in unseen, and prey can hide without a burrow.

## World

`hills` (bool, off by default). With it on:

- **Height map.** Each cell has a fixed height in [0, 1]: the sum of 10 waves whose periods divide the
  map, so the torus stays seamless, normalised to [0, 1]. `hill_seed` picks the map. The map is the
  same in every episode.
- **Climbing.** An animal pays `climb_cost` (4) energy per unit of height gained in a step. On the
  48×48 map a step changes height by 0.07 on average (0.14 at the 90th percentile), so an uphill step
  costs about 0.3. A predator's base upkeep is 0.5.
- **High ground.** By day, sight radius grows by `floor(hill_sight · 2 · max(h − 0.5, 0) + 0.5)`
  (`hill_sight` 2). That is +0 on 68% of cells, +1 on 30% and +2 on the peaks (2%).
- **Line of sight.** An animal's eye is `eye_height` (0.05) above its cell. A cell in range is hidden
  when any cell on the straight line to it rises above the line from the eye to the target's ground.
  Hidden cells show nothing in any channel, like cells out of range. On the default map this hides
  19% of the cells within radius 2, 32% within radius 3 and 41% within radius 4.
- **Observation.** A new channel gives each visible cell's height relative to the animal's own (0
  where hidden). It comes after the signal channel and before the brain gene, which stays last.

Registered worlds: `hills` (evo + hills) and `flank_hills` (flank + hills).

The 3-D viewer draws the real height map for these worlds (`episode3d.json` carries it).

## Success test (can fail)

`pp-hills-001` on `hills` and `pp-flank-hills-001` on `flank_hills`, seed 0, 30 minutes each.
`python -m scripts.hills --run-dir runs/NAME --evolved` evaluates the latest brains on fixed seeds
three ways: as trained, with the predators' height channel zeroed, and with the prey's height channel
zeroed. Hills still hide and still cost energy when the channel is zeroed; only the animal's
knowledge of the slopes is taken away.

- **Pass (per run):** predators catch at least 10% less without the channel, or prey live at least
  10% less long without it.
- **Stage pass:** both runs pass.
- Reported: time spent on high ground (above mid height) for trained and random brains, against
  the map's share; catches and prey lifetime; gene means; and a 3-D replay of each run.

## Result (2026-10-29): FAIL in both runs

Seed 0, about 150 updates each, evaluated with evolved bodies on 256 fixed seeds.

| run | catches per predator-step (trained / predators blind / prey blind) | prey lifetime (trained / prey blind) | predator drop | prey drop | learned ÷ chaser |
|---|---|---|---|---|---|
| pp-hills-001 | 0.0107 / 0.0102 / 0.0103 | 75.7 / 74.2 | 4.8% | 1.9% | 1.6× |
| pp-flank-hills-001 | 0.0028 / 0.0030 / 0.0028 | 71.9 / 70.5 | −5.4% | 2.0% | 2.3× |

Neither species loses 10% without the height channel. **FAIL.** The bar is not moved.

Time on high ground (above mid height; 64% of the map):

| run | trained prey | trained predators | random prey | random predators |
|---|---|---|---|---|
| pp-hills-001 | 65% | **78%** | 63% | 64% |
| pp-flank-hills-001 | 67% | **56%** | 63% | 64% |

What it shows:

- **The animals do use the terrain; the slope channel adds little to catching.** In the plain hills
  world, trained predators spend 78% of their time on high ground, against 64% for random ones,
  perhaps to see further. In flank_hills, predators stay *lower* than chance (56%), perhaps to close
  in unseen. Zeroing the slope channel barely changes catches, so these habits may come from what an
  animal sees (more cells come into view on a hill, fewer behind one) rather than from the slope.
  That was not measured: high-ground time was not recorded with the channel zeroed. The test
  measured the channel, not terrain use.
- **Hills make flanking harder.** Learned flank predators beat chasers 2.3× among hills, against
  3.0-3.8× on flat ground (Stage 6c), and catch about a quarter as often as predators in plain hills.
- A fairer follow-up (8a v2) would compare against a flat-world brain playing the hills world, or
  measure the high-ground habit directly, with its own bar set before running.
