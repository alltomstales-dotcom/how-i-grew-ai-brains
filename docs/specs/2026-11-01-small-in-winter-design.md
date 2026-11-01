# Stage 12h: why do tundra's grazers end up small?

Date: 2026-11-01. Written before any of these runs.

## Why

Stage 12g: grazers reach about 0.90 in meadow and about 0.14 in tundra, the opposite of my guess, and hunter pressure doesn't obviously explain it. Tundra is meadow plus winter. My guess now is upkeep: a big body costs up to 0.5 energy every step, and winter thins the food that pays for it. That can be tested cheaply in the small map.

## World

`bulk_feast_50` (stage 12f, where grazers reach 0.91 with no winter), plus tundra's winter, with no crossing:

- `bulk_winter`: winter on, size as usual.
- `bulk_winter_free`: winter on, and size costs grazers nothing (`prey_size_cost` 0); hunters pay as usual.

Seeds 0 and 1, 150 updates, with one seed on each machine.

## Success test (can fail)

Grazer size over the last 15 updates, on both seeds:

1. **It reproduces here:** `bulk_winter` at most 0.4. Otherwise tundra's small grazers need something the small map lacks, such as arrivals or size of the map, and the second arm says little.
2. **Upkeep is why:** `bulk_winter_free` at least 0.7.

If bar 1 is met and bar 2 isn't, upkeep isn't enough to explain it, and other things to look at include catches, digestion, and how winter changes who survives.

Prediction: both bars met. Reported, not bars: hunter size and numbers, and both species alive at step 1,999.

## Result (2026-11-01): PASS on both bars

| | grazer size, seeds 0, 1 | hunter size | both alive at step 1,999 | hunters | grazers |
|---|---|---|---|---|---|
| `bulk_feast_50`, no winter (stage 12f) | 0.909, 0.909 | 0.94, 0.94 | 0.980, 0.945 | 29.7, 29.5 | 95.3, 95.3 |
| `bulk_winter` | **0.119, 0.120** | 0.68, 0.70 | 0.098, 0.098 | 16.1, 16.9 | 86.7, 87.8 |
| `bulk_winter_free` | **0.926, 0.927** | 0.94, 0.94 | 0.934, 0.844 | 30.1, 29.3 | 95.5, 95.5 |

Both bars are met, by a wide margin, and my prediction held. Without winter, upkeep alone doesn't shrink grazers. With winter and no upkeep, they stay big. Together they shrink to about 0.12 by update 100, about what tundra's reach. So winter plus upkeep is enough to explain tundra's small grazers, with no crossing and no big map.

What this doesn't show: I didn't run no winter with free size, since 12f already has grazers at 0.91. And this doesn't rule out other things adding to it in tundra itself.

**Worlds last better with big grazers.** `bulk_winter` lasts in 0.098 of worlds on both seeds, and `bulk_winter_free` in 0.934 and 0.844. Freeing grazers' upkeep also gives more of them through winter, and bigger meals: a meal is 50 x (0.5 + 0.12) = 31 against 71. So tundra's survival trouble may come partly from its small grazers being small meals. That's a guess, and the two effects aren't separated here.

