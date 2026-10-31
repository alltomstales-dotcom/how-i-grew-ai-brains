# Stage 10a: is there a world that lasts?

Date: 2026-10-31. Written before the sweep. It replaces a first version, withdrawn with no training run.

## Why

Stage 9 showed real hunters dying out within an episode, with restarts from a saved pool keeping them in the story. The first version of this spec asked whether bigger meals (50, 100, 150) would let hunters last, scored by both species alive at the end of the episode.

Its validity check on `pp-fatfix-0` stopped it. With meals of 25 both species alive at step 499 was 0.000, as expected, but grazers died out too: 86 at step 100, 0.3 at step 499, even with hunters gone by step 300. Then three probes, on brains already trained:

- Plants spreading 2x, 4x and 8x as fast left 0.5, 0.7 and 4.8 grazers at step 499.
- About 220 unbitten plants were still on the 48 x 48 map at the end, with grazers at about 8.6 energy.
- A hand-written grazer that walks to the nearest plant in view also died out (0.0 at step 499), and plant cover fell the same way, 0.50 to 0.09, whoever was eating.

So bigger meals couldn't have passed, and the test would have taught little. Every world in this project so far has been a boom and bust of about 150 to 300 steps. That doesn't undo earlier results, each of which measured what happens in those worlds, but it limits what they can say about lasting populations.

## Test

No training. `python -m scripts.lasting`: hand-written grazers (walk to the nearest plant in view) and hand-written hunters (walk to the nearest grazer in view), with evolution on, 64 worlds, 2,000 steps, over a grid of `seed_rate` (0.03, 0.08, 0.2), `bite_regrow` (0.1, 0.3), `catch_energy` (25, 75, 150) and `die_rate` (0.02, 0.005).

**Bar:** at least one setting has both species alive at step 2,000 in at least half of worlds, on two separate sets of random seeds. The hand-written animals are a floor: they don't flee or plan, so a pass says the world can hold both, and a fail says it probably can't with these settings.

Reported: grazers, hunters and plant cover over time, and whether numbers cycle. If a setting passes, stage 10b trains brains in it, with worlds that don't restart when a species dies.

## Result (2026-10-31): PASS, with a large caveat

12 of 36 settings hold both species at step 2,000 in at least half of worlds on both seed sets. All 12 are the settings with `seed_rate` 0.2, and none of the 24 at 0.03 or 0.08 do (best 0.05). So plant spread is a sharp threshold, and the other three knobs barely matter. Shares for the 12 run from 0.81 to 1.00.

**The caveat:** these worlds sit at or near the slot caps, about 96 grazers and 32 hunters, in many settings exactly. So the cap may be what holds them up: 32 hunters can't eat grazers faster than they breed, and there's no visible cycle of booms and busts. A code comment from long ago already noted that 0.2 "left food plentiful and prey stuck at the slot cap". This shows a lasting world exists for simple animals, not that it behaves like an ecosystem.

`catch_energy` 75 and 150 give identical curves, as `pred_energy_cap` 60 clips any meal that size. Stage 10b trains brains in this world.

