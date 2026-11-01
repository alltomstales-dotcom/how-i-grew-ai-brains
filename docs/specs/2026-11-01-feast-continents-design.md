# Stage 12g: with a race that lasts, does size follow climate?

Date: 2026-11-01. Written before any of these runs.

## Why

Stage 12a asked whether a body part that matters follows climate, and couldn't answer: size wiped out the hunters. Stage 12f found a world where size decides catches and both species last, in the small map, at meals of 50. So this asks 12a's question again, with its bars unchanged.

## World

The three continents of stage 12a, with the same crossing, pace, displacing and lineage tags, plus `catch_energy` 50 and `pred_energy_cap` 170. Worlds `bulk_feast_meadow`, `bulk_feast_forest`, `bulk_feast_tundra`. Sets I (seeds 46, 47, 48) and J (seeds 49, 50, 51).

## Success test (can fail)

Stage 12a's bars, unchanged. Means over the last 10% of training, on both sets:

1. **Size follows climate:** in grazers or hunters, size differs by at least 0.15 between highest and lowest continents, with the same order on both sets.
2. **Worlds still last:** `scripts.persist --evolved --steps 2000 --episodes 32`, both species alive at step 1,999 in at least 0.5 of worlds, on meadow and tundra.
3. **Hunters still hunt:** a fixed chaser catches at most 1.0x what learned hunters do, on meadow and tundra.

Prediction: bars 2 and 3 pass, since the small map does. Bar 1 is the risk. In stage 12f both species ran to the top of the size range, which leaves little room for a difference, and meadow and tundra may swap places, which fails "same order" as it did in stage 12a. Forest's hunters shrank to about 0.05 in stage 12a; if they do again, the gap is easily met and the order between the other two decides it.

Reported, not bars: grazer sight's order, forest's hunters, hunter numbers against stage 12a's, and migrants' shares.

## Sets I and J are void, and rerun as I2 and J2 (2026-11-01)

Set I's tundra trainer stopped at update 60 of 600, and set J's meadow trainer at about the same point. The kernel's out-of-memory killer took both (`OOMKilled` true, exit 137). So a ring member was gone for nearly all of each set, and neither is scored. Evaluations still ran on the early checkpoints and printed normal-looking numbers, which is why every run's update count is checked before scoring now.

**The cause is my own video clips.** At each checkpoint a trainer renders a clip, holding every frame in memory. A 256 x 256 map gives frames of 2560 x 2592 x 3 bytes, about 20 MB, so 500 frames is about 10 GB, and both of the Spark's trainers do this at the same update. One of the two died in each set, and the other then ran on alone without trouble. Set J's survivor held 12 GB of ordinary memory after its clip.

It hadn't bitten since stage 11c because worlds that die early give short clips: sets G and H had 200 to 370 frames, where these had 500. Sets A to F had 500 too and survived, so the margin was already thin. I don't have a record of how much memory was free then.

I first guessed a burst from something else on the machine. That was wrong.

Sets I2 and J2 use the same seeds with `--no-clips` in the trainer, and nothing else changes. The bars haven't moved, and I'm excluding the first two sets on the crash alone.

## Result (2026-11-01): FAIL on tundra's survival, and size follows climate

All six trainers reached update 600. Means over the last 60 updates, for sets I2 and J2:

| | meadow | forest | tundra | bar |
|---|---|---|---|---|
| grazer size | 0.888, 0.901 | 0.234, 0.334 | 0.126, 0.154 | **met**: a gap of 0.76 and 0.75, same order |
| hunter size | 0.935, 0.935 | 0.843, 0.890 | 0.702, 0.780 | **met**: a gap of 0.23 and 0.16, same order |
| both alive at step 1,999 | 1.000, 1.000 | 0.000, 0.000 | 0.344, 0.875 | meadow and tundra at least 0.5: **missed**, by tundra on I2 |
| chaser against learned hunters | 0.76x, 0.75x | 0.70x, 0.74x | 0.66x, 0.73x | meadow and tundra at most 1.0: met |
| grazer sight | 0.826, 0.682 | 0.303, 0.301 | 0.091, 0.086 | reported, same order |
| hunters per world in training | 874, 878 | 64, 60 | 400, 399 | reported |
| catches per step | 25.0, 24.3 | 2.6, 2.5 | 20.5, 21.3 | reported |

Bars 1 and 3 are met, and bar 2 fails on one number, 0.344. It doesn't move, so this is a FAIL. Against stage 12a's zeroes, though, meadow went to lasting in every world, and tundra to 0.344 and 0.875.

My prediction was backwards: I said survival would pass and climate was the risk. And stage 12a's guess that tundra's grazers would be biggest is the opposite of what happens. They're the smallest, at about 0.14 against meadow's 0.90.

Meadow's size curves show a race. Hunters reach 0.9 by about update 50 and grazers follow to 0.9 by about update 200. In tundra grazers go the other way, to about 0.11 by update 100, and hunters come partway down. So the race runs in one climate and not in another. Why tundra's grazers end up small isn't tested here. Upkeep through winter is an obvious guess, but only that.

Forest's hunters are carried by arrivals: every one descends from migrants, about 60 per world, and with no crossing in the evaluation they die out by step 600. They also stay big, at about 0.87, unlike stage 12a's 0.05, probably since they're newly arrived from a continent where big pays, which isn't tested either.

Meadow's hunters sit near their cap of 910, so its numbers are limited by the cap. Tundra's evaluation on I2 fell from 1.000 at step 600 to 0.344, so its worlds decline slowly, and a longer run might see more of them end. Set J2's meadow hunters show 0.705 from arrivals at the end, where I2's show 0.025, so that share swings a lot and I wouldn't read much into it.

