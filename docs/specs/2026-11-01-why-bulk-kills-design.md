# Stage 12b: why does size kill the hunters?

Date: 2026-11-01. Written before any of these runs.

## Why

Stage 12a: switching on size took meadow and tundra from lasting in every world to none, with hunters dropping about tenfold by update 76. Size changes three things at once: it costs energy every step, a catch stops being certain, and a meal's worth depends on the grazer's size. I guessed the energy budget. This separates them, cheaply, in the 48 x 48 world with camouflage and lasting plants, where each run takes about 15 minutes.

## World

Four worlds, all `lasting_stealth`, with no crossing:

- `lasting_stealth` itself is stage 10f's world, already run (`pp-st-0`, `pp-st-1`): the no-size reference, 0.957 and 0.961.
- `bulk_base`: size on as in stage 12a (`size_cost` 0.5, `catch_scale` 0.25).
- `bulk_cheap`: the same, with `size_cost` 0.1.
- `bulk_nocatch`: size costs and digestion stay, and size plays no part in predation (`size_catch` False), so catches are certain and meals are the flat 25.

Seeds 0 and 1, 150 updates. Three worlds x two seeds, with one seed on each machine.

## Success test (can fail)

`python -m scripts.persist --run-dir runs/NAME --evolved --steps 2000`, 256 fixed seeds, both species alive at step 1,999.

1. **It reproduces here:** `bulk_base` under 0.5 on both seeds. If it lasts in this smaller world, stage 12a's collapse needs the big continents, and the other two arms say little.
2. **The cause:** an arm counts as the cause if it reaches at least 0.5 on both seeds.

Prediction: `bulk_nocatch` restores lasting worlds, and `bulk_cheap` restores some without reaching 0.5, since a cheaper body doesn't help a hunter whose catches fail. I hold this loosely.

Reported, not bars: size genes, hunter numbers, catches, and the chaser ratio.
