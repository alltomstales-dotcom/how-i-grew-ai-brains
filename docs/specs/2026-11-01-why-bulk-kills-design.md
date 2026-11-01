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

## Result (2026-11-01): the cause is in predation, not the energy bill

Both species alive at step 1,999, and means over the last 15 updates:

| | seed 0 | seed 1 | hunter size | grazer size | hunters in training |
|---|---|---|---|---|---|
| `lasting_stealth`, no size (stage 10f) | 0.957 | 0.961 | | | |
| `bulk_base` | 0.000 | 0.000 | 0.92, 0.92 | 0.69, 0.63 | 6.3, 6.4 |
| `bulk_cheap` | 0.004 | 0.004 | 0.94, 0.94 | 0.84, 0.81 | 8.8, 11.7 |
| `bulk_nocatch` | **0.984** | **0.852** | 0.08, 0.08 | 0.41, 0.39 | 29.4, 28.2 |

1. **It reproduces here: met.** The collapse doesn't need big continents or crossings.
2. **The cause: `bulk_nocatch` meets it, `bulk_cheap` doesn't.** A fifth of the energy bill changed almost nothing, so the bill alone isn't why, and size's part in predation is.

My prediction was half right: `bulk_nocatch` did restore lasting worlds, and `bulk_cheap` restored essentially none, where I'd guessed some.

An arms race shows up in the genes. Where size decides catches, hunters climb to about 0.92 by update 60 and grazers keep following, to 0.84 when bodies are cheap. Where it doesn't, hunters shrink to 0.08, since size then only costs them. Learned hunters beat the chaser in all six runs (0.64x to 0.72x), so in `bulk_base` they aren't failing to learn to hunt.

This doesn't separate catches from meals. `bulk_nocatch` removed both. Stage 12c does.

**A correction from stage 12c.** "Size's part in predation, not its energy bill" is too strong. The cheap body did save upkeep (about 0.46 down to 0.09 per step), but the race took much of it back: catches per hunter fell from about 0.040 to 0.030, so food minus upkeep only rose from about 0.69 to 0.90. So this stage shows that cheaper bodies don't rescue hunters when size still decides catches. It doesn't show that energy doesn't matter.

