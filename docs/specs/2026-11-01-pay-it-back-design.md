# Stage 12d: is it just the hunters' energy budget?

Date: 2026-11-01. Written before any of these runs.

## Why

Stage 12c ordered six worlds, roughly, by one number: a hunter's food minus size's upkeep, per step. Worlds near 1.2 last, and worlds near 0.7 don't. I noticed that after seeing the results, so it needs its own test. If the budget is the whole story, paying the deficit back should restore lasting worlds, even with the arms race still running.

## World

Each arm is a world from stage 12c with bigger meals, sized to bring food minus upkeep back to about `bulk_nocatch`'s 1.2 at last stage's catch rates. Nothing else changes:

- `bulk_rich`: `bulk_base`, the full race, `catch_energy` 37. (0.039 catches x 37 x 1.16 minus 0.46 upkeep is 1.21.)
- `bulk_certain_rich`: `bulk_certain`, `catch_energy` 28, so a typical meal is back to 25.
- `bulk_coin_rich`: `bulk_coin`, `catch_energy` 34. (25 / 0.73 is 34.)

Seeds 0 and 1, 150 updates, with one seed on each machine. Catch rates may change once meals do, since more hunters share the same grazers. That is part of the test.

## Success test (can fail)

`python -m scripts.persist --run-dir runs/NAME --evolved --steps 2000`, 256 fixed seeds, both species alive at step 1,999. An arm lasts at 0.5 or more on both seeds, and dies under 0.5 on both. Anything else is mixed.

What each pattern would mean, fixed now:

| `bulk_rich` | `bulk_certain_rich` | `bulk_coin_rich` | reading |
|---|---|---|---|
| lasts | lasts | lasts | **the budget is enough**: pay it back and the race does no further harm that matters |
| dies | lasts | lasts | **the race does harm beyond energy** |
| any | any | dies | **failed catches do harm beyond their average**, such as long waits between meals |
| any other | | | not separated, and reported as that |

Prediction: all three last. If `bulk_rich` lasts, the next stage puts it on the continents, which is what stage 12a set out to test.

Reported, not bars: size genes, hunter numbers, catches, food minus upkeep, and the chaser ratio.

## Result (2026-11-01): not separated, and the test of the race was flawed

| | seed 0 | seed 1 | hunter size | grazer size | meal | food minus upkeep, on paper |
|---|---|---|---|---|---|---|
| `bulk_rich` | 0.000 | 0.000 | 0.92, 0.92 | 0.71, 0.68 | 44.7, 43.7 | 1.30, 1.26 |
| `bulk_certain_rich` | 0.570 | 0.281 | 0.08, 0.07 | 0.31, 0.28 | 22.8, 21.9 | 1.10, 0.98 |
| `bulk_coin_rich` | 0.887 | 0.586 | 0.08, 0.08 | 0.41, 0.35 | 34 | 1.19, 1.17 |

`bulk_certain_rich` is mixed, so by the table this is "not separated". I predicted all three would last, and they didn't.

**`bulk_coin_rich` lasts on both seeds.** Paying failed catches back in bigger meals works, from 0.219, 0.055 to 0.887, 0.586.

**Grazers answered `bulk_certain_rich`.** They shrank from about 0.40 to about 0.30, so a meal stayed near 22 and never reached the 25 I set out to restore.

**`bulk_rich` doesn't test what I meant it to, because I missed the energy cap.** A hunter holds at most 60 energy, and a meal of about 44 only fits whole below 16. A hunter that has just bred holds about 24, so these meals are often clipped. The 1.30 and 1.26 are paper numbers, and so was "paid back". It's the same limit stage 10c raised for its big meals, and I should have checked. The zeroes stand as results, but they don't show that the budget fails.

The same applies to the table in stage 12c: meals of about 30 to 33 in `bulk_base` and `bulk_cheap` are clipped too, which may be why `bulk_cheap` didn't fit. Stage 12e removes the cap's effect.

