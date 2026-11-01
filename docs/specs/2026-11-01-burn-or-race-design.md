# Stage 12e: with the cap out of the way, is it burn or the race?

Date: 2026-11-01. Written before any of these runs.

## Why

Stage 12d's `bulk_rich` kept the race and tried to pay hunters back, but the energy cap of 60 clipped its meals, so it never tested that. Big hunters also differ from small ones in a second way: they burn energy about twice as fast, so the same 60 energy lasts about half as long between meals. This stage crosses the two things with all meals fitting whole.

## World

All four use `pred_energy_cap` 120. A hunter breeds at 58.5 energy at most, and the biggest possible meal is 55.5, so every meal fits whole. Energy doesn't bank up, since a hunter above its threshold breeds on the next step while there's room.

| | low burn | high burn |
|---|---|---|
| **no race** (fixed chance 0.73, flat meals) | `bulk_cap_coin`: meals 34 | `bulk_cap_burn`: meals 47, `pred_base_cost` 0.96 |
| **race** (size decides catches and meals) | `bulk_cap_free`: hunters pay no upkeep for size (`pred_size_cost` 0), grazers pay as usual, `catch_energy` 25 | `bulk_cap_rich`: `catch_energy` 37, upkeep as usual |

Meals are set so that food minus burn above the usual 0.5 base is about 1.2 in every cell, at last stage's catch rates:

- `bulk_cap_coin`: 0.036 x 34 is 1.2.
- `bulk_cap_burn`: 0.036 x 47 minus the extra 0.46 is 1.2.
- `bulk_cap_free`: 0.04 x 25 x (0.5 + about 0.75) is 1.25. Grazer size is a guess, and I report what it ends up at.
- `bulk_cap_rich`: 0.039 x 37 x 1.2 minus 0.46 is 1.27.

Seeds 0 and 1, 150 updates, with one seed on each machine.

## Success test (can fail)

`python -m scripts.persist --run-dir runs/NAME --evolved --steps 2000`, 256 fixed seeds, both species alive at step 1,999. A cell lasts at 0.5 or more on both seeds, and dies under 0.5 on both. Anything else is mixed.

`bulk_cap_coin` has to last, or the cap itself did something and the other cells can't be read. Given that:

| `bulk_cap_burn` | `bulk_cap_free` | `bulk_cap_rich` | reading |
|---|---|---|---|
| lasts | lasts | lasts | **the budget is enough**: pay it back in meals that fit and neither burn nor the race matters. Stage 12d's zeroes were the cap |
| dies | lasts | dies | **burn kills**: fast-burning hunters die even when the average is paid back |
| lasts | dies | dies | **the race kills**, beyond energy |
| lasts | lasts | dies | **burn and race only kill together** |
| any other | | | not separated, and reported as that |

Prediction: all four last, so the cap was the flaw and the budget is the story. I hold this loosely, having got the last two wrong.

Reported, not bars: size genes, hunter numbers, catches, food minus burn, and the chaser ratio.
