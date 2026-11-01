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

## Result (2026-11-01): the table says the race, and my prediction was wrong

Both species alive at step 1,999:

| | low burn | high burn |
|---|---|---|
| **no race** | `bulk_cap_coin` **0.996, 1.000** | `bulk_cap_burn` **0.723, 0.543** |
| **race** | `bulk_cap_free` 0.402, 0.297 | `bulk_cap_rich` 0.176, 0.008 |

`bulk_cap_coin` lasts, so the other cells can be read. Burn lasts, and both race cells die, which is the row for **the race kills, beyond energy**. I predicted all four would last.

What the last 15 updates show, with both seeds:

| | catches per hunter | meal | hunter speed | hunters | income | income after speed |
|---|---|---|---|---|---|---|
| `bulk_cap_coin` | 0.033, 0.035 | 34 | 0.33, 0.35 | 29.6, 29.6 | 1.09, 1.13 | 0.98, 1.01 |
| `bulk_cap_burn` | 0.042, 0.041 | 47 | 0.48, 0.45 | 25.1, 25.5 | 1.46, 1.39 | 1.23, 1.19 |
| `bulk_cap_free` | 0.030, 0.028 | 34.1, 33.1 | 0.33, 0.31 | 20.1, 14.0 | 1.01, 0.94 | 0.90, 0.84 |
| `bulk_cap_rich` | 0.036, 0.034 | 49.3, 49.2 | 0.52, 0.50 | 16.8, 13.9 | 1.32, 1.21 | 1.05, 0.96 |

Income here is food minus size's upkeep and any extra burn. Speed's cost is speed squared.

- **The average doesn't explain it.** `bulk_cap_rich` has about the income of `bulk_cap_coin`, and it lasts in 0.18 and 0.01 of worlds against 1.00. `bulk_cap_coin`'s hunters sit near their cap of 32, so its income is limited by crowding, not by what the world can pay. So the budget idea from stage 12c doesn't hold as the whole story.
- **`bulk_cap_free` wasn't fully paid back.** I guessed grazers would reach about 0.75, and they reached 0.86 and 0.82, which cuts catches. Its income is about 15% under the 1.2 I set out for, so it's weaker evidence than `bulk_cap_rich`, and 12c shows these hunters are sensitive to about that much.
- **Burn matters too, and less.** It costs about 0.3 to 0.45 of lasting worlds without a race, and about 0.2 to 0.3 with one.
- **The cap mattered by itself.** Meals of 34 went from 0.887, 0.586 under the cap of 60 to 0.996, 1.000 under 120.

The race runs hunters to about 0.94 and grazers to about 0.85, near the top of the range, which brings the catch chance down to about 0.60. The no-race cells were fixed at 0.73. So part of "the race" may be that lower chance, and this doesn't show how the race does its harm. Size's effect on individuals may matter too: a catch chance that varies from pair to pair, say.

