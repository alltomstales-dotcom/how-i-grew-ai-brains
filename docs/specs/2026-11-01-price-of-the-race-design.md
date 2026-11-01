# Stage 12f: can a big enough meal carry the race?

Date: 2026-11-01. Written before any of these runs.

## Why

Stage 12e: with meals that fit and an income of about 1.2 per step, the race still wins out: `bulk_cap_rich` lasts in 0.176 and 0.008 of worlds. That doesn't say the race is impossible to live with. Size on the continents was stage 12a's question, and it needs a world where size decides catches and both species last. So this asks what that costs, with nothing else changed.

## World

`bulk_cap_rich` with bigger meals and a cap high enough to hold them whole:

- `bulk_feast_50`: `catch_energy` 50, `pred_energy_cap` 170.
- `bulk_feast_70`: `catch_energy` 70, `pred_energy_cap` 170.

A hunter breeds at 58.5 energy at most, and the biggest possible meal is 70 x 1.5 = 105, so 170 holds every meal in both worlds.

Seeds 0 and 1, 150 updates, with one seed on each machine.

## Success test (can fail)

`python -m scripts.persist --run-dir runs/NAME --evolved --steps 2000`, 256 fixed seeds. A world passes if, on both seeds:

1. Both species are alive at step 1,999 in at least 0.5 of worlds.
2. Size still races: hunter size at least 0.6 and grazer size at least 0.5, over the last 15 updates. Without this, a world could last just by leaving the race, which wouldn't answer the question.
3. Learned hunters beat the fixed chaser (chaser at most 1.0x).

The stage passes if either world does. If only `bulk_feast_70` passes, I'd use it, and note that it took nearly three times the usual meal. If neither passes, size stays out of predation on the continents and that's a result: this race can't be paid for in meals up to 70.

Prediction: `bulk_feast_50` passes. I've been wrong on my last three predictions, so read this as a guess.

Reported, not bars: hunter numbers against the cap of 32, grazer numbers, plant cover, and income.

## Result (2026-11-01): PASS at both meal sizes

| | seed 0 | seed 1 | hunter size | grazer size | chaser against learned hunters | hunters |
|---|---|---|---|---|---|---|
| `bulk_cap_rich`, meals 37 (stage 12e) | 0.176 | 0.008 | 0.93, 0.93 | 0.83, 0.83 | 0.64x, 0.66x | 16.8, 13.9 |
| `bulk_feast_50` | **0.980** | **0.945** | 0.94, 0.94 | 0.91, 0.91 | 0.75x, 0.72x | 29.7, 29.5 |
| `bulk_feast_70` | **0.996** | **0.988** | 0.93, 0.93 | 0.91, 0.91 | 0.90x, 0.88x | 31.3, 31.3 |

All three bars are met at both sizes, and this is the first world where size decides catches and both species last. The prediction held this time.

- **It's a steep step.** Meals of 37 last in almost no worlds, and meals of 50 in nearly all. So stage 12e's race is a higher price to pay, not a wall, and this qualifies "beyond energy": the same income doesn't carry a race, and more does.
- **Both species run to the top of the range.** Grazers at 0.91 and hunters at 0.94 are close enough that the catch chance is about 0.53. A meal is about 50 x 1.41 = 70, so about half of those are missed and the rest pay well.
- **Both sit near their caps,** hunters at 30 of 32 and grazers at 96 of 96, so this isn't a free-running population.
- **Richer isn't simply better.** Learned hunters' lead over the chaser shrinks at meals of 70, to 0.89x from 0.74x.

Stage 12g uses meals of 50, the smaller size that passes.

