# Stage 12c: the failed catch, the meal, or the arms race?

Date: 2026-11-01. Written before any of these runs.

## Why

Stage 12b: taking size out of predation restores lasting worlds (0.984, 0.852), and a cheaper body doesn't (0.004, 0.004). But that arm removed two things together: catches that depend on size, and meals that depend on size. It also stopped an arms race, in which hunters climb to about 0.92 and grazers follow. If failed catches are the trouble, any failed catch should do it. If the race is the trouble, the same failure rate with no race should be fine.

## World

Three worlds, all `bulk_base` (stage 12a's rule, `lasting_stealth` with size) with no crossing:

- `bulk_certain`: every catch succeeds; a meal is still worth 25 x (0.5 + grazer size).
- `bulk_flat`: catches depend on sizes as in `bulk_base`; every meal is worth a flat 25.
- `bulk_coin`: every catch succeeds with a fixed chance of 0.73, whatever the sizes, and every meal is a flat 25. 0.73 is `bulk_base`'s chance at its final sizes: sigmoid((hunter size - grazer size) / 0.25) is 0.71 and 0.76 on its two seeds. Size then does nothing for a hunter, so nothing should drive a race.

Seeds 0 and 1, 150 updates, with one seed on each machine.

## Success test (can fail)

`python -m scripts.persist --run-dir runs/NAME --evolved --steps 2000`, 256 fixed seeds, both species alive at step 1,999. An arm lasts if it reaches at least 0.5 on both seeds. It fails if it stays under 0.5 on both. Anything else is mixed, and I'll say so.

What each pattern would mean, fixed now:

| `bulk_certain` | `bulk_flat` | `bulk_coin` | reading |
|---|---|---|---|
| lasts | dies | lasts | **the arms race**: failed catches at this rate are survivable when size isn't racing |
| lasts | dies | dies | **failed catches**: at this rate they kill hunters however they arise |
| dies | lasts | any | **meals tied to size** are the trouble |
| any other | | | not separated, and reported as that |

Prediction: the first row, the arms race. I hold this loosely. A hunter in `bulk_coin` should also end up small, so it pays no upkeep, which makes it a kinder test than a pure failure rate. That's a limit of this design: if it lasts, the race could matter through either the upkeep or the falling catch odds.

Reported, not bars: size genes, hunter numbers, catches, and the chaser ratio.

## Result (2026-11-01): not separated, and my prediction wasn't confirmed

| | seed 0 | seed 1 | hunter size | grazer size | hunters in training |
|---|---|---|---|---|---|
| `bulk_certain` | 0.469 | 0.434 | 0.08, 0.08 | 0.41, 0.38 | 21.8, 20.9 |
| `bulk_flat` | 0.000 | 0.000 | 0.88, 0.88 | 0.52, 0.47 | 6.2, 6.3 |
| `bulk_coin` | 0.219 | 0.055 | 0.07, 0.07 | 0.44, 0.43 | 13.4, 11.6 |

No arm lasts, so this is the table's "any other" row. `bulk_certain` is close to the bar, under it on both seeds, and it stays a miss. I predicted the arms race alone, which needed `bulk_certain` and `bulk_coin` to last.

Two comparisons in these runs change one thing each:

- `bulk_nocatch` against `bulk_coin`: the only change is a fixed 0.73 chance instead of a certain catch, with no race in either. Lasting falls from 0.984, 0.852 to 0.219, 0.055.
- `bulk_nocatch` against `bulk_certain`: the only change is meals worth about 22 instead of 25. Lasting falls from 0.984, 0.852 to 0.469, 0.434.

So failed catches do harm with no race at all, and so does a meal about 11% smaller. These hunters live close to the edge.

## After the fact: one number nearly orders them

This reading came after seeing the results, so it's a hypothesis, tested in stage 12d. Food per hunter per step is catches per step / hunters x meal, and subtracting size's upkeep gives this, from the last 15 updates:

| | catches per hunter | meal | food minus upkeep | both alive at step 1,999 |
|---|---|---|---|---|
| `bulk_nocatch` | 0.050, 0.051 | 25 | 1.20, 1.22 | 0.984, 0.852 |
| `bulk_certain` | 0.049, 0.047 | 22.7, 22.0 | 1.07, 1.01 | 0.469, 0.434 |
| `bulk_coin` | 0.036, 0.037 | 25 | 0.87, 0.90 | 0.219, 0.055 |
| `bulk_cheap` | 0.028, 0.032 | 33.5, 32.7 | 0.84, 0.96 | 0.004, 0.004 |
| `bulk_base` | 0.038, 0.041 | 29.8, 28.4 | 0.68, 0.70 | 0.000, 0.000 |
| `bulk_flat` | 0.043, 0.045 | 25 | 0.65, 0.67 | 0.000, 0.000 |

`bulk_cheap` doesn't fit: it has about `bulk_coin`'s income and far fewer lasting worlds. Its hunters catch least often, so they wait longest between meals, which may matter beyond the average. These numbers come from training, so they describe the same populations and aren't independent of how many hunters survive.

