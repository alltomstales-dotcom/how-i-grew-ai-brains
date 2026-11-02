# Stage 15e: is it the place, or the mix?

Date: 2026-11-01. Written before any of these runs.

## Why

Stage 15d: worlds with mutation zones last far better than even mutation at the same average. Zones do two things at once: they make mutation depend on where a child is born, and they make it uneven, with most children mutating little and a few a lot. My guess was that the second is what counts, since most animals then carry less mutation damage. If so, the environment isn't doing anything by itself, and the same mix at random should work as well. That matters for Tom's idea: only if place adds something is this really environmental.

## World

`mut_random_tundra`: isolated tundra as in stage 15d, on the same seeds (58 and 59), 600 updates, sets A2 and B2. Each child, wherever it's born, has a one in four chance of mutating at 5x, and otherwise mutates at 1x. Same average and same mix as zones, with no place.

## Success test (can fail)

Both species alive at step 1,999, `scripts.persist --evolved --steps 2000 --episodes 32`, on both sets. References from stage 15d: zones 1.000, 1.000, and even 0.719, 0.156.

1. **The mix is enough:** `mut_random_tundra` at least 0.25 above even mutation, and lasting in at least 0.5 of worlds.

What the result would mean, fixed now:

- **Bar met, and within 0.25 of zones on both sets:** it's the mix. Place adds nothing that this test can see, so "environmental" mutation helps here because it's uneven, not because it's environmental.
- **Bar missed:** place matters beyond the mix. Animals may be keeping away from zones, or a zone's lineages may behave differently, and that's worth following up.
- **Anything else:** not separated, and reported as that.

Prediction: the bar is met, and it's the mix. Reported, not bars: hunters' gene and hunter numbers.

## Result (2026-11-01): the bar is missed, and two sets can't settle this

| isolated tundra | both alive at step 1,999, A2, B2 | hunters' gene | hunters |
|---|---|---|---|
| zones (stage 15d) | 1.000, 1.000 | 0.888, 0.886 | 283, 297 |
| `mut_random_tundra` | **1.000, 0.000** | 0.894, 0.891 | 313, 214 |
| even (stage 15d) | 0.719, 0.156 | 0.904, 0.904 | 256, 256 |

**The mix is enough: missed.** On B2 the random mix is at 0.000, under even's 0.156. By the reading fixed above, that says place matters beyond the mix, and my prediction was wrong.

I don't trust that reading yet. The same arm scored 1.000 and 0.000, so one set can swing the whole answer, and the fixed reading was written without knowing how wide that swing would be. It stands as what the rule says, and stage 15f measures it properly with more sets.

