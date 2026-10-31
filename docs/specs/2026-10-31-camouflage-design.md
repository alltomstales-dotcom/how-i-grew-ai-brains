# Stage 10f: can grazers dodge what they can't see coming?

Date: 2026-10-31. Written before any of these runs.

## Why

Stage 10c's swap showed trained grazers are what kill hunters. Meals, walking costs, thickets and speed haven't changed that. Those all act on hunters or on the map. This acts on what grazers perceive: hunters are camouflaged, and a grazer sees one only from the next cell. I have no measure yet of how grazers keep away from hunters, so this also tests whether seeing them coming is how they do it.

## World

`lasting` plus `stealth_radius` 1: in a grazer's view, the hunter channel is blank beyond 1 cell. Hunters see grazers as before, and grazers see plants and each other as before. `lasting_stealth` has meals of 25, `lasting_stealth_m100` has meals of 100 with `pred_energy_cap` 200. `pp-st-{0,1}`, `pp-stm100-{0,1}`, seeds 0 and 1, 150 updates each.

## Success test (can fail)

For at least one of the two worlds, on both seeds, the same three bars as stage 10d:

1. **It lasts:** `python -m scripts.persist --run-dir runs/NAME --evolved --steps 2000`, 256 fixed seeds, both species alive at step 1,999 in at least 0.5 of worlds.
2. **Grazers aren't just stuck:** at most 90 grazers per world at step 1,999.
3. **They still hunt:** a fixed chaser catches at most 1.0x what learned hunters do (`chase.py --evolved`).

Prediction: survival rises, since catch rates should rise, and I'd guess it stays under 0.5 at meal 25. The best so far is 0.098.

Reported, not bars: catch rates against stage 10b and 10c, grazer numbers, and whether cycles appear in single worlds.

## Result (2026-10-31): FAIL by the rule, and the first trained brains in a world that lasts

| 256 seeds, evolved | `lasting_stealth` (meal 25) | `lasting_stealth_m100` (meal 100) | bar |
|---|---|---|---|
| both alive at step 1,999 | **0.957, 0.961** | 1.000, 1.000 | at least 0.5: met |
| grazers per world at step 1,999 | 96, 96 | 96, 96 | at most 90: missed |
| fixed chaser against learned hunters | 0.64x, 0.66x | 3.09x, 3.07x | at most 1.0: met only at meal 25 |

Bar 2 fails everywhere, so this fails by the rule, and I'm not moving the bar. It was there because stage 10a's worlds sat on the caps, and these do too.

But meal 25 is unlike anything before it. The best share so far was 0.098, at a meal ten times the size. Hunters hold 29.7 and 30.5 per world at step 1,999, and catch 0.0457 and 0.0441 per hunter-step in `chase.py`, about seven times stage 10b's 0.0062. Their sight gene is 0.48 and 0.45 against 0.09 in stage 10b, and their speed 0.32 and 0.29 against 0.18 and 0.16. Grazers are born and die at about 1.3 per step per world, so hunters are eating them and the cap refills them. My guess was that survival would stay under 0.5 at meal 25, and that was wrong.

At meal 100 the worlds also last, at the hunter cap of 32, but learned hunters catch a third of what a fixed chaser does. They can live on few catches and needn't hunt well.

This supports the idea that seeing hunters coming is how grazers keep them from a living, since this is the only thing changed. Fixed hunters do well here too (`chase.py` 0.0291 and 0.0292, up from about 0.0046).

**Remaining doubt:** the numbers sit at a limit in the code, grazers on theirs and hunters near theirs. Stage 10g gives them room.

