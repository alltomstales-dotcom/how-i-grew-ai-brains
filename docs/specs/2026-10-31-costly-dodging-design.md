# Stage 10d: can a viewer's idea make a world last?

Date: 2026-10-31. Written before any of these runs.

## Why

Stage 10c's probe found trained grazers are what kill hunters: dodging is free here, so hunters starve. Moving costs nothing and standing still saves nothing. Stage 8b, a viewer's idea, added walking costs and thickets that hide animals, and hunters there learned to wait in cover. So: if running costs grazers energy and hunters can hide, do both species last?

## World

Stage 8b's `ambush` rules (`move_cost`, thickets) with stage 10a's plants (`seed_rate` 0.2). `lasting_ambush` has meals of 25, `lasting_ambush_m100` has meals of 100 with `pred_energy_cap` 200. Nothing else changes. `pp-la-{0,1}`, `pp-lam100-{0,1}`, seeds 0 and 1, 150 updates each.

## Success test (can fail)

For at least one of the two worlds, on both seeds:

1. **It lasts:** `python -m scripts.persist --run-dir runs/NAME --evolved --steps 2000`, 256 fixed seeds, both species alive at step 1,999 in at least 0.5 of worlds.
2. **Grazers aren't just stuck:** grazers below the slot cap of 96 at step 1,999 (at most 90 per world), so hunters are having an effect.
3. **They still hunt:** a fixed chaser catches at most 1.0x what learned hunters do (`chase.py --evolved`).

I have no strong prediction. Stage 10c says the bar is hard: the best so far is 0.098.

**Validity:** the bar is already known to fail for `pp-last-*` and `pp-m100-*`. Test 2 is new, and is there because stage 10a's lasting worlds sat on the caps, where hunters barely matter.

Reported, not bars: both swaps with hand-written animals, hunter numbers against the cap, whether numbers cycle within single worlds, how often hunters and grazers stand still.

## Result (2026-10-31): FAIL

| 256 seeds, evolved | `lasting_ambush` (meal 25) | `lasting_ambush_m100` (meal 100) | bar |
|---|---|---|---|
| both alive at step 1,999 | 0.000, 0.000 | 0.000, 0.000 | at least 0.5: missed |
| grazers per world at step 1,999 | 96, 96 | 96, 96 | at most 90: missed |
| fixed chaser against learned hunters | 2.05x, 1.57x | 0.72x, 0.86x | at most 1.0: met only at 100 |

Walking costs and thickets did not make a lasting world, and this does no better than stage 10c's meal of 250 without them (0.066 and 0.098). Grazers sit at the cap in every run with 22 to 29 of 40 energy, so the walking cost shows up in their energy but not in their survival.

At a meal of 100 learned hunters catch 0.0155 and 0.0163 per hunter-step, about twice stage 10c's 0.0083 at the same meal, so thickets may help hunting. By the sum that would pay about 1.6 energy per step, yet they still die out. That catch rate comes from a different script and averages over living hunters, so survivors may flatter it. Why they die out at that rate isn't settled here.

Nothing here counts against the viewer's idea as stage 8b tested it. It answered a different question there and passed. It just doesn't solve this one.

