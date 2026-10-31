# Stage 10e: do hunters built for the chase last?

Date: 2026-10-31. Written before any of these runs.

## Why

Stages 10b to 10d: trained grazers kill the hunters, whose meals and walking costs don't rescue them. Part of it may be plain geometry: at equal speed on an open map, a chaser catches a dodger only by its mistakes. Speed already exists as a gene, the chance of moving two cells, at an energy cost of speed squared per step. That cost falls on hunters as hard as on grazers, though a hunter's whole income depends on speed, and in stage 10b's runs hunters did not evolve to be fast. Real predators are often built for the sprint.

## World

`lasting` plus `pred_speed_cost` 0: speed costs hunters nothing, so evolution can make them fast. Grazers pay as before. `lasting_sprint` has meals of 25, `lasting_sprint_m100` has meals of 100 with `pred_energy_cap` 200. `pp-sp-{0,1}`, `pp-spm100-{0,1}`, seeds 0 and 1, 150 updates each.

## Success test (can fail)

For at least one of the two worlds, on both seeds, the same three bars as stage 10d:

1. **It lasts:** `python -m scripts.persist --run-dir runs/NAME --evolved --steps 2000`, 256 fixed seeds, both species alive at step 1,999 in at least 0.5 of worlds.
2. **Grazers aren't just stuck:** at most 90 grazers per world at step 1,999, so hunters are having an effect.
3. **They still hunt:** a fixed chaser catches at most 1.0x what learned hunters do (`chase.py --evolved`).

Prediction: hunters evolve to high speed and last in more worlds. The risk is that they then wipe out the grazers, which test 1 would catch.

Reported, not bars: the speed gene of hunters and grazers, numbers against the caps, whether numbers cycle in single worlds, and the swap with hand-written grazers and hunters.

## Result (2026-10-31): FAIL, and the prediction was wrong

| 256 seeds, evolved | `lasting_sprint` (meal 25) | `lasting_sprint_m100` (meal 100) | bar |
|---|---|---|---|
| both alive at step 1,999 | 0.000, 0.000 | 0.008, 0.008 | at least 0.5: missed |
| grazers per world at step 1,999 | 96, 96 | 96, 96 | at most 90: missed |
| fixed chaser against learned hunters | 0.78x, 0.75x | 0.63x, 0.63x | at most 1.0: met |

Hunters did evolve for speed: 0.82 and 0.78 at meal 25, against 0.18 and 0.16 when it costs them (stage 10b) and grazers' 0.31 and 0.28. It made little difference. Their sight stayed at 0.08 and 0.07 at meal 25, so they are fast and nearly blind, with 0.52 and 0.55 at meal 100. Both species share 0.223 and 0.172 of worlds at step 200 at meal 25, up from about 0.06, so hunters last a little longer and still go.

## Probe after the result: is it the hunters' genes?

Not a pre-set test. `scripts.persist --evolved --steps 2000 --force-pred`, which starts already-trained hunters with chosen genes, on `pp-last-0` (meal 25) and `pp-m100-0` (meal 100). Both species alive at step 200, against 0.062 and 0.066 unforced:

| forced | meal 25 | meal 100 |
|---|---|---|
| breed 0.1 | 0.000 | 0.012 |
| sight 0.7 | 0.039 | not run |
| speed 0.9 | 0.000 | not run |
| all three | 0.000 | 0.000 |

All reach 0.000 by step 600. So better genes don't rescue them, and speed and sight cost energy these hunters don't have. These brains trained with other bodies and read their own genes, so this is not a clean test of those genes, and stage 10e shows only speed evolving with them.

