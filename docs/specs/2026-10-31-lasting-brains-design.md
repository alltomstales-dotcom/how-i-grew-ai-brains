# Stage 10b: do trained brains last too?

Date: 2026-10-31. Written before any of these runs.

## Why

Stage 10a found worlds that last for hand-written animals, with plants spreading at `seed_rate` 0.2 instead of 0.03. Trained brains have only known worlds that die within about 300 steps, and in training a world restarts when a species dies out. Probes on those brains, with evolved plant pools, still died out at `seed_rate` 0.25. So it's open whether learning, evolution and a lasting world work together.

## World

`lasting`: `EvoConfig` with `seed_rate` 0.2. Nothing else changes. `pp-last-{0,1}`, seeds 0 and 1, 150 updates each.

## Success test (can fail)

Both must hold on both seeds:

1. **It lasts, with evolved brains and genes:** `python -m scripts.persist --run-dir runs/NAME --evolved --steps 2000`, 256 fixed seeds, both species alive at step 1,999 in at least 0.5 of worlds. That is four times the longest episode they trained on.
2. **Hunters still hunt:** `python -m scripts.chase --run-dir runs/NAME --evolved`, learned hunters catch at least as much as a fixed chaser against trained prey (chaser ratio at most 1.0).

**Validity check, run before training:** test 1 on `pp-last-u`, one update and so effectively untrained. If that passes too, test 1 says nothing about brains, and the spec gets changed before training.

It didn't pass: 0.000 at step 1,999, with hunters gone by step 600 (0.73 per world at step 200, in 0.316 of worlds) and grazers at the cap of 96 with 39 of 40 energy, among plants at 0.90 of the map. So test 1 stands.

Reported, not bars: hunter and grazer numbers against the slot caps (96 and 32), since stage 10a's worlds sat on them; plant cover; whether numbers cycle; the plant-spread gene over training, since evolved plant pools may undo the world's fast spread.

## Result (2026-10-31): FAIL on the first bar, on both seeds

| 256 seeds, evolved | seed 0 | seed 1 | bar |
|---|---|---|---|
| both species alive at step 1,999 | 0.000 | 0.000 | at least 0.5: missed |
| fixed chaser against learned hunters | 0.75x | 0.65x | at most 1.0: met |

The world lasts and the hunters don't. Both species are together in 0.062 and 0.059 of worlds at step 200 and none from step 600. Grazers sit at the cap of 96 with about 29 of 40 energy, on a map 0.91 to 0.92 plants. Learned hunters are still real hunters, catching 0.0062 per hunter-step, about 0.16 energy per step at 25 a meal.

Hand-written animals lasted in this world in stage 10a. The likely difference is that trained grazers flee, and hand-written ones walk to the nearest plant whatever is near. That's my reading, and I haven't tested it directly.

Also seen, in training metrics: plants' spread gene rose from 0.50 to 0.89 by update 31 and stayed; grazers' breeding gene fell to about 0.08; and hunters' rose from 0.50 to 0.81 and 0.78, not explained here.

So this world has plenty of grazers and too little income for hunters. That makes meal size a fair test at last.
