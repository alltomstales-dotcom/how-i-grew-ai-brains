# Stage 10g: with room to move, what sets the numbers?

Date: 2026-10-31. Written before any of these runs.

## Why

Stage 10f gave the first trained brains in a world that lasts: both species alive at step 2,000 in 0.957 and 0.961 of worlds. But grazers sit on their slot cap of 96 and hunters near their 32, so a limit in the code may be setting the numbers. An ecosystem would set its own.

## World

`lasting_stealth` with three times the slots: `max_prey` 288, `max_pred` 96, on the same 48 x 48 map. Called `lasting_room`. `pp-room-{0,1}`, seeds 0 and 1, 300 updates, `--num-envs 256` so memory stays about the same with three times the animals. That is a third as many worlds per update, so it runs for twice as many updates.

## Success test (can fail)

Scored at step 1,999 of `python -m scripts.persist --run-dir runs/NAME --evolved --steps 2000`, 256 fixed seeds. All on both seeds:

1. **It lasts:** both species alive in at least 0.5 of worlds.
2. **Grazers set their own number:** at most 259 per world, 0.9 of the cap.
3. **Hunters set their own number:** at most 86 per world, 0.9 of the cap.
4. **They still hunt:** a fixed chaser catches at most 1.0x what learned hunters do (`chase.py --evolved`).

Prediction: 1 and 4 hold, grazers hit the new cap since plants are plentiful (bar 2 fails), and hunters end below theirs. I hold this loosely, having been wrong on the last two.

Reported, not bars: plant cover, and whether numbers rise and fall within single worlds. If bar 2 fails, plants are what to look at next.

## Result (2026-10-31): FAIL by the rule, on hunters at their cap, and grazers now set their own number

| 256 seeds, evolved | seed 0 | seed 1 | bar |
|---|---|---|---|
| both alive at step 1,999 | 0.984 | 0.996 | at least 0.5: met |
| grazers per world at step 1,999 | 114.0 | 111.8 | at most 259: met |
| hunters per world at step 1,999 | 94.50 | 95.62 | at most 86: missed |
| fixed chaser against learned hunters | 0.74x | 0.73x | at most 1.0: met |

My prediction had this backwards: I expected grazers to hit their new cap and hunters to stay under theirs.

With three times the slots, grazers settle at about 39% of their cap, with births and deaths about equal at 5 to 6 per step per world, and at about 9 of 40 energy. Plants hold at 0.44 of the map, down from 0.91. So hunters now hold grazers well under what the slots allow, and something other than a slot limit sets the grazer number. These numbers are nearly flat from step 1,000 to 1,999 in the averages, so no cycle shows at this level, though averaging across worlds would hide one in any single world.

Hunters still fill their slots, 94.5 and 95.6 of 96, so a cap still sets their number, and bar 3 fails. They outnumber a world's grazers almost one to one. Grazers' breeding gene is about 0.15, and hunters' sight 0.40, with speed 0.22 to 0.24.

