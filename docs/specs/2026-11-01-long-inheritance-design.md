# Stage 14c: does inherited knowledge build up, given time?

Date: 2026-11-01. Written before any of these runs.

## Why

Stage 14a's runs were about 50 generations. If what Tom's after is knowledge building up down a family line, 50 generations may be too short to show it, and the early harm may not be the final word. These runs take minutes, so there's no reason not to look much further.

## World

Stage 14a's arms, settings and founders, on `tiny_stealth` and `tiny_winter`, for 1,000 updates in place of 150, which is about 350 generations. In addition, `tiny_winter` Lamarckian with the heritable learning rate of stage 14b. Seeds 0 and 1, with one seed on each machine.

## Success test (can fail)

`scripts.eval_indiv`, 64 worlds, 2,000 steps, on `tiny_stealth`, on both seeds:

1. **Inherited knowledge builds up:** Lamarckian hunters catch at least 1.10x what Darwinian hunters catch per hunter-step.
2. **Past where they started:** Lamarckian hunters catch at least 1.10x what the shared founder brain catches in the same evaluation's terms: 0.0419 and 0.0391 in stage 14a's reference.

And on `tiny_winter`, on both seeds:

3. **A heritable rate rescues it, given time:** Lamarckian worlds with the gene last in at least 0.8 of cases.

Bars 1 and 2 test Tom's idea, and 3 is stage 14b's question with more time. The stage passes on 1 and 2.

The reference in bar 2 comes from `scripts.chase` with 256 worlds of 500 steps, not this evaluation, so it's rough, and I'm fixing it now rather than choosing after the results.

Prediction: bars 1 and 2 fail, since selection alone may do as well and this rule looks weak. Bar 3 passes with the gene near zero. Reported, not bars: every arm's catches over training, how far weights spread, Darwinian against the founder, and the gene's path.

## Result (2026-11-01): FAIL on all three bars, and Lamarckian is the only arm still rising

All fourteen runs reached update 1,000, about 320 to 490 generations. Evaluation is 64 worlds of 2,000 steps with no restarts.

| | catches per hunter-step | both alive | chaser catches against these grazers |
|---|---|---|---|
| `tiny_stealth` Darwinian | 0.0416, 0.0366 | 0.938, 0.703 | 0.0419, 0.0285 |
| `tiny_stealth` Baldwinian | 0.0378, 0.0368 | 1.000, 0.875 | 0.0369, 0.0328 |
| `tiny_stealth` Lamarckian | 0.0429, 0.0408 | 0.984, 1.000 | 0.0431, 0.0372 |
| `tiny_winter` Darwinian | 0.0421, 0.0415 | 0.906, 0.984 | 0.0313, 0.0321 |
| `tiny_winter` Baldwinian | 0.0415, 0.0412 | 0.969, 0.938 | 0.0307, 0.0338 |
| `tiny_winter` Lamarckian | 0.0426, 0.0432 | 0.938, 0.953 | 0.0385, 0.0393 |
| `tiny_winter` Lamarckian with the gene | 0.0409, 0.0375 | 0.766, 0.562 | 0.0356, 0.0330 |

1. **Inherited knowledge builds up: missed.** Lamarckian against Darwinian, 1.03x and 1.11x, and the bar needs 1.10x on both.
2. **Past where they started: missed.** Against the founders' 0.0419 and 0.0391, 1.02x and 1.04x.
3. **A heritable rate rescues it, given time: missed.** 0.766 and 0.562, with the gene at 0.207 and 0.217.

Per hunter in training, seed 0, per hundred updates:

| | start | ... | end |
|---|---|---|---|
| Darwinian | 0.0390 | flat | 0.0391 |
| Baldwinian | 0.0378 | falling slowly | 0.0368 |
| Lamarckian | 0.0372 | rising | 0.0420 |

**My prediction was half wrong.** Bars 1 and 2 fail as I said, but I expected no sign of build-up, and Lamarckian is the only arm that rises, on both seeds. The winter harm of stage 14a was also temporary: 0.516 and 0.547 at about 50 generations, 0.938 and 0.953 now. And bar 3 goes the other way from what I predicted. With the rate fixed, winter worlds end at about 0.94, and letting evolution turn learning down leaves them at 0.766 and 0.562. Selection cuts learning, and on these numbers worlds do worse for it.

**A warning about that rise.** Hunters' catches depend on grazers too, and a fixed chaser also catches more against Lamarckian grazers: 0.0431 and 0.0372 against Darwinian's 0.0419 and 0.0285. So some or all of the rise may be grazers getting easier and not hunters getting better. Stage 14d tests that directly.

**Stage 14d's answer:** the rise was grazers getting easier. Against the same grazers, Lamarckian hunters catch less than Darwinian hunters and their founders, 0.92x and 0.85x. So "Lamarckian alone keeps rising" above shouldn't be read as hunters improving.

