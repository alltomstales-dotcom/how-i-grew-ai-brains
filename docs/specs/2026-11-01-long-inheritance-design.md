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
