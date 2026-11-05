# Stage 16c: gentle lessons on top of instincts

Date: 2026-11-02. Written before any of these runs.

## Why

Stage 16b: a child born with its parent's instincts and then given lessons at lr 1.0 hunted at near random level, because lessons that strong overwrite instincts (see 16b's diagnosis). lr 1.0 was chosen for blank brains. Here the same arms run with lessons ten times gentler, lr 0.1, which in stage 16's pilot barely moved a blank brain.

## World, arms and test

As stage 16b, with only `--lr 0.1`: genes + watch and genes + teach (cost 0.05), seeds 0 and 1, 1,000 updates. The test is against `pp-cul-genes-*` grazers, culture live, 64 worlds of 500 steps. Diagnostics as in 16b (`--births`, `--frozen`), reported.

## Success test (can fail)

1. **Gentle lessons add to instincts:** each arm at least 1.10x genes (.0365, .0344), on both seeds.
2. **Teach against watch:** 1.10x either way, or a tie.
3. **It lasts in its own world:** at least 0.5.

## Prediction

Bar 1 fails, at about 0.9x to 1.0x genes. Gentle lessons do little harm and little good, and the hiding effect still blunts selection a little. Bar 2: a tie. Bar 3 passes. If bar 1 passes, gentle teaching on top of instincts is the first culture result here that helps.

## Result (2026-11-02): FAIL on bar 1, a tie on bar 2, bar 3 passes. Gentle lessons do no harm to instincts, and no good

Against `pp-cul-genes-*` grazers, catches per hunter-step, seeds 0 and 1 (genes .0365, .0344):

| hunters, tested as | genes + watch | genes + teach |
|---|---|---|
| culture live (the test) | .0326, .0309 (0.89x, 0.90x) | .0344, .0332 (0.94x, 0.97x) |
| born brains only (`--births`) | .0367, .0362 | .0367, .0365 |
| adults as they stand (`--frozen`) | .0356, .0345 | .0362, .0350 |

1. **Gentle lessons add to instincts: fail.** 0.89x to 0.97x (bar 1.10x).
2. **Teach against watch: a tie.** 1.06x, 1.07x.
3. **It lasts in its own world: pass.** Watch .922, .828; teach .938, .938.

**Diagnosis:** at a tenth of the strength, the inborn brains are as good as the genes arm's (1.00x to 1.06x). So 16b's "hiding effect" came from lessons strong enough to blur what selection could see, and it goes away here. Adults after their lessons are about level with genes. What costs a little is newborns taking lessons during the test. Lessons at this strength neither help nor hurt much; teaching is a little better than watching, and not by enough to count.

**My prediction:** right on all three: 0.9x to 1.0x, a tie, and lasting.

**Where the culture stages leave it (16, 16b, 16c):** in this world, with these brains and one lesson a step, passing knowledge by lessons never beat passing it by genes. Lessons alone lose everything with the founders; strong lessons on top of instincts wreck them; gentle ones do nothing. Teaching and watching tied every time.

## Addendum, correction (2026-11-04, from the stage 20 reviews)

"Gentle ones do nothing" and "neither help nor hurt much" understate a measured cost. Live, with lessons running during the test, watch caught 0.89x and 0.90x of the genes arm, about 10% less, on both seeds; teach 0.94x and 0.97x. Stage 16c measured no gain from gentle lessons, and a loss of about 10% for watching. The diagnosis's "hiding effect" in 16b is also a hypothesis, not a shown cause: lessons changed both species, and so the selection both faced (Astra's review, `docs/reviews/2026-11-04-next-stage-astra.md`). The result above is left as written.
