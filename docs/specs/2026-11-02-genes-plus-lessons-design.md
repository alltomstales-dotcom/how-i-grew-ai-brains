# Stage 16b: born with instincts, then taught

Date: 2026-11-02. Written before any of these runs.

## Why

Stage 16: with no inherited wiring, lessons alone lost everything the founders knew within about 20 updates, and every culture arm hunted at random level against grazers with inherited brains. Real animals are born with instincts and then learn from parents. So here a child is born with its parent's inherited (birth) brain plus noise, exactly as in the genes arm, and then takes lessons from its parent during childhood. What it learns in life isn't passed on as wiring; its own children are born from its birth brain, not its taught one.

## World and settings

As stage 16: `tiny_stealth`, founders `pp-ts-0` and `pp-ts-1`, 64 worlds, 1,000 updates, seeds 0 and 1, lr 1.0, K 32, sigma 0.003, clip 1.0, teach cost 0.05. These are stage 16's pilot settings, unchanged, so the only difference is the inherited starting brain.

## Arms

- **genes:** stage 16's runs `pp-cul-genes-*`, reused as the control and as the grazers in the test.
- **genes + watch:** `--mode watch --innate`.
- **genes + teach:** `--mode teach --innate --teach-cost 0.05`.

## Test

Same as stage 16: hunters from each arm against `pp-cul-genes-*` grazers, frozen, same seed, culture live, 64 worlds of 500 steps. Random hunters at .0134 and .0132 are the floor, from stage 16.

## Success test (can fail)

On both seeds, catches per hunter-step:

1. **Lessons add to instincts:** genes + teach at least 1.10x genes, and genes + watch at least 1.10x genes, each its own question.
2. **The vote's question again:** genes + teach at least 1.10x genes + watch, or the reverse; otherwise a tie.
3. **It lasts:** both species alive at step 1,999 in at least 0.5 of worlds in each arm's own world.

## Prediction

Bar 1 fails for both. A child born with its parent's instincts already acts much like its parent, so lessons have little to add. At lr 1.0 they mostly shove a decent brain about, which could even cost a little (0.9x to 1.05x). Bar 2 is a tie again. Bar 3 passes. If bar 1 passes, culture does add something once instincts carry the basics, and that would be the more hopeful story of the two.

## Result (2026-11-02): FAIL on bar 1, a tie on bar 2, bar 3 passes. Lessons did real harm

Against `pp-cul-genes-*` grazers, culture live, catches per hunter-step, seeds 0 and 1 (genes .0365, .0344; random .0134, .0132):

| hunters | culture live | against genes |
|---|---|---|
| genes + watch | .0160, .0147 | 0.44x, 0.43x |
| genes + teach | .0142, .0164 | 0.39x, 0.48x |

1. **Lessons add to instincts: fail, badly.** Both arms land near random.
2. **Teach against watch: a tie.** 0.89x, 1.12x.
3. **It lasts in its own world:** genes + watch .688, .750; genes + teach .578, .703. Pass.

**Where the skill went. These three checks were added after the result, so they're diagnosis, not bars:**

| same hunters, tested as | genes + watch | genes + teach |
|---|---|---|
| the brain they were born with, no lessons (`--births`) | .0288, .0273 | .0271, .0279 |
| adults as they stand, no more lessons (`--frozen`) | .0236, .0209 | .0242, .0227 |
| culture live: newborns keep taking lessons | .0160, .0147 | .0142, .0164 |

Each step costs skill.
- **The inborn brains:** these evolved worse than the genes arm's, at about 0.77x. Selection only ever saw animals after their lessons, so it couldn't sort good instincts from bad as sharply. I believe biologists call this a "hiding effect" of learning; I'm citing that from memory, unchecked.
- **Lessons on top:** they cost about another fifth.
- **Newborns' lessons during the test:** these pull hunters the rest of the way down to about random.

**My prediction:** bar 1 failing was right, but I said 0.9x to 1.05x and it was about 0.4x, so the harm was far bigger than I expected. The tie and bar 3 were right.

**Limit:** lr 1.0 was set by stage 16's pilot for blank brains, where a weak lesson does nothing. For a child that already has decent instincts, it's a sledgehammer: one lesson can move a brain further than its instincts are worth. So this shows lessons this strong do harm, not that teaching can't help.

**Next:** the same two arms with lessons ten times gentler (lr 0.1), stage 16c.
