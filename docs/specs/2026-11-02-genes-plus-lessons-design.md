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
