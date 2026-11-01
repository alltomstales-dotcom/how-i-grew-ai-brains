# Stage 14a: individual brains that descendants inherit

Date: 2026-11-01. Written before any of these runs. Tom's idea: train individual brains, and let descendants pick up what their predecessors learned.

## Why

Today there's one brain per species, trained on every animal's experience and shared by all of them. No animal has weights of its own, nothing it learns is its own, and a child inherits only genes. So this tests a different kind of inheritance. Each animal gets its own small network. Three versions are the classic comparison:

- **Darwinian:** a child gets its parent's birth brain plus mutation, and nothing is learned in life.
- **Baldwinian:** animals learn in life, and a child gets its parent's birth brain plus mutation, so learning helps only by keeping good brains alive.
- **Lamarckian (Tom's idea):** animals learn in life, and a child gets its parent's brain as it stands at that birth, plus mutation, so knowledge can build up down a family line.

This isn't recursive self-improvement: the way an animal learns doesn't change, only what it knows.

## World

`tiny_stealth` is `lasting_stealth`, with camouflage set by rule, meals of 25 and no size, and a network of about 4,700 weights in place of 115,750, so that 8,192 animals can each hold one. `tiny_winter` adds tundra's winter.

Founders aren't blank: each gets the same brain, trained with the existing shared trainer for 150 updates on that world and seed, plus a little noise. That shared brain is also the reference. A brain one life old could never compete with one that's had millions of steps, so the question is what happens from there.

Each animal learns online, every step, from its own experience, with a one-step actor-critic and plain gradient descent (the rate is set below). Its reward is its change in energy / 25, or +1 when it breeds, or -1 when it dies. That reward is for learners only: breeding is what selects in every arm. Mutation is Gaussian noise on every weight (also set below). When a world ends, its new animals draw brains from living animals in any world. All worlds use 64 worlds at once, 150 updates of 32 steps.

A short pilot on seed 99 may change the learning rate and noise, only to check that learning moves weights without blowing up. Any change gets recorded here before the real runs.

## What the pilot changed, before any real run

Seed 99, Lamarckian only, 12 to 15 updates, on `tiny_stealth`, looking only at speed and stability. No arm was compared.

- **Learning rate 0.001 and noise 0.01:** a hunter's weights moved about 0.0014 from learning (root mean square per weight), against 0.01 from mutation at each birth. Mutation would drown out anything learned, which isn't a fair test of Tom's idea.
- **Learning rate 0.005, noise 0.003, no clipping:** grazers' weights blew up, to about 10^13, and hunters did better only because grazers were broken. At 0.02 weights went to NaN.
- **So every animal's gradient is now clipped to a norm of 1 each step.** With it, 0.005 moves hunters about 0.0006 and 0.02 about 0.0018, with no blow-up.

**Real runs use a learning rate of 0.02, noise of 0.003 at birth and for founders, and a clip of 1.** Learning and mutation then move weights by similar amounts. These runs take about 5 to 10 minutes each.

## Runs

Seeds 0 and 1, on each machine in turn: the founders, then all three arms, on both worlds.

## Success test (can fail)

`scripts.eval_indiv`, 64 worlds, 2,000 steps, with no restarts, each arm keeping its own rules throughout.

1. **The small brain works:** the shared `tiny_stealth` brain has both species alive at step 1,999 in at least 0.5 of worlds, on both seeds. If not, the rest can't be read.
2. **Learning in life helps:** on `tiny_stealth`, Baldwinian hunters catch at least 1.10x what Darwinian hunters catch per hunter-step, on both seeds.
3. **Inheriting what was learned helps:** on `tiny_stealth`, Lamarckian hunters catch at least 1.10x what Baldwinian hunters catch per hunter-step, on both seeds.

The stage passes on bars 1 to 3.

Prediction: bar 2 fails, and so does bar 3. A hunter eats a dozen meals at most, and one life is a small sample against millions of steps. So I expect Darwinian selection among mutants to do about as well as learning. I hold that loosely, and I'd like to be wrong.

Reported, not bars: both species alive at step 1,999 in every arm, the same comparisons on `tiny_winter`, where inherited lessons can go out of date, each arm against the shared brain and a fixed chaser, how far weights differ across hunters, and how far a hunter's weights move from its birth brain.

Bar 1 is already met: the shared `tiny_stealth` brain lasts in 0.973 and 0.637 of worlds, and `tiny_winter`'s in 0.961 and 0.895. Learned shared hunters catch 0.0419 and 0.0391 per hunter-step on `tiny_stealth`, 0.81x and 0.76x for the chaser.

## Result (2026-11-01): FAIL on bars 2 and 3, and Lamarckian inheritance does harm

All twelve runs reached update 150. Evaluation is 64 worlds of 2,000 steps with no restarts, with each arm keeping its own rules.

| | catches per hunter-step | both alive at step 1,999 | hunters at the end | chaser against these hunters |
|---|---|---|---|---|
| `tiny_stealth` Darwinian | 0.0391, 0.0381 | 0.938, 0.688 | 29.0, 19.6 | 0.87x, 0.80x |
| `tiny_stealth` Baldwinian | 0.0383, 0.0374 | 0.953, 1.000 | 30.4, 31.3 | 0.94x, 0.79x |
| `tiny_stealth` Lamarckian | 0.0375, 0.0351 | 0.906, 0.750 | 27.0, 21.6 | 1.09x, 0.91x |
| `tiny_winter` Darwinian | 0.0424, 0.0447 | 0.812, 0.938 | 21.2, 27.4 | 0.79x, 0.78x |
| `tiny_winter` Baldwinian | 0.0429, 0.0435 | 0.953, 0.922 | 29.6, 28.6 | 0.80x, 0.81x |
| `tiny_winter` Lamarckian | 0.0384, 0.0403 | **0.516, 0.547** | **11.0, 13.5** | 0.97x, 1.00x |

1. **The small brain works: met.**
2. **Learning in life helps: missed.** Baldwinian against Darwinian, 0.98x on both seeds.
3. **Inheriting what was learned helps: missed.** Lamarckian against Baldwinian, 0.98x and 0.94x.

My prediction was right on both, but I expected the two to come out about equal, and they didn't. **Lamarckian inheritance does measurable harm,** most of all in winter, where its worlds last in about half the cases against more than nine in ten, and hunters number about half as many.

**Why isn't settled, and this isn't a verdict on the idea.** Baldwinian barely differs from no learning at all, so this learning rule, online and one step at a time from a single noisy life, doesn't appear to improve a brain. If so, Lamarckian children inherit accumulated noise, which is what I'd expect to do harm whatever the merits of inheriting real knowledge. That the harm is largest in winter also fits inherited lessons going out of date, and these runs can't separate the two. All of this applies to this rule, this brain and these 150 updates. A better learner could change the answer.

**Individual brains do hold up.** With 4,702 weights each and no shared training, Darwinian and Baldwinian worlds last, and hunters keep ahead of the chaser. Against the shared brain's 0.0419 and 0.0391, though, they don't improve on where they started.

