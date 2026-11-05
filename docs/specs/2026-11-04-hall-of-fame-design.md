# Stage 19: was the problem the opponent? Evolution alone against grazers that stay trained

Date: 2026-11-04 (written 2026-11-03, the evening stage 17 was scored). Written before any of these runs.

## Why

Stage 17 evolved hunters from random brains with no training. Against stage 16's genes-arm grazers (inherited trained brains) they caught only 1.19x and 1.23x what random hunters catch. In their own worlds they looked as good as trained hunters, because their grazers had co-evolved from random too and stayed weak. Both sides got worse together, or rather never got good.

The standard remedy in co-evolution is to keep learners facing strong, fixed opponents: a hall of fame (Rosin and Belew, 1997, cited from memory; it needs opening before it goes in a video). So the question here is narrow: **if the hunters had faced competent grazers the whole time, would mutation and selection alone have built a real hunter?** If yes, stage 17's failure was the opponent. If no, it was the search.

## World and arms

Everything as stage 17's full arm: `tiny_stealth`, 4,702 weights a brain, 64 worlds, 1,000 updates, seeds 0 and 1, `darwin` (no learning in life), **no founders** (hunters start from random brains, the same random brains as stage 17's full arm, since the seed is the same), mutation on all weights, sigma 0.03. Hunters live, breed and mutate normally in every world.

The one change: in a fraction of worlds, grazers are **anchored**. Every grazer that appears in an anchored world (the first animals, every newborn, and every animal of a world that restarts) takes a brain drawn at random from a frozen pool, not from its parent, and unmutated. The pool is every living grazer in `runs/pp-cul-genes-<seed>/checkpoints/pop_latest.npz` (stage 16's genes arm: 3,541 brains on seed 0, 5,064 on seed 1). So those grazers stay competent and their brains never evolve.

- **anchored 100%:** all 64 worlds anchored.
- **anchored 50%:** worlds 0 to 31 anchored, 32 to 63 not, fixed by index.
- **control:** stage 17's `pp-evo-full-0` and `-1` (0% anchored), already run and reused.

Choices, stated plainly:

- **Body genes evolve as usual in every world**, from the usual founder bodies, as in stage 17. Only brains are anchored. So anchored grazers carry trained brains in bodies that start untrained (the test grazers' bodies have evolved: for instance breed .13 rather than .5, sight .64 and .77 rather than .5). Keeping bodies as stage 17 keeps the one change to brains.
- **Unanchored grazers never take a pool brain.** When an unanchored world restarts, its grazers draw brains only from living grazers in unanchored worlds. Hunters still draw from living hunters in any world, as before, so in the 50% arm hunters mix across both kinds of world through restarts.
- With anchoring off the code runs exactly as before (a test checks that an unanchored world runs identically).

## Settings fixed by a pilot

A pilot on seed 2, 150 updates, both arms, pool `pp-cul-genes-0` (there is no seed-2 genes run), checks only two things: that worlds don't all die out at once, and that grazers in anchored worlds really carry pool brains (logged as `anch_prey_from_pool`, by fingerprint, and `free_prey_from_pool`, which should be 0). **No setting is changed because hunters starve.** Hunters starving against competent prey is a possible result, not a bug. Sigma stays 0.03 unless the pilot shows a mechanical failure. The pilot's numbers go in an addendum here before the full runs.

## Test

Exactly as stage 17: `scripts.cross_indiv`, hunters from each arm frozen (children copy the brain), against `runs/pp-cul-genes-<seed>` grazers, frozen, same seed, 64 worlds of 500 steps. Random hunters catch .0134 and .0132 against those grazers, stage 17's full arm .0159 and .0163, trained genes hunters .0365 and .0344.

## Success test (can fail)

On both seeds, catches per hunter-step:

1. **Anchoring builds real skill:** anchored 100% catches at least 1.5x random (about .020).
2. **It beats unanchored evolution:** anchored 100% catches at least 1.25x what stage 17's full arm caught (about .020).
3. **Half is enough:** anchored 50% catches at least 1.25x stage 17's full arm.

The bars are the suggested ones, unchanged. Bars 1 and 2 are nearly the same number here (1.5 x .0134 = .020, 1.25 x .0161 = .020), which I keep: bar 2 is what the question is about, and bar 1 ties it to the series' usual floor.

Reported, not bars:

- **Hunters alive over training**, in anchored and unanchored worlds separately (`anch_pred_alive`, `free_pred_alive`), and catches over training. If hunters in anchored worlds die out all at once, I say so plainly.
- **Own world:** each arm's hunters against its own grazers, `--steps 2000`, `--out own.json`, for both species alive at step 1,999.
- **A held-out check.** The anchored hunters trained against the very pool they're tested on, which stage 17's hunters never saw. So I also test each arm's hunters, and stage 17's full arm, against the *other* seed's genes grazers (`pp-cul-genes-1` for seed 0 and the reverse), which neither trained against, with random hunters as the floor there (.0132 against `-1`, .0134 against `-0`). If the bars pass but this check falls back to stage 17's level, I'll report the pass as skill against these particular grazers, not general hunting skill.

## Prediction

Bars 1 and 2 pass, narrowly: anchored 100% reaches about .022 (about 1.65x random, 1.35x stage 17's full arm). In stage 17 the hunters did improve against what they faced; what they faced was weak. Against trained grazers, selection acts on what the test measures. Bar 3 fails: anchored 50% reaches about .018 (about 1.1x full). Unanchored worlds hold about four times as many hunters and rarely restart, while anchored worlds restart often and redraw their hunters mostly from the easy worlds, so the easy worlds swamp the selection.

Hunters in anchored worlds: random hunters lose energy against trained grazers (about .0134 x 25 = 0.34 energy a step from meals, against at least 0.5 a step in upkeep), so I expect anchored worlds to restart often and hold about 6 to 10 hunters on average, against about 30 in stage 17, but not to die out all at once. On the held-out check I expect anchored 100% to keep most of its edge (about 1.4x random), since trained grazers from the two seeds probably flee in similar ways. Bar 1 is my least confident call; a real chance is that hunters starve too fast for selection to see anything, and anchored 100% ends near stage 17's 1.2x.

Nothing here reaches trained hunters (.0365, .0344).

## Addendum, pilot (2026-11-03, before any full run)

Seed 2, 150 updates, sigma 0.03, pool `pp-cul-genes-0` (3,541 brains), per world, first 15 updates → last 15. Stage 17's full-arm pilot at sigma 0.03 for comparison. Catches are per world-step in the arm's own worlds, so they say nothing yet about skill.

| arm | worlds | hunters | catches | grazers with a pool brain |
|---|---|---|---|---|
| anchored 100% | all | 7.9 → 7.1 | .12 → .13 | 1.000 at every check |
| anchored 50% | anchored | 7.8 → 7.3 | .12 → .13 | 1.000 at every check |
| anchored 50% | unanchored | 7.8 → 29.1 | .16 → .90 | 0.000 at every check |
| stage 17 full (pilot) | all | 8.4 → 29.6 | .18 → .95 | |

Both checks pass: worlds don't die out at once, and anchored grazers carry pool brains (every check, update 1 to 150) while unanchored grazers never do. In anchored worlds hunters hover around 7 a world, about what a world restarts with (8): they starve and worlds restart, but not all at once, and nothing improved in 150 updates. Unanchored worlds in the 50% arm look just like stage 17. **Fixed for the full runs: nothing changes. Sigma 0.03, as stage 17.** Already visible, not a bar: in 150 updates anchored hunters' catches haven't moved.

## Result (2026-11-03): FAIL on all three bars. Trained opponents helped, but only to about 1.5x random, short of the bars and far from trained hunters

Against stage 16's genes-arm grazers (the same seed, which is also the anchored arms' training pool), 64 worlds of 500 steps, catches per hunter-step, seeds 0 and 1 (random .0134, .0132; stage 17's full arm .0159, .0163; trained genes hunters .0365, .0344):

| hunters | catches | against random | against stage 17 full | against trained genes |
|---|---|---|---|---|
| anchored 100% | .0200, .0193 | 1.495x, 1.466x | 1.26x, 1.19x | 0.55x, 0.56x |
| anchored 50% | .0176, .0166 | 1.31x, 1.26x | 1.11x, 1.02x | 0.48x, 0.48x |
| stage 17 full (control) | .0159, .0163 | 1.19x, 1.23x | | 0.44x, 0.47x |

1. **Anchoring builds real skill: fail, by a hair.** Anchored 100% reached 1.495x and 1.466x random (bar 1.5x). Seed 0 missed by .0001 catches per hunter-step; it's still a miss.
2. **It beats unanchored evolution: fail.** 1.26x on seed 0 passes, 1.19x on seed 1 misses (bar 1.25x on both).
3. **Half is enough: fail.** Anchored 50% got 1.11x and 1.02x of stage 17's full arm.

**Held-out check (reported, not a bar),** against the *other* seed's genes grazers, which no arm trained against (random .0132 against `-1`, .0134 against `-0`):

| hunters | catches | against random |
|---|---|---|
| anchored 100% | .0153, .0177 | 1.16x, 1.32x |
| anchored 50% | .0160, .0127 | 1.21x, 0.95x |
| stage 17 full | .0123, .0133 | 0.93x, 0.99x |

So part of the edge was specific to the grazers trained against, and part of it carried: anchored 100% still beat stage 17's hunters on unseen trained grazers on both seeds (1.24x and 1.33x of them), where stage 17's hunters were at random level.

**Hunters alive over training.** They didn't die out at once. In anchored worlds hunters averaged 7.2 to 7.8 a world over training on every run, the lowest update averaging 4.6 to 5.4, about what a world restarts with (8): hunters run down and worlds restart, all through training. Unanchored worlds in the 50% arm held about 29.6, like stage 17 (29.9). Hunter generations by update 1,000: 286 and 288 (anchored 100%), 297 and 339 (50%), against 313 and 310 in stage 17. Grazers in anchored worlds carried pool brains at all 68 checks of every run, and grazers in unanchored worlds never did.

**Catches over training, per hunter-step in anchored worlds:** anchored 100% went from .0140 and .0132 (updates 1 to 25) to .0183 and .0181 (101 to 200) and .0191 and .0190 (901 to 1,000). Almost all the gain came in the first 150 or so updates, then it went flat for 800. The 50% arm's anchored worlds went the same way (.0148 → .0179, .0135 → .0178). Its unanchored worlds did what stage 17 did (about .033), and since they hold about four in five of the arm's hunters, the tested population is mostly those, which is why the 50% arm lands near the control.

**Own world, 2,000 steps, no restarts:** anchored 100% hunters die out in every world (both alive 0.000, 0.000; .0190 and .0192 catches per hunter-step). Anchored 50%: both alive 0.109 and 0.141. Stage 17's full arm: 0.984 and 0.984. Hunters facing the pool's grazers can't keep a world going without restarts.

**Correction to the pilot addendum:** it said anchored hunters' catches "haven't moved" in 150 updates. That was per world-step (.12 → .13). Per hunter-step they had risen, from about .015 to .018, which is the early gain seen in the full runs. The addendum is left as written.

**My prediction:**
- Bars 1 and 2 pass, at about .022 (1.65x random): wrong. They reached .0200 and .0193 (1.50x, 1.47x), just under both bars.
- Bar 3 fails, at about .018: right (.0176, .0166).
- Hunters in anchored worlds at about 6 to 10 a world, not dying out at once: right (7.2 to 7.8).
- Held-out about 1.4x random: too high (1.16x, 1.32x).
- Nothing reaches trained hunters: right (0.55x, 0.56x).

**What it adds to the series:** the opponent was part of the problem, not most of it. Facing competent grazers the whole time more than doubled what evolution alone gained over random (from 19 to 23% to 47 to 50%), and some of that skill held against trained grazers it had never met, where stage 17's hunters held none. But the gain stopped after about 150 updates and then stayed flat for 800, at about half of what trained hunters catch. With a frozen opponent the target didn't move, so the plateau is the search's limit here, not a Red Queen: mutation and selection at sigma 0.03, with about 7 hunters a world that keep running down, found a first step and no second one. Mixing half easy worlds in threw most of the gain away, because the easy worlds supplied most of the hunters.

**Limits:** two seeds, one mutation size (0.03, from stage 17), 1,000 updates, one brain size. The anchored grazers had trained brains in untrained bodies (body genes evolved from the founders' values), so they may be weaker than the test grazers, which have both. The main test grazers are the anchored arms' training pool, which flatters the anchored arms; the held-out check is the fairer number and it's lower. A hall of fame here was one fixed pool, not a growing archive of past opponents as Rosin and Belew describe (cited from memory). Bar 1 was missed by a margin smaller than seed-to-seed noise, so "about 1.5x random" is the honest summary, not "nothing".

## Addendum, correction (2026-11-04, from the stage 20 reviews)

"With a frozen opponent the target didn't move" is false. Only grazer **brains** were anchored; grazer **bodies** evolved all run (body genes, from the founders, under selection), and hunter bodies evolved too. So the opponent did move, through bodies, and a gain or plateau can't be credited to brains alone. Second, "the plateau is the search's limit" ignored a competing reading: at about .019 catches per hunter-step a hunter earns .019 x 25 = 0.48 energy a step against upkeep of 0.5 or more, so one at 1.5x random starves on about the same clock as one at 1.0x, and selection may not have seen better brains at all. "The opponent was part of the problem, not most of it" isn't a causal split either, since stronger opponents also changed hunter numbers, breeding and restarts. Stage 20 (`2026-11-04-visible-selection-design.md`) freezes bodies and tests the selection reading. The result above is left as written.
