# Stage 14d: did Lamarckian hunters get better, or did their grazers get easier?

Date: 2026-11-01. Written before any of these runs.

## Why

Stage 14c: Lamarckian hunters' catches rise over 1,000 updates, and no other arm's do. But a fixed chaser also catches more against Lamarckian grazers, by 3% and 31%, so the rise may come from worse grazers. Catches in each arm were measured against that arm's own grazers, so those numbers can't tell these apart. The fair test puts hunters from every arm against the same grazers.

## Test

No training. `scripts.cross_indiv`, on `tiny_stealth`, seeds 0 and 1, within a seed. Hunters come with their own genes and brains from one run, and grazers with their own genes and brains from another. 64 worlds of 500 steps, no restarts. Brains are frozen in these worlds: no learning and no mutation, so this measures what each population ended up with.

Hunters: Darwinian, Baldwinian, Lamarckian (from stage 14c), and the founders' shared brain copied to every hunter.
Grazers: Darwinian and Lamarckian, from stage 14c.

## Success test (can fail)

Catches per hunter-step against Darwinian grazers, on both seeds:

1. **Lamarckian hunters beat Darwinian hunters:** at least 1.10x.
2. **Lamarckian hunters beat their founders:** at least 1.10x.

Grazers, reported as a third question with its own bar, on both seeds:

3. **Lamarckian grazers are easier:** each kind of hunter catches at least 1.10x as many of them as of Darwinian grazers, on average across the hunters above.

Prediction: bars 1 and 2 fail, and bar 3 passes. So I expect the rise to be mostly grazers getting easier. If bars 1 and 2 pass, that's real knowledge building up down family lines in hunters, and it's Tom's idea working.

## Result (2026-11-01): FAIL on all three bars, and Lamarckian hunters got worse

Catches per hunter-step, seeds 0 and 1, brains frozen:

| hunters | against Darwinian grazers | against Lamarckian grazers | ratio |
|---|---|---|---|
| Founders' shared brain | 0.0402, 0.0363 | 0.0440, 0.0422 | 1.09x, 1.16x |
| Darwinian | 0.0401, 0.0360 | 0.0426, 0.0425 | 1.06x, 1.18x |
| Baldwinian | 0.0374, 0.0343 | 0.0401, 0.0423 | 1.07x, 1.23x |
| Lamarckian | 0.0369, 0.0307 | 0.0416, 0.0391 | 1.13x, 1.27x |

1. **Lamarckian hunters beat Darwinian hunters: missed.** 0.92x and 0.85x, so they're worse.
2. **Lamarckian hunters beat their founders: missed.** 0.92x and 0.85x, so they're worse.
3. **Lamarckian grazers are easier: narrowly missed on one seed.** 1.09x and 1.21x on average, against a bar of 1.10x. The direction holds for every kind of hunter on both seeds.

My prediction held on bars 1 and 2, and bar 3 misses by 0.01 on seed 0, so it stays a miss.

**So stage 14c's rise wasn't knowledge building up.** Lamarckian hunters catch less than Darwinian ones against the same grazers, and against their own Lamarckian grazers they still catch less than Darwinian hunters and the founders. What rose was catches against grazers that had got easier. Both species seem to decline together, which looks like success if each is only measured against the other.

**Nothing has improved on the founders.** Darwinian hunters match them, at 0.0401 against 0.0402 and 0.0360 against 0.0363, after about 400 generations. Selection held the skill and didn't add to it. Hunters with frozen founders' brains also do as well as anyone against these grazers.

**Against Darwinian grazers, Lamarckian hunters also fade:** 21.4 and 9.1 at the end against 30.5 and 24.7.

**What this doesn't show.** Brains here are frozen, so Lamarckian animals lose any within-life learning they'd normally do. It's this learning rule, this reward, this small brain and these runs, and not Lamarckian inheritance in general. A likely reason, not tested: a grazer's reward is mostly energy, with -1 on death, so learning may pull grazers toward eating and away from fleeing, and inheritance keeps that drift. A better rule could give a different answer. Stages 15a to 17 stand, and culture (stage 16) now has a clear test to pass: this one, with the same grazers.

**Lesson for measuring:** in a world with two learning species, a rise in catches says nothing by itself. Compare hunters against the same opponents.

