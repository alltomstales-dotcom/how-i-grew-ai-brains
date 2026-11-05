# Stage 20: was the plateau the search, or a selection that couldn't see better brains?

Date: 2026-11-04. Written before any of these runs, the pilot included.

## Why

Stage 19's anchored hunters rose from random to about 1.5x random against trained grazers in about 150 updates, then stayed flat for 800. Its spec read that as "the search's limit here". Both reviews (`docs/reviews/2026-11-04-next-stage-fable.md`, `-astra.md`) say that doesn't follow, and Fable names a competing reason: at stage 19's catch rates a hunter at 1.5x random earns about .019 x 25 = 0.48 energy a step against upkeep of 0.5 or more, so it starves on the same clock as one at 1.0x. Worlds sat at the restart floor (about 7 hunters a world). If being better barely changes how long a hunter lives or how often it breeds, selection can't see better brains even if mutation offers them.

**Question: if a better hunter breeds and a random one doesn't, does evolution keep climbing past stage 19's plateau?** If yes, the plateau was selection. If no, it was the search (or this brain, or sigma).

This is Fable's main proposal (Astra's runner-up), with Astra's method fixes: bodies frozen, held-out opponents primary, evaluation noise measured, effect sizes reported beside pass or fail.

## World and arms

Everything as stage 19's anchored 100% arm: `tiny_stealth`, 4,702 weights a brain, 64 worlds, 1,000 updates, seeds 0 and 1, `darwin` (no learning in life), no founders (random brains, the same ones as stages 17 and 19 since the seeds are the same), mutation on all weights, sigma 0.03. Anchored worlds draw every grazer brain from the frozen pool `runs/pp-cul-genes-<seed>` (stage 16's genes arm), as in stage 19.

**New in every arm: bodies are frozen.** `--world-set gene_sigma=0 --world-set founder_noise=0 --world-set plant_sigma=0.05`. Every hunter and every grazer has exactly the founder body (speed .2, sight .5, breed .5 and so on) for the whole run, so any gain is brains, not bodies. Plants still mutate as before (`plant_sigma` was `gene_sigma`, .05); with `founder_noise` 0 their first generation starts uniform instead of slightly spread. This differs from stage 19, where grazer and hunter bodies evolved (the "target didn't move" error), so no stage 19 run is reused.

| arm | grazers | hunter upkeep | run |
|---|---|---|---|
| **A, control** | anchored, all 64 worlds | as usual (`pred_base_cost` .5) | `pp-vis-A-<seed>` |
| **B, visible selection** | anchored, all 64 worlds | lowered so break-even sits at random's catch rate | `pp-vis-B-<seed>` |
| **C, confound control** | unanchored, co-evolving brains (stage 17's full arm) | as B | `pp-vis-C-<seed>` |
| **D, C's control** | unanchored, co-evolving brains | as usual | `pp-vis-D-<seed>` |

D is A's unanchored equivalent: stage 17's full arm with frozen bodies. Stage 17's own runs can't serve, since their bodies evolved. C against D asks whether easier living lifts hunters with no competent opponent at all, which would mean B's gain (if any) is "easier living lets weak hunters breed", not visible selection against good grazers.

**One knob: `pred_base_cost`.** Hunter upkeep is base + speed cost x speed² + sight cost x (sight radius − 1) / 3; with frozen founder bodies every hunter pays the same. I lower the base and keep meals at 25, because raising the meal instead to about 60 would run into the hunters' energy cap of 60 (stage 12d's lesson), which would make it two knobs. The value is set by the pilot below, by this rule, fixed now:

> base = 25 x r0 − (U − 0.5), where r0 is random hunters' catches per hunter-step and U their measured upkeep per hunter-step at base .5, both in A-style worlds. Then a hunter catching at random's rate breaks even, one at 1.5x gains about 0.5 x 25 x r0 a step, and anything worse loses energy.

Worked from the config before measuring: U should be .5 + .04 + .333 = .873, and r0 near stage 19's early .014, which would put the base at about −.02, a total upkeep of about .35. A base below zero only offsets the fixed body costs; total upkeep stays positive. If the measured value lands somewhere else, the measured value is used.

At exact break-even a random hunter doesn't die quickly; it drifts, and every birth costs it 8 energy, so its line runs down slowly. Hunters better than random gain and breed; worse ones starve. That's the intended gradient.

## Pilot (before any full run)

Seed 2, 150 updates, pool `pp-cul-genes-0`, frozen bodies, as stage 19's pilot.

1. **A-style pilot** (`pp-vis-pilot-A`). Checks: every body stays the founder body (gene spreads 0 at every update), grazers in anchored worlds carry pool brains, worlds don't all die out at once.
2. **Measure r0 and U:** `cross_indiv --random`, hunters and grazers from the A-style pilot, in its own world (frozen bodies, pool grazers, no restarts), 64 worlds of 500 steps, on three evaluation seeds; r0 and U are their means. Also reported: the random-brain hunters' catch rate in the pilot's first 25 updates.
3. **B-style pilot** (`pp-vis-pilot-B`) at the base from the rule. A mechanical check only: worlds neither all die out nor stop changing. **The base is not changed because of how many hunters live or breed.** It changes only for a mechanical failure, and then I say so.
4. **Evaluation noise, measured once.** `cross_indiv` rerun on six evaluation seeds (the usual 12345, then 1 to 5), with stage 19's `pp-hof-a100-<s>` and stage 17's `pp-evo-full-<s>` hunters, against held-out grazers (`pp-cul-genes-<1−s>`), under this stage's test settings. Per training seed and evaluation seed I take the paired ratio a100 / full, like B / A. **s_eval** is the pooled standard deviation of the log of that ratio. Training-window noise for bar 2: **s_train** is the pooled standard deviation of the log of catches per hunter-step across stage 19 anchored 100%'s seven windows of 100 updates from 301 to 1,000, its flat part. These numbers go in an addendum before the full runs.

This measures evaluation noise (which worlds the test draws) and window noise, not run-to-run noise between independent evolutions, which two seeds can't measure. A big seed-to-seed difference is reported as that.

## Test

`scripts.cross_indiv`, each arm's hunters frozen (children copy the brain), 64 worlds of 500 steps, no restarts. The test world is the grazers' run's (`tiny_stealth`, usual upkeep) with `--world-set gene_sigma=0 --world-set plant_sigma=0.05`, so no body changes during the test either: hunters keep the founder body, grazers keep their own pool's bodies.

- **Primary opponent, held out:** the other seed's genes grazers, `pp-cul-genes-<1−seed>`, which no arm trained against. Every bar uses this.
- **Secondary:** the same seed's genes grazers, `pp-cul-genes-<seed>`, which are A's and B's training pool and so flatter them.
- **References in the same bodies against the same opponents:** random hunters (`--random`, founder bodies), and trained hunters, `pp-cul-genes-<seed>`'s hunter brains in founder bodies (`--bodies` from an arm's run); trained hunters in their own evolved bodies too, for comparison with earlier stages.

## Success test (can fail)

All three on held-out grazers or in training, on both seeds:

1. **Visible selection builds better brains:** B catches at least **1.25x** what A catches, per hunter-step.
2. **It keeps climbing:** in B's training, catches per hunter-step in updates 901 to 1,000 are at least **1.15x** those in updates 101 to 200 (stage 19's anchored arm went .0183 to .0191, 1.04x).
3. **It isn't just easier living:** C catches at most **1.10x** what D catches. This bar is one-sided: C well below D wouldn't explain a gain in B, and is reported, not failed.

**Noise rule, fixed now.** For a ratio R against a bar t on one seed: past the bar is a pass; short of the bar but within a factor e^(2s) of it is **no result**, not a fail; further short is a fail. s is s_eval for bars 1 and 3, and √2 x s_train for bar 2 (a ratio of two windows). A pass within e^(2s) of the bar is reported as "pass, inside noise". A bar passes when both seeds pass, fails when either seed fails, and is otherwise no result.

**Reported beside every ratio, not bars:** each arm against random and against trained hunters (both in founder bodies), on held-out and same-seed grazers; total catches; hunters alive and both species alive at step 500; and from training, hunters per world, hunter generations and catches per hunter-step by window of 100 updates, for every arm.

## Prediction

Bar 1 passes, narrowly: B reaches about 1.35x A on held-out grazers. Once a good hunter outlives and outbreeds a bad one, selection should keep finding the small gains that stage 19's hunters were offered and lost. Bar 2 passes at about 1.2x, but it's my least confident call: the plateau could still come back at a higher level. Bar 3 passes: C within about 1.05x of D, since with co-evolving grazers that stay weak, easier living mostly adds hunters, not skill. A on held-out grazers lands near stage 19's anchored arm there, about 1.2x to 1.3x random. Nothing reaches trained hunters.

## What each outcome would mean

- **1, 2 and 3 pass:** the plateau was selection, not the search. Making better brains pay lets mutation keep finding them. Next: does it go on climbing past 1,000 updates, and how near trained hunters can it get?
- **1 passes, 2 fails:** visible selection gave a one-time lift, then a new plateau. Selection was one limit and something else (sigma, brain size, the search) is the next. Next: Astra's sigma ladder under B's upkeep.
- **1 and 2 fail, B about A:** selection's visibility wasn't the limit; the search is. Next: the sigma ladder (Astra's proposal), or a different search.
- **2 passes, 1 fails:** B climbs against its own pool but the gain doesn't carry to unseen grazers. Lesson 15 in a new form: skill against these particular grazers, not hunting.
- **1 passes, 3 fails:** easier living lifts hunters even against weak, co-evolving grazers, so B's gain can't be credited to selection against competent prey. Report it as "lower upkeep helps evolution", not as this question's answer.
- **B well below A:** easy living let weak hunters persist and diluted selection. Then the upkeep lever was the wrong one and the meal-size or breeding threshold lever should be tried, with its own spec.
- **Any bar no result:** say so, with the ratio and band, and don't count it either way.

## Commands

Queue `~/autolearn-queue-20.sh` on the second machine, one seed per invocation, logs `~/autolearn-queue-20-<seed>.log`. Training:

```
EVAL_GB=5 bash deploy/remote.sh train_indiv --world tiny_stealth --mode darwin --run-id pp-vis-<arm>-<s> --seed <s> --updates 1000 --num-envs 64 --sigma 0.03 \
  --world-set gene_sigma=0 --world-set founder_noise=0 --world-set plant_sigma=0.05 [--world-set pred_base_cost=<base> for B, C] \
  [--anchor-frac 1.0 --anchor-pool runs/pp-cul-genes-<s> for A, B]
```

Test: `cross_indiv --hunters runs/pp-vis-<arm>-<s> --grazers runs/pp-cul-genes-<o> --world-set gene_sigma=0 --world-set plant_sigma=0.05 --out heldout_pp-vis-<arm>-<s>.json`, and the same against `pp-cul-genes-<s>` for the secondary.

## Addendum, pilot (2026-11-04, before any full run)

Seed 2, 150 updates, sigma 0.03, pool `pp-cul-genes-0`, frozen bodies.

**1. A-style pilot (`pp-vis-pilot-A`): all checks pass.** Every living animal kept the founder body (largest gene spread over the run .0002, which is rounding; hunter genes exactly .2, .5, .5, .5, .5, .5, .5). Grazers in anchored worlds carried pool brains at every check (11 of 11). Worlds didn't all die at once: hunters 7.8 a world in the first 15 updates, 7.3 in the last 15, lowest update 5.2, as in stage 19. Catches per hunter-step in training went from .0149 (updates 1 to 25) to .0216 (126 to 150).

**2. r0 and U, random hunters in the A-style pilot's own world, 64 worlds of 500 steps:**

| evaluation seed | catches per hunter-step | upkeep per hunter-step |
|---|---|---|
| 12345 | .0143 | .8733 |
| 1 | .0157 | .8733 |
| 2 | .0144 | .8733 |
| mean | **r0 = .01478** | **U = .8733** |

U is what the config predicts (.5 base + .04 speed + .333 sight), and every hunter pays it, since bodies are frozen. Break-even at the usual upkeep is .8733 / 25 = .0349 catches per hunter-step: random hunters catch 0.42 of what they need, and even trained genes hunters (.0365 against their own grazers in stage 19) barely clear it. That backs Fable's reading of stage 19.

**Fixed for B and C by the rule: `pred_base_cost` = 25 x .01478 − (.8733 − .5) = −.0038,** a total upkeep of .3695 a step, break-even at .0148 catches per hunter-step.

**3. B-style pilot (`pp-vis-pilot-B`) at −.0038: passes the mechanical check.** Worlds neither died nor froze: hunters 16.9 a world in the first 15 updates, 20.8 at 61 to 75, 29.2 at 136 to 150 (the cap is 32); births .14 to .18 and deaths .13 to .15 a step all through; grazers carried pool brains at every check. Random hunters in its world: .0165 catches per hunter-step, upkeep .3695 (break-even .0148), both species alive at step 500 in 0.95 of worlds. Catches per hunter-step in training .0162 (1 to 25), .0170 (51 to 75), .0206 (126 to 150). **The base is not changed.** Not a bar, and already visible: B's hunters turn over more slowly than A's (hunter generation 47 by update 150 against A's 90), because they live longer. So B gets more hunters a world but fewer generations.

**4. Noise, measured once.** Stage 19 anchored 100% against stage 17 full, against held-out grazers under this stage's test settings, on six evaluation seeds:

| training seed | a100 / full on evaluation seeds 12345, 1, 2, 3, 4, 5 | range |
|---|---|---|
| 0 | 1.194, 1.214, 1.215, 1.273, 1.234, 1.231 | 1.19 to 1.27 |
| 1 | 1.290, 1.325, 1.189, 1.287, 1.302, 1.302 | 1.19 to 1.33 |

**s_eval = .031** (pooled SD of the log ratio), so e^(2s) = 1.064. Training windows (stage 19 anchored 100%, seven windows from 301 to 1,000, both seeds: .0184 to .0198): **s_train = .0169**, so for bar 2 e^(2 x √2 x s) = 1.049. The bands, fixed now:

| bar | pass | no result | fail |
|---|---|---|---|
| 1, B / A | at least 1.25 (inside noise below 1.33) | 1.175 to 1.25 | under 1.175 |
| 2, B late / early | at least 1.15 (inside noise below 1.21) | 1.096 to 1.15 | under 1.096 |
| 3, C / D | at most 1.10 (inside noise above 1.034) | 1.10 to 1.170 | over 1.170 |

The standard evaluation seed's numbers in this table (.0152 and .0177 for anchored 100%) match stage 19's held-out check (.0153, .0177), so freezing bodies during the test changed little there.

## Result (2026-11-04): FAIL on bar 1. Cheaper living made hunters worse, not better. With bodies frozen, stage 19's plateau didn't appear

Against held-out grazers (the other seed's genes pool, never trained against), 64 worlds of 500 steps, catches per hunter-step, all hunters in the founder body, seeds 0 and 1. References: random .0137 (against `pp-cul-genes-1`, seed 0's held-out) and .0142 (against `-0`, seed 1's); trained genes hunters in the founder body .0424 and .0437.

| arm | held out | x random | x trained (founder body) | same-seed pool |
|---|---|---|---|---|
| A, anchored, usual upkeep | .0188, .0224 | 1.37x, 1.58x | 0.44x, 0.51x | .0262, .0242 |
| B, anchored, cheap upkeep | .0174, .0156 | 1.27x, 1.10x | 0.41x, 0.36x | .0237, .0233 |
| C, unanchored, cheap upkeep | .0138, .0142 | 1.01x, 1.00x | 0.33x, 0.33x | .0155, .0153 |
| D, unanchored, usual upkeep | .0243, .0306 | 1.77x, 2.16x | 0.57x, 0.70x | .0278, .0257 |

1. **Visible selection builds better brains: FAIL.** B / A = 0.929 and 0.694 (bar 1.25; the fail line is 1.175). B caught less than A on both seeds, well outside the noise band (±6.4%).
2. **It keeps climbing: pass, but it doesn't mean what it was meant to.** B's training catches per hunter-step, updates 901 to 1,000 against 101 to 200: 1.220 and 1.281 (bar 1.15; above the band, so a clean pass). A climbed more: 1.308 and 1.404.
3. **It isn't just easier living: pass, one-sided.** C / D = 0.567 and 0.463, far under 1.10. The cheap upkeep didn't lift hunters in unanchored worlds; it cut them to random level.

**Reported beside the bars:**

| arm | hunters a world (whole run, last 100) | hunter generations by 1,000 | hunter births a step | training catches per hunter-step, updates 1-100 → 901-1,000 |
|---|---|---|---|---|
| A | 7.5, 7.7 / 7.4, 7.4 | 643 / 657 | .120 / .117 | .0161 → .0275 / .0143 → .0246 |
| B | 29.8, 30.9 / 27.8, 30.5 | 182 / 259 | .140 / .157 | .0179 → .0259 / .0152 → .0240 |
| C | 31.0, 31.2 / 31.0, 31.2 | 107 / 109 | .097 / .098 | .0264 → .0322 / .0258 → .0328 |
| D | 15.6, 24.5 / 17.9, 28.6 | 661 / 661 | .268 / .279 | .0261 → .0427 / .0260 → .0476 |

(seed 0 / seed 1.) Total catches on held-out grazers, seed 0 then seed 1: A 1,081 and 1,577, B 955 and 785, C 649 and 678, D 1,909 and 3,937. In the usual test every arm's hunters were gone by step 500 (both species alive in 0.000 of worlds, except D on seed 1, 0.047). The trained hunters in the founder body kept both alive in 0.81 to 0.95 of worlds.

**After the fact, not a bar:** the same held-out test at B's cheap upkeep (`--world-set pred_base_cost=-0.0038`), to check that B and C weren't only losing because the test world charges the usual upkeep. The order held: A .0183, .0227; B .0179, .0176; C .0169, .0165; D .0230, .0302. B / A 0.98 and 0.78, C / D 0.73 and 0.55.

**My prediction:**
- Bar 1 passes at about 1.35x: wrong, and in the wrong direction (0.93x, 0.69x).
- Bar 2 passes at about 1.2x: right in number (1.22x, 1.28x), but my reason was wrong: the control climbed as well.
- Bar 3 passes, C within about 1.05x of D: passes, but not as predicted (0.57x, 0.46x). I didn't expect cheap living to do harm.
- A at about 1.2x to 1.3x random on held-out grazers: too low (1.37x, 1.58x).
- Nothing reaches trained hunters: right (best D, 0.57x and 0.70x).

**What's measured:**
- Lowering hunter upkeep to break-even at random made anchored hunters worse, on both seeds, in both test worlds, and made unanchored hunters no better than random.
- In cheap worlds hunters sat at 28 to 31 of 32 slots almost all run, with about a quarter (B) to a sixth (C) of the usual worlds' hunter generations.
- With bodies frozen, A kept climbing in training through 1,000 updates (1.31x and 1.40x from updates 101-200 to 901-1,000). Stage 19's anchored arm, with bodies evolving, went flat (1.04x, 1.05x). On held-out grazers A reached .0188 and .0224, against .0152 and .0177 for stage 19's anchored hunters under the same test settings (the noise runs).
- With bodies frozen, unanchored co-evolution (D) gave the best hunters of the stage, 1.77x and 2.16x random on held-out grazers and better than anchored A on both seeds (A / D 0.77, 0.73). Stage 17's unanchored arm, with bodies evolving, was at random level on held-out grazers (0.93x, 0.99x, from stage 19's check).

**What's inferred, not tested:**
- *Why cheap living hurts.* When a world's hunter slots are full, a hunter can only breed when one frees, and `_reproduce` hands free slots to ready parents in slot order (`rank < free.sum()`), not to the best hunters. So at the cap, breeding stops tracking hunting skill, and hunters turn over three to six times more slowly. That fits B and C, but no run separated it. Making break-even cheap did make better hunters gain energy; it didn't make them breed more, because slots, not energy, became the limit.
- *Why stage 19 plateaued.* Bodies frozen, A doesn't plateau, so the plateau looks tied to evolving bodies (the grazers' bodies moved the target, or the hunters' bodies drifted, or both), not to a limit of the search. This stage froze both at once, so which one it was isn't separated. Two seeds, and only the training-window noise was measured, not run-to-run noise.
- *Stage 17 and D.* Freezing bodies turned evolution alone from random level into about 2x random on unseen trained grazers. The difference between stage 17 and D is bodies (frozen and uniform here), so evolving bodies look like what made stage 17's co-evolved worlds useless as training. Whose bodies, and how, wasn't looked at; that reading is untested.

**Answer to the question:** the stage asked whether the plateau was the search or a selection that couldn't see better brains. Making better brains pay energy didn't help; it hurt, very likely because full worlds stop selection seeing skill at all. The plateau itself didn't come back once bodies were frozen. So it wasn't shown to be a search limit, and this selection fix wasn't the cure. Freezing bodies, added as a method fix, turned out to matter more than the treatment.

**Limits:** two seeds; one sigma (0.03); 1,000 updates; one upkeep value, set by the rule (no ladder, which Fable suggested); a frozen founder body for everyone, which isn't the body any trained grazer had; the test world charges the usual upkeep (checked after the fact at B's). The noise band is evaluation noise only. The cap mechanism is a reading of the code and the numbers, not a test.
