# Stage 22b: is the office loss lost skill, or just scarcer prey?

Date: 2026-11-05. Written and committed before any code or run. An evaluation-only follow-up to stage 22 (`2026-11-05-office-design.md`, Result in d145569). The design is Fable's, from (review, not published) ("Cheapest follow-up").

## Why

Stage 22's bar 1 is confounded. The frozen outdoor grazers starve heavily in the office, so living grazer-steps indoors are about .68 of home's. The home hunters' office/home catch ratio is .71. On a linear reading, scarcer prey alone could explain the whole drop. This test takes starvation away from the grazers and asks whether the drop is still there.

## The change

- **No training.** Every brain is the one stage 22 tested: the last checkpoint of `pp-st-s` (arm 1), `pp-of-home-s<s>` (arm 2) and `pp-of-office-s<s>` (arm 3), and the grazers of `pp-st-0` and `pp-st-1`.
- **One new world field**, off by default: `prey_no_upkeep: bool = False` in `EvoConfig`. When on, grazers pay no body upkeep at step 4 (metabolism). Their energy is still clipped at the cap, and the dead still hold 0.
  - **Why upkeep, not energy held at the cap.** In these worlds upkeep is a grazer's only steady energy cost. Alarm calls, walking costs, climbing and poison are all off, and a birth leaves each half with (27 − 8) / 2 > 0. So with upkeep at zero, a grazer's energy can't reach 0 and it can't starve. Holding energy at the cap would also put every grazer over its breeding threshold (27 against a cap of 40) every step, so it would change breeding far more. It would also pin the energy input the grazer brain sees. Zero upkeep is the smaller change.
  - **What it still changes.** Grazers still eat, so they get fatter and breed more often, at home and in the office alike. The grazer brain also sees its own energy, which will run higher than it did in training. Both effects hit home and office alike. Living grazer-steps are reported per condition, so whatever density gap remains can be seen.
- **`scripts/office.py eval --world-set KEY=VALUE`**, repeatable, as in `train_indiv` and `cross_indiv`. It is applied to every layout's config. Here it is `--world-set prey_no_upkeep=true`.
- **A test** (`tests/test_office_prey.py`): with the field off, a step is identical to before. With it on, a grazer at low energy keeps its energy and stays alive, where the same grazer with the field off starves.

## Tests

The settings are exactly stage 22's `scripts/office.py eval`, with the one world field changed:
- 256 worlds × 500 steps
- evaluation seed 12345, the same reset keys, layouts, founders and placements
- founder bodies, no restarts, nothing learning
- the layout sets home, held-out office (`office_seed` 1000) and training office (`office_seed` 0)
- arms 1, 2 and 3
- seeds 0 and 1
- the training ("own") grazer and the held-out grazer

Files go to `runs/of22b/main/s<s>_<arm>_<grazer>_<layout>.npz`, scored by the unchanged `scripts/office.py score` with the pilot spread .0086. That is 36 tests, about 20 s each after compiling. Arm 1 on home and held-out office is the test that matters. The others are there for bar 2 and for reference.

## Bar (fixed now, on both seeds, judged against the training grazer)

**Skill loss, not just scarce prey.** With grazers that can't starve, arm 1's catches per living hunter-step on held-out office layouts are at most 0.80 of its own on home. The 95% paired interval of that ratio must be below 1.

- **Verdict per seed:** stage 22's rule and spread, unchanged. Pass if the point is at or below .80 and the interval's top is below 1. No result if the point is within 2 × .0086 of the threshold (up to .814), or at or below .80 with the interval reaching 1. Fail otherwise.
- **Across seeds:** pass only if it passes on both. Fail if it fails on either. Otherwise no result.
- The held-out grazer's ratio is reported beside it, and any disagreement in direction is flagged.

## Readings (by arm 1's point estimate against the training grazer, on both seeds)

| arm 1 office / home, grazers can't starve | reading |
|---|---|
| at or below 0.80 (bar passes) | Most of the drop survives with prey that can't starve. It's a loss of hunting skill in corridors, not just scarcer prey. |
| above 0.80, up to 0.90 | Mixed. Part of the drop was scarce prey and part was lost skill. |
| above 0.90 | The drop was mostly prey scarcity. Stage 22's "caught less" stands, but not as lost skill. |
| seeds in different rows | Report both seeds, and say which reading each supports. No single reading. |

## Also reported (not bars)

- **Living grazer-steps** for every arm, layout and grazer, beside stage 22's. Also the office/home ratio of living grazer-steps for arm 1, with and without starvation.
- **Grazer deaths against catches.** With the field on, these should be equal. That shows the field did its job in the real test, not only in the unit test.
- **Bar 2 recomputed under the same rule.** Arm 3 over arm 2 on held-out office layouts, with grazers that can't starve. Stage 22's threshold of 1.25 applies, with the same verdict rule.
- **The rest of stage 22's reported numbers** under the new rule, as `score` prints them: arm 2's own office/home ratio, home retention, seen against held-out layouts, and recovery share.
- **Catches per living hunter-step per living grazer** (a density-adjusted rate). For each layout this is catches / (hunter-steps × mean living grazers), and the office/home ratio of that is computed for arm 1. It is computed both for stage 22's files (`runs/of22eval/main`, zero compute) and for these. It assumes catches scale linearly with prey numbers, which nothing here tests, so it is only reported.

## Prediction (fixed before any run)

I expect arm 1's ratio to rise from about .71 to between .78 and .88, so I lean "mixed". Hunters die out by step ~200 in the office, before most grazer starvation builds up, so I think scarcity covers less of the drop than the linear reading says. I hold this loosely.

## Cost

36 tests of about 20 s each, plus compiling: about 15 minutes on one node (the second Spark, or the Spark), with the memory guard and `EVAL_GB`. Stage 21b may be training on the same nodes, so nothing of theirs is stopped.

## Result (2026-11-05): PASS on both seeds. The office loss is not just scarce prey.

36 tests on the second Spark: 256 worlds × 500 steps, evaluation seed 12345, founder bodies, no restarts, `--world-set prey_no_upkeep=true`. Intervals are 95%, from 1,000 paired world resamples. Numbers are against the training grazer, with the held-out grazer in brackets. Files: `runs/of22b/main/` (`score_s{0,1}.json`, `prey_s{0,1}.json`, and `prey_stage22_s{0,1}.json` for stage 22's files), on the second Spark and copied locally.

### Measured

**The bar: arm 1's office / home, grazers that can't starve: PASS on both seeds.**

| seed | stage 22 (grazers starve) | stage 22b (they can't), 95% | loss |
|---|---|---|---|
| 0 | 0.710 [0.743] | **0.644**, 0.629 to 0.658 [0.695, 0.680 to 0.708] | 35.6% [30.5%] |
| 1 | 0.728 [0.708] | **0.681**, 0.667 to 0.695 [0.652, 0.637 to 0.665] | 31.9% [34.8%] |

The held-out grazer agrees in direction on both seeds. Every point estimate is far below 0.80 and outside the no-result band.

**Reading (row 1):** "Most of the drop survives with prey that can't starve. It's a loss of hunting skill in corridors, not just scarcer prey."

**The field did its job.** With it on, no grazer starved in any test: grazer deaths equal catches exactly, at home and in the office. Living grazer-steps are now level:

| arm 1 | home | office, held out | office / home |
|---|---|---|---|
| stage 22, seed 0 / 1 | 12.10M / 12.08M | 8.17M / 8.25M | 0.675 / 0.683 (about 218,000 grazers starved) |
| stage 22b, seed 0 / 1 | 12.17M / 12.17M | 12.15M / 12.15M | 0.998 / 0.998 (none starved) |

The 96 grazer slots are now about 99% full in both places (12.29M is the ceiling), so the density gap is gone. With arm 2 the numbers are the same. With arm 3 the office/home ratio is .968 and .985 [.993, .978], because arm 3 catches far more grazers indoors.

**Bar 2 recomputed: PASS, and much larger.** Arm 3 over arm 2 on held-out office layouts: **3.21** (3.14 to 3.30) [2.57] on seed 0 and **2.91** (2.83 to 2.98) [3.00] on seed 1. Stage 22 had 1.50 and 1.42.

**Rates** (training grazer; seed 0 / seed 1):

| hunter | layout | rate | catches | living hunter-steps | hunters extinct by 500 (median step) |
|---|---|---|---|---|---|
| arm 1, home | home | .0451 / .0451 | 166,956 / 164,351 | 3.70M / 3.65M | .01 / .02 |
| arm 1, home | office, held out | .0290 / .0307 | 12,052 / 14,543 | 0.42M / 0.47M | 1.00 (143) / 1.00 (154) |
| arm 2, continuation | home | .0468 / .0499 | 185,965 / 198,630 | 3.97M / 3.98M | 0 / 0 |
| arm 2, continuation | office, held out | .0268 / .0271 | 9,482 / 9,806 | 0.35M / 0.36M | 1.00 (132) / 1.00 (135) |
| arm 3, office | home | .0400 / .0396 | 89,998 / 80,654 | 2.25M / 2.04M | .23 / .33 |
| arm 3, office | office, held out | .0862 / .0789 | 345,333 / 312,703 | 4.01M / 3.97M | .02 / .02 |

**Also reported:**
- **Arm 2's own office/home ratio:** .573 and .544 [.693, .617], down from .653 and .657.
- **Home retention:** arm 3 over arm 2 at home was .855 and .793 [.883, .917]; over arm 1 it was .889 and .878 [.851, .911]. Seed 1 against the training grazer is just under the forgetting line of 0.8. That's one of four, so it is not "below 0.8 on both seeds".
- **Seen against held-out layouts** (arm 3): 1.019 and 1.016 [1.001, 1.018]. Still no memorising.
- **Recovery share:** 2.97 and 2.27 [3.55, 3.22]. It's over 1 because arm 3 in the office (.086, .079) now beats arm 2 at home (.047, .050). As a share it no longer means much. Read it as "more than won back".
- **The density-adjusted rate.** The linear rescaling of stage 22's own files gives arm 1 office/home **1.05 and 1.07 [1.09, 1.05]**: on that reading, scarcity explains all of the drop and more. The real test, with prey levelled, gives .64 and .68. So the linear reading was wrong here. Catches did not scale with prey numbers. In this run grazer-steps are level, so the adjusted ratio equals the plain one.

### Prediction against the result

I said .78 to .88, leaning "mixed". It was .64 and .68. Wrong: removing starvation made the office ratio worse for the home hunter, not better.

### Inferred (not tested)

- **Why the ratio fell.** At home, prey that can't starve change little (the rate goes .0449 → .0451). In the office the home hunter catches less per step (.0318 → .0290) and dies out sooner (median step 201 → 143), though there is more prey. A guess: in stage 22, starving grazers were weak, stuck in eaten-out rooms and easy to catch, and that propped up the home hunter's office rate. Well-fed grazers also see high energy in their own input, which may change how they move. Neither is measured.
- **Arm 3's big gain.** With prey that last, office-trained hunters keep their population (2% of worlds extinct, against 49 to 50% in stage 22), so they keep hunting. The gain is partly kept hunters, not only a better catch per step.
- **What this does not show.** It still doesn't show what the home hunter does wrong indoors, and doesn't show doorway use. "Skill" here means the catch rate falls with prey density held level. Nothing here says which behaviour breaks.

### Determinism check

Before the test, arm 1 was re-run with the field off, on home and held-out office for seed 0. It came out at 159,366 catches at home, against stage 22's 159,613, and 148 of 256 worlds differ slightly. A second re-run gave exactly 159,366 again, so the second Spark is deterministic now. The 0.15% gap to stage 22's file has an unknown source: the only code change between them is the off-by-default `oracle_call` flag. The ratio is unaffected: 0.7104 against stage 22's 0.7097, with grazer-steps office/home .676 against .675. Every 22b comparison is within this one run.

### What can and can't be said

Allowed on screen:
- "Even when we stopped the grazers starving, so the office held as many as outdoors, hunters trained outdoors caught about a third less there. It wasn't just scarce prey."

Not earned:
- "they can't use doors", or any account of what they do wrong
- "the office hunters got three times better". That's with prey that can't starve. Stage 22's 1.4 to 1.5× is the number for the real office.

This test replaces stage 22's confound caveat: the drop holds with prey levelled. Stage 22's own numbers stand as measured.

### Cost

36 tests plus two check re-runs, about 25 minutes on the second Spark, including a failed first launch that ran nothing. No training. Nobody else's work was stopped.
