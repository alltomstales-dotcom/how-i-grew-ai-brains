# Stage 21b: would grazers have listened if anyone had spoken? (the oracle call)

Date: 2026-11-05. Written before any code or run. The design comes from Fable's follow-up (review, not published; item 2). The fixed listener cohort, the danger-prevalence report and the interval wording come from Astra's cautions (review, not published). Stage 21 is `2026-11-04-alarm-calls-design.md`.

## Why

Stage 21 failed every bar. Fable read the code as a bootstrap failure. Listeners start with zero weights on the call channel, so a call changes nothing. So calling can't earn the caller anything, and calls never become something worth learning. That is a symmetric zero gradient.

Stage 21 can't tell which side failed:
- the speaker, who never said anything worth hearing
- the channel, where radius-1 news, heard up to 4 cells away, might be worth nothing in this world

This stage removes the speaker. Each grazer's call bit is forced to the truth, and the listeners are trained to use it.

**An oracle is never "invented a word".** Here the simulation does the calling, not the grazers. Whatever this stage finds, no video, caption or note may say that the grazers invented, learned or evolved a call. The most this stage can say is "they'd have listened if anyone had spoken", and only if it passes.

## Mechanism

- **The oracle (new flag `oracle_call`):** a living grazer's call bit on a step is on exactly when a living hunter is within wrapped Chebyshev 1 of it. That is measured in the state it observes, before it moves, the same "hunter adjacent" as stage 21's bar 1.
  - The grazer's own call output is ignored, and its chosen move is kept.
  - The caller still pays the .3 energy cost whenever the bit is on.
  - Everything else is stage 21's channel unchanged:
    - Calls are heard on the next step, within wrapped Chebyshev 4.
    - Killed callers are silent.
    - Nobody hears their own call.
- **Arm O (oracle):** stage 21's A1 with `oracle_call` on.
  - Alarm world, `signal_cost` .3, `kin_share` .5.
  - Grazers learn movement, starting from `pp-st-0`'s grazer brain, whose call-channel weights start at zero.
  - They play against `pp-st-0`'s hunters, frozen and deaf.
  - Bodies and plants are frozen.
  - Seeds 0, 1 and 2, 300 updates, 768 worlds, `--sample-frac 0.35`, a 16 GB cap. These are stage 21's settings (addendum 2).
- **Nothing else changes.** No retuning.

## Tests

The tests are stage 21's: 256 worlds × 500 steps, evaluation seed 12345, the final checkpoint at update 300, and conditions that share starting states and random streams.
- **Bar opponent:** the training opponent `pp-st-0`.
- **Held-out opponent:** `pp-st-1`, reported beside it.
- **Two reception interventions, as in stage 21:**
  - **Muted:** grazers hear nothing.
  - **Shuffled:** last step's calls move among living grazers of the same world, step, local density and energy third. Counts and positions are kept, and the link to danger is broken.

### The bar's measure: one fixed listener cohort (Astra's fix)

In stage 21, each condition's rollout picked out its own listeners. The rollouts diverged, so the conditions compared different selected populations.

Here the listeners are chosen once:
1. **Reference rollout:** grazers as trained, real reception.
2. **Branch steps:** t = 10, 20, ..., 450 (45 per world).
3. **The cohort at each branch step:** grazers in the reference that are alive, have no hunter within 1 cell and hear at least one call. This is stage 21's listener definition, from real calls.
4. **Branches:** from the reference state at t, the world is played for 50 steps (t to t + 49) three times, with real, muted and shuffled reception throughout the window. All three use the reference's random stream for those steps, so the real branch replays the reference exactly.
5. **Risk:** the share of the cohort that dies from any cause within those 50 steps, in each branch.

The same animals, in the same states, are compared under three receptions.

**Statistics:**
- **The two ratios:** real over muted, and real over shuffled. Each one pools deaths and cohort members over every branch step and world.
- **Uncertainty:** 95% intervals from 1,000 resamples of worlds. A world's branch steps move together.

### Reported, not bars

- **Stage 21's measure:** each condition in its own full rollout, as in stage 21's `eval` and `score_A`. It is reported beside the fixed-cohort result, and any disagreement in direction is flagged.
- **Danger prevalence (Astra):** per 50-step bin of the reference rollout:
  - hunters alive per world
  - the share of grazer-steps with a hunter adjacent, which under the oracle is the call rate
  - listener-steps per world
  - deaths per grazer-step
- **Early and hunter-present windows (Astra):**
  - both ratios over branch steps 10 to 150 only
  - both ratios over branch steps in worlds with at least one hunter alive at t
- **Other numbers:**
  - cohort size and the cohort's real-branch risk
  - how often a listener's next move goes away from the nearest call, against the shuffled value
  - total catches, hunter extinction by step 500, and overall deaths per grazer-step
- **Controls (eval only, before any training):**
  - **Pipeline check:** `pp-st-0`'s untrained grazers in the oracle world. Their call-channel weights are zero, so the three branches must be identical and both ratios exactly 1. If they aren't, the test is broken: stop and fix it before training.
  - **Untrained listeners:** stage 21's A1 grazers (seeds 0 to 2) in the oracle world. They were trained on meaningless calls, so this shows what the fixed-cohort test reads when listeners never learned what an oracle call means. It is reported only.
- **Evaluation spread:** O seed 2 at evaluation seeds 12345, 1 and 2. The spread is the standard deviation of each ratio over the three seeds. It is measured after training and before scoring, as stage 21 did in its pilot, and written into the Result.

## The one bar

On every seed (0, 1 and 2), against `pp-st-0`, in the fixed cohort, listener 50-step death risk must meet both of these:
- **real over muted:** at or below .9
- **real over shuffled:** at or below .9

Each needs the point estimate at or below .9 and the 95% interval's upper end below 1.

A point estimate that misses .9 by no more than 2 evaluation spreads is "no result" on that seed, not a fail, as in stage 21.

**Verdicts:**
- **PASS:** both ratios pass on all three seeds.
- **FAIL:** a ratio fails on any seed.
- **Inconclusive:** anything else. That means no fails, with at least one "no result".

The Result reports every ratio as a point estimate with its 95% interval. It never says "no effect". When a ratio fails, it says how large a benefit the interval still allows. For example: "listeners died at most X% less with real calls than with shuffled ones".

## Readings (fixed now)

- **PASS:** the channel works, and stage 21's failure was the speaker: "they'd have listened if anyone had spoken". A real retry, such as a listening bias or a cost in the reward, makes sense after this. Fable's item 2 puts it after this answer.
- **FAIL:** radius-1 news, heard up to 4 cells away, is worth nothing usable in this world, to these learners, at this budget. That holds within the interval's bound. Alarm calls are closed for this world.
- **Inconclusive:** neither reading is earned. Report the intervals, and don't rerun or retune to break the tie.

## Prediction (fixed before any run)

PASS, with both ratios around .75 to .85 on every seed.

The reasoning: a grazer sees a hunter only from the next cell, and a listener 2 to 4 cells from an oracle caller has a hunter 1 to 5 cells away that it can't see. Hunters and grazers have the same founder speed, so a grazer that steps away early mostly stays ahead. That news should be worth something.

**The most likely failure is thin data, not a true zero.** The frozen `pp-st-0` hunters starve out in about 95% of stage 21's phase-A test worlds by step 500. So oracle calls, and with them cohort listeners, mostly come early and in few worlds, and the intervals may be wide enough to give "no result" or a fail on the interval rule. The early-window report shows whether that happened.

A second risk: listeners may learn to treat a call as a hunter, which would be a pass, or only as "a grazer nearby", which would fail the shuffle half while still passing the mute half.

## Cost

- 3 training runs at about 21 minutes each (stage 21's A runs took 1,198 to 1,259 s), so about 1.1 node-hours.
- Tests and controls take under half an hour of GPU time.
- No pilot is planned: no parameter is chosen, and all settings are stage 21's. A 2-update smoke run checks the build. If anything has to change after that, an addendum is written here before the full runs.

Nodes are shared with stage 22: one trainer per node, 16 GB cap under `deploy/remote.sh`'s memory guard.

## Wording

Allowed only on PASS: "We made the warning call automatic. Grazers that heard it learned to get away." It may be followed by "they'd have listened if anyone had spoken".

**Never allowed:** "invented a word", "learned to warn", or "evolved an alarm".

On FAIL: "Even when we made the call automatic and true, hearing it didn't help them."

## Result (2026-11-05): PASS on every seed

**Runs.** Three runs, `pp-or-O-s0`, `-s1` and `-s2`, each 300 updates, trained on the Spark only, because stage 22 held the second Spark.
- Training took 1,633, 1,774 and 1,540 s.
- The oracle held throughout. Training logged P(call | hunter adjacent) = 1 and P(call | none) = 0.
- The call rate was about .035 in training and .010 to .011 in the tests.

**Tests.** 256 worlds × 500 steps, evaluation seed 12345. Intervals are 95%, from 1,000 world resamples.
- Score files: `runs/or21eval/score/O_st0.json` and `O_st1.json`.
- Spread: `runs/or21eval/spread.json`.

There was no addendum. Nothing changed between the spec and the runs. One scoring bug, a key collision in `score_O`, was found and fixed before the first score was read.

### Measured

**Controls, run before training.**
- **Pipeline check (`pp-st-0` grazers, which have zero call weights):** all three branches were identical, with both ratios exactly 1.000 on 230,001 cohort grazers. The test is sound.
- **Untrained listeners (stage 21's A1, seeds 0, 1 and 2, in the oracle world):**
  - real over muted: .95, .97 and .96
  - real over shuffled: .93, .98 and .97

  So a true call is worth a few percent even to listeners that never learned what it means.

**The bar.** This is the fixed cohort, against `pp-st-0`, with `pp-st-1` in brackets.

| seed | cohort | 50-step risk: real / muted / shuffled | real over muted (≤ .9) | real over shuffled (≤ .9) |
|---|---|---|---|---|
| 0 | 28,782 | .138 / .364 / .290 | **.380**, .362 to .399 [.345] | **.476**, .452 to .499 [.448] |
| 1 | 28,521 | .140 / .373 / .287 | **.376**, .361 to .394 [.351] | **.488**, .466 to .510 [.458] |
| 2 | 25,729 | .130 / .355 / .277 | **.367**, .350 to .385 [.358] | **.472**, .449 to .496 [.471] |

Every ratio passes on every seed, against both opponents.
- **Mute:** real calls cut a listener's 50-step death risk to 35 to 40% of its muted value. The interval allows 60 to 65% less risk.
- **Shuffle:** real calls cut the risk to 45 to 51% of its shuffled value.
- **Evaluation spread** (O seed 2, evaluation seeds 12345, 1 and 2; sample SD): .0064 for real over muted and .018 for real over shuffled, so no result was near the band.

**Same cohort, first move.** Listeners stepped away from the nearest call .85 to .87 of the time with real calls. That fell to .35 to .39 muted and .44 to .46 shuffled.

**Reported, not bars.**
- **Stage 21's measure:** each condition in its own rollout. It agrees in direction and is stronger:
  - real over muted .24 to .25 [.22]
  - real over shuffled .40 to .43 [.38 to .42]
- **Danger prevalence (reference rollout, against `pp-st-0`, per 50-step bin):**
  - Hunters alive per world: about 17, 7, 2, 0.7 and 0.2 in the first five bins, then about 0. Hunters died out by step 500 in 100% of test worlds.
  - The share of grazer-steps with a hunter adjacent, which is the oracle call rate: .068, .032, .010, .003 and .001.
  - Listener-steps per world: about 500, 400, 140, 45 and 15, then about 0.
- **Early and hunter-present windows:**
  - **Early (branch steps 10 to 150):** this holds about 95% of the cohort. The ratios are .37 to .38 muted and .48 to .49 shuffled.
  - **Hunter present:** this holds 99.8% of the cohort, and the ratios equal the bar's.
  - So the result comes from the first 150 steps, while hunters are alive. Pooling didn't dilute it, because there was almost no later data to pool.
- **Overall:**
  - O grazers died .0007 to .0008 per grazer-step against `pp-st-0`, with 7,900 to 9,800 catches in 256 worlds.
  - Stage 21's A1 died .0016 to .0018 per grazer-step, with 18,843 to 21,177 catches.
  - O and A1 were trained separately, so this is a comparison of two arms, not a test of calls.

### Reading (as fixed)

**The channel works, and stage 21's failure was the speaker: they'd have listened if anyone had spoken.**
- Radius-1 news heard up to 4 cells away is worth a lot in this world. It more than halves a listener's 50-step risk against the same listener hearing shuffled calls.
- In 300 updates, listeners learned to treat a call as "danger over there", and that wasn't just "a grazer nearby". They step away from it 86% of the time.
- So stage 21's null comes from the speaker side: the zero gradient on calling that Fable diagnosed. Listening itself was learnable.

### Prediction check

I predicted a PASS with ratios around .75 to .85. The direction was right, but I underestimated the size by about 2x.

My named failure risk, thin data from hunters dying out, was real in shape: almost no cohort after step 200. But it wasn't real in effect, because about 27,000 cohort grazers came from the first 150 steps.

### What can and can't be said

**Allowed:** "We made the warning call automatic. Grazers that heard it learned to get away. Hearing a real call cut their risk of dying in the next 50 steps to about 40% of what it was with the calls switched off, and about half of what it was with the calls scrambled." It may be followed by "they'd have listened if anyone had spoken".

**Never:**
- "invented a word", "learned to warn" or "evolved an alarm": the simulation made every call.
- "the grazers learned to call"

**Next, from Fable's item 2:** a real speaker retry comes after this answer, with a listening bias or a cost in the reward. It is not part of this stage.

### Cost

- Training: about 1.4 node-hours (4,947 s).
- Smoke run, controls and tests: about 0.5 node-hours.
- All on the Spark, 01:41 to 03:12 node time, so about 1.5 hours of wall clock. The second Spark wasn't used.
