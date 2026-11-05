# Stage 21: do grazers invent an alarm call, and do hunters learn to eavesdrop?

Date: 2026-11-04. Written before any of these runs. Merged from two independent designs, Fable's and Astra's, written from one brief (reviews, not published). Tom asked for social dynamics to drive the series' drama; his reply to the plan was "Do it!!".

## Why calls can work now when 6b's didn't

Stage 6b's `signals` (ten move-and-call actions, one heard channel, radius 4, cost 0.05, own species only) failed: a call could say nothing that sight didn't already say, and 6c showed that a call can turn into a location beacon. In `lasting_stealth`/`tiny_stealth`, a grazer sees a hunter only from the next cell (`stealth_radius` 1) but hears calls up to 4 cells away. So a neighbour can pass on news the listener can't get any other way.

A selfish learner has no reason to warn others when calling costs the caller and helps someone else (Fable). So this stage also tests a **kin reward**, Hamilton's rule. Dynasties use the existing lineage `tag`, which is set per founder and inherited. Children are born on the parent's cell, so nearby animals are mostly kin.

## Mechanism

- **Call:** 6b's binary channel, on or off with each move. The caller pays a cost. The pilot sets the cost from the ladder {0.05, 0.15, 0.3}: the largest at which A1 grazers still call.
- **Hearing:**
  - Last step's calls, wrapped Chebyshev radius 4. No self-hearing.
  - Calls from animals killed that step disappear.
  - Newborns are silent.
  - Channels are zeroed at reset.
- **Eavesdropping:** a new flag `eavesdrop` lets hunters hear grazer calls (as a separate input channel). Grazers never hear hunters. Hunter-to-hunter calls stay off in every arm.
- **Kin share** (new): a grazer's reward adds `kin_share` times each death penalty paid by a grazer of its own dynasty within hearing. The ladder is {0.25, 0.5, 1.0}, and the pilot picks the smallest value at which calls carry information on the pilot seed. If none does, A1 runs at 0.5.
- **Learner:** the shared PPO trainer, one species at a time (`--learners`), with the other frozen.
  - Movement starts from competent `lasting_stealth`/`tiny_stealth` checkpoints, through a verified mapping from 5 actions to 10. Call outputs start with no meaning.
  - Body and plant genes are frozen throughout, so learned behaviour is isolated.
  - Not individual brains: evolution alone tops out at about 1.5x random (stages 17 and 19).
- **Nothing encodes danger** in the sim, and nothing rewards calling itself.

## Arms

All against frozen opponents (lesson 15). Equal update budgets, fixed by the pilot on seed 2.

- **Phase A, grazers learn** against frozen deaf hunters. Seeds 0, 1, 2.
  - **A0:** no kin share (6b's setup on the new map).
  - **A1:** kin share.
  - **A2:** kin share, grazers deaf in training. Same reward, no call, which isolates herding.
- **Phase B, hunters learn** against frozen A1 grazers. Seeds 0 and 1.
  - **B1:** hunters hear grazer calls.
  - **B0:** hunters are deaf.
- **Phase C, grazers adapt.** Both arms continue from A1's brain against the frozen B1 hunter. Seeds 0 and 1.
  - **C1:** hunter reception on.
  - **C0:** hunter reception off.

  This tests whether being overheard changes calling without also changing the hunter's skill (Astra).

If phase A finds no information in calls (bar 1 fails), stop the alarm claims. Eavesdropping on non-alarm beacons can still be reported.

## Tests

- 256 worlds × 500 steps, no extinction restarts, all learning and genes frozen.
- Paired starting states and random streams, and the declared final checkpoints, not the best.
- Repeated against an independently trained, held-out opponent.
- Interventions at test time:
  - as trained
  - **muted** reception
  - **shuffled** reception: call bits reassigned among existing grazer positions within density, energy and time strata, which keeps call counts and occupancy but breaks the link to danger
- Callers still decide and pay during the interventions.
- Uncertainty: bootstrap over worlds, not timesteps. The eval spread is measured once by rerunning on new world seeds. A miss inside that spread is "no result", not a fail.

## Bars (each needs a positive 95% interval and the threshold on every seed in its phase)

1. **Calls mean something (A1):**
   - mutual information between a grazer's call and a hunter within 1 cell before it acts is at least 0.05 bits, above the stratified permutation null
   - P(call | hunter adjacent) is at least 2× P(call | none)
2. **Listeners use it (A1):** among listeners with no hunter in sight, 50-step death risk with real calls is at least 10% lower than with muted calls and than with shuffled calls. A1 also dies at least 10% less than A2.
3. **Eavesdropping (B):** B1 catches at least 1.15× B0 per living hunter-step against identical frozen A1 grazers, and loses at least 10% when only its grazer-call reception is muted or shuffled.
4. **Counter-move (C):**
   - C1 lowers danger-conditioned calling by at least 20% compared with C0
   - C1 grazers die at least 10% less against B1 than A1 grazers do

   Policies are compared on identical saved states.

**Reported, not bars:**
- A0's information
- nearest-neighbour distance in A0, A1 and A2: A1 must not simply cluster tighter than A2, or the gain is herding
- how often an unaware listener moves away from a call
- overall survival, births and starvation
- total catches and extinction beside every rate

## Prediction (fixed before the pilot)

A0 fails, at about 0 bits, as in 6b.

A1 passes bars 1 and 2, at 0.1 to 0.3 bits with drops of 10 to 20% (Fable's call). Astra expects beacon exploitation to come more easily than a true alarm, so it's a real risk.

Bar 3 is the least confident: hunters see 3 cells and hear 4, so the news is a one-cell ring. I expect about 1.1×, a miss.

C: fewer calls next to hunters, but still selective, so a whisper rather than silence.

The most likely failure is credit. The caller pays now, and the benefit arrives later through another animal. Calls then collapse, and the kin reward teaches herding instead (A1 ≈ A2). That's a result, not a reason to retune.

## Biggest confound

**Herding or beacons passing for alarms.** The kin reward pays for keeping kin alive, and clustering does that without any call, while a call can also become a beacon (6c). Survival alone can't separate these. The design handles it with:
- A2 (same reward, no hearing)
- the occupancy-preserving shuffle
- the information measure
- listeners with no hunter in sight
- the clustering check

## Cost

- Pilot: about 1 GPU-hour.
- Runs: A 9, B 4, C 4, so 17 shared-PPO runs, plus evaluation.
- The reviewers' estimates differ by 3× (Astra: 10 to 14 GPU-hours; Fable: about 30), so measure a run's time in the pilot and write it here before the full runs.

## For the videos

**"They invented a word for danger"**, only if bars 1 and 2 pass: a paired clip from saved states. A grazer next to a hunter only it can see calls, and grazers 2 to 4 cells away, who can see nothing, turn away. On screen: the two call rates and the shuffled drop. "One call, one meaning": there is one kind of hunter here, not vervet-style distinct calls.

**"The hunters learned it"**, only if bar 3 passes: a B1 hunter beyond its sight turns toward a call and catches. Exploiting calls doesn't mean understanding a word.

C ends it, with the call-rate curve: "the grazers went quiet", or "they kept talking and paid for it".

**The kin reward ties to the dynasties:** families that warn each other.

## Addendum 1, build choices (2026-11-04, before any pilot run)

Where the spec was silent or didn't match the code, this is what was built. Nothing here changes an arm, a test or a bar.

**Brains.** The competent starting brains are stage 10f's `lasting_stealth` runs `pp-st-0` and `pp-st-1` (seeds 0 and 1, 150 updates, 768 worlds, the usual 115,750-weight net). The spec allows `lasting_stealth` or `tiny_stealth`; the usual net is the stronger learner (18a). `pp-st-0` gives every starting brain and the training opponent in phase A; `pp-st-1`, trained independently, is phase A's held-out hunter.

**Five actions to ten.** The logits layer is tiled, so move *a* with the call off and with it on get the same logit. Each move's chance is exactly the old brain's, and the call is on half the time whatever the animal sees. The new input channel gets zero weights. `tests/test_alarm.py` checks both.

**Hunters' outputs.** Both species share one net shape per world, so hunters have ten outputs too. In the alarm world a hunter's call bit does nothing: it costs nothing and nobody hears it. That is how hunter-to-hunter calls stay off in every arm.

**One input channel.** A grazer hears the calls of other living grazers that called on the last step, counted per cell across its 9 × 9 view (wrapped Chebyshev radius 4), as in 6b, minus its own call. A hunter's channel in the same place carries the same grazer calls when `eavesdrop` is on, and zeros when it is off. Since hunters never call, a second hunter-call channel would always be empty. The "separate input channel" is this one, in the hunter's own net.

**Dynasties.** The spec says the lineage tag "is set per founder". In the code, founders start with tag 0 (11c's tags are 0/1 marks). The new flag `dynasties` gives each founder its own tag (slot + 1) at every reset, including the restarts during training, whatever the gene pool carries. Pool tags would drift through restarts until everyone shared one tag, which would turn the kin reward into a group reward. Kin means the same tag. Children inherit it and are born on the parent's cell.

**Kin share.** It counts deaths from any cause, since the spec says "each death penalty paid". The distance is wrapped Chebyshev 4 or less, between the listener after it moves and the grazer where it died, measured before reproduction reuses the slot.

**Frozen bodies and plants.** `gene_sigma` 0, `founder_noise` 0, `plant_sigma` 0 (stage 20 kept plants at 0.05; this spec freezes them). Every animal has the founder body: speed .2 and sight .5, so radius 3. New world `alarm` (`AlarmConfig`) = `lasting_stealth` + `alarm` + `dynasties` + those three settings. Arms set `signal_cost`, `kin_share`, `prey_deaf` and `eavesdrop`.

**Training.** `scripts.train --world alarm --num-envs 768 --no-clips`. The species that learns is set with `--learners`, and the frozen one keeps its loaded brain. The new `--init-prey` and `--init-pred` take each species from its own checkpoint.
- **Phase A:** grazers start from `pp-st-0`'s grazer brain, against `pp-st-0`'s hunters, frozen and deaf. Seeds 0, 1 and 2 differ in training randomness, not in starting brains.
- **Phase B:** hunters start from `pp-st-0`'s hunter brain, against frozen A1 grazers of the same seed. B1 hears and B0 is deaf.
- **Phase C:** grazers continue from A1 of that seed, against that seed's frozen B1. C1 has hunter reception on, C0 off.
- **Held-out opponents:**
  - A is tested against `pp-st-1`.
  - B is tested against A1 seed 2, which B never trains on.
  - C is tested against B1 of the other seed.
  - Bars are judged against the training opponent. The held-out repeat is reported beside it, and any disagreement in direction is flagged in the result.

**Tests (`scripts/alarm.py`).**
- **Setup:**
  - Evaluation seed 12345, 256 worlds × 500 steps, no restarts.
  - Founder bodies and dynasties.
  - The last checkpoint of the fixed budget, which is the declared final one.
  - Conditions share starting states and random streams. Shuffles draw from separate streams.
- **Hunter within 1 cell before it acts:** wrapped Chebyshev 1 or less, in the state the grazer observes. The call is the action chosen in that state.
- **Information:** mutual information of the pooled 2 × 2 table, hunter adjacent × call.
  - The null is the mean information of 4 permutations of the calls per step. Each permutation is within strata of world × step × local density × energy.
  - Local density is the other living grazers within 4: 0, 1–2, 3–5 or 6+. Energy is in thirds of the cap.
  - Bar 1's first half: information at least .05 bits, and the 95% interval of information minus null above 0.
- **Shuffled reception:** the same strata. Last step's calls are reassigned among living grazers of the same world, step, density and energy. Counts and positions are kept, and only the listeners' input changes.
- **Listener:** a living grazer with no hunter within 1 cell that hears at least one call. Listeners are defined by the real calls in every condition, so muted and shuffled runs count the same kind of moment.
  - Death risk is death from any cause within 50 steps, the step itself included. Only steps up to 450 count, so every window is complete.
- **"A1 dies at least 10% less than A2":** deaths per living grazer-step, all causes, in the same evaluation worlds. A2 is tested as trained (deaf). Predation and starvation are reported separately.
- **Bar 3:** catches per living hunter-step.
  - Muted: the hunter's grazer-call channel is zero.
  - Shuffled: hunters hear shuffled calls and grazers hear the real ones.
- **Bar 4:**
  - **Danger-conditioned calling:** each policy's mean softmax chance of calling when a hunter is within 1 cell, on identical states. The states come from a reference rollout of A1 grazers against B1 hunters with reception on. C1 and C0 read every state as probes and never act. Each policy's own-rollout rate is reported too.
  - **Second half:** deaths per grazer-step, C1 against A1, both facing B1 with reception on.
- **Verdict:**
  - **Pass:** the point estimate at or beyond the threshold, and the 95% interval entirely on the effect's side of 1 for ratios, or of 0 for information minus null.
  - **The interval:** 1,000 resamples of worlds, with the same resampled worlds in paired conditions.
  - **No result:** a point estimate that misses its threshold by no more than 2 evaluation spreads. The spreads are measured once, in the pilot, on evaluation seeds 12345, 1 and 2 of the chosen A1 pilot run:
    - each bar-1 and bar-2 statistic: its standard deviation over the three seeds
    - ratios of catch or death rates between runs: √2 × the relative standard deviation of the rate

**Pilot rules (fixed now, seed 2, 300 updates, checkpoints every 25).**
1. **Cost ladder at kin share 0.5:** A1 at `signal_cost` .05, .15 and .3. "Still call" means a test call rate of at least .02 against `pp-st-0`, 256 worlds. That is 6b's lower bound for a channel in use. The pilot takes the largest such cost, or .05 if none qualifies.
2. **Kin ladder at that cost:** A1 at kin share .25 and 1.0 (.5 comes from step 1). The pilot takes the smallest value whose point estimates meet both of bar 1's thresholds (.05 bits, ratio 2). If none does, it takes .5.
3. **Update budget:**
   - The chosen run is tested at updates 100, 200 and 300 (64 worlds).
   - The budget is the first of those at which information, call ratio and deaths per grazer-step are each within 10% of their values at 300. Otherwise it's 300.
   - Every phase uses the same budget.
4. **Recorded in addendum 2 before any full run:** the chosen values, the measured time per run and the evaluation spreads.

## Addendum 2, pilot (2026-11-04/05, seed 2, before any full run)

**One setting matched to the starting brains.** The first launches ran out of device memory on update 1: a 12 GB cap at 768 worlds, with a single 7.6 GB allocation. `pp-st-0` and `pp-st-1` were trained with `--sample-frac 0.35` under a 16 GB cap, so every stage 21 run uses the same: `--num-envs 768 --sample-frac 0.35`, `--mem-gb 16`. No pilot result came from the failed launches.

**Step 1, cost ladder (kin share .5, 300 updates, test against `pp-st-0`, 256 worlds):**

| run | cost | call rate | information (bits) | P(call \| hunter adjacent) / P(call \| none) | deaths per grazer-step | hunters extinct by step 500 |
|---|---|---|---|---|---|---|
| `pp-al-pilot-c05` | .05 | .502 | .0001 | 0.92 | .0016 | .99 |
| `pp-al-pilot-c15` | .15 | .509 | .00001 | 1.02 | .0017 | .97 |
| `pp-al-pilot-c30` | .30 | .510 | .0002 | 1.10 | .0019 | .95 |

All three still call at about the starting .5, so by the rule **the cost is .3**. In training the call rate never moved from about .5 at any cost. Grazers sit at the 96-slot cap with plenty of food, so a call's energy hardly touches the reward. They did learn to evade without calls: catches per world-step fell from 1.4 to .32 over 300 updates, and hunters from 31 to 10 a world.

**Step 2, kin ladder (cost .3):**

| run | kin share | call rate | information (bits) | call ratio | deaths per grazer-step | hunters extinct |
|---|---|---|---|---|---|---|
| `pp-al-pilot-k25` | .25 | .522 | .0014 | 0.69 | .0013 | 1.00 |
| `pp-al-pilot-c30` | .5 | .510 | .0002 | 1.10 | .0019 | .95 |
| `pp-al-pilot-k100` | 1.0 | .579 | .0069 | 1.31 | .0087 | .13 |

None meets bar 1's thresholds (.05 bits, ratio 2), so by the rule **kin share stays .5**. Reported, not used: at kin share 1.0 the grazers die four to five times as often and hunters last, so a strong kin reward changed the world rather than producing a call.

**Step 3, budget** (`pp-al-pilot-c30` tested at updates 100, 200 and 300, 64 worlds):
- Information was .00065, .00002 and .00015 bits.
- The call ratio was 1.10, 1.03 and 1.10.
- Deaths per grazer-step were .0082, .0017 and .0017.

Information at 200 isn't within 10% of its value at 300, so **the budget is 300 updates** in every phase.

**Evaluation spread** (`pp-al-pilot-c30`, evaluation seeds 12345, 1 and 2; standard deviation over the three):
- information .00001 bits
- call ratio .0022
- listener risk, real over muted, .062 (values .74, .62, .66)
- listener risk, real over shuffled, .056 (values 1.02, .91, .96)
- ratios of death rates .052 (√2 × relative standard deviation .037)
- ratios of catch rates .0099 (√2 × .0070)
- ratios of danger-conditioned calling .0026 (√2 × .0018)

**What the pilot already shows (reported, not a bar):** muting grazers' reception raised listener risk by a third (real over muted .62 to .74), but shuffling barely changed it (.91 to 1.02). Pilot calls carry no information, so the muted drop comes from an input the brain had never seen silent, not from news. That is why the spec requires the occupancy-preserving shuffle as well.

**A caveat for phase A:** with founder bodies, `pp-st-0`'s frozen hunters starve in 95% of test worlds within 500 steps against learned grazers (catches .033 per hunter-step, against a break-even of about .035). So many listener-steps happen in worlds with few hunters left. That is the matchup the spec fixes, and the tests count it as it is.

**Measured cost:**
- One 300-update run takes 1,231 to 1,433 s, about 22 to 24 minutes, on either node. Each node was also shared with another session's light `train_indiv` runs.
- One 256-world test condition takes about 18 s after a compile of about a minute.
- The pilot used 5 runs, about 2 node-hours.
- The full stage is 17 runs, about 6.7 node-hours, or about 3.5 hours on two nodes. That's below both reviewers' estimates (Astra 10 to 14, Fable about 30).

**Full runs:** cost .3, kin share .5 for A1, A2 and C, 300 updates, seeds as in the spec. Seed 0's chain and seed 2's phase-A runs train on the Spark, seed 1's chain on the second Spark. Run names: `pp-al-{A0,A1,A2}-s<seed>`, `pp-al-{B1,B0}-s<seed>`, `pp-al-{C1,C0}-s<seed>`.

## Result (2026-11-05): FAIL on all four bars

17 runs (A 9, B 4, C 4) at cost .3, kin share .5, 300 updates, against frozen opponents. Every test: 256 worlds × 500 steps, evaluation seed 12345, 95% intervals from 1,000 world resamples. Numbers are against the training opponent, with the held-out opponent in brackets. Score files are in `runs/al21eval/score/`: `A_st0`, `A_st1`, `B_own`, `B_heldout`, `C_own`, `C_heldout0`, `C_heldout1`.

### Measured

**Bar 1, calls mean something (A1): FAIL on every seed, both halves.**

| seed | information (bits) | information − null, 95% | P(call \| hunter adjacent) | P(call \| none) | ratio, 95% (bar ≥ 2) |
|---|---|---|---|---|---|
| 0 | .0008 [.0007] | .0007 to .0008 | .399 | .510 | 0.78, 0.777 to 0.787 [0.78] |
| 1 | .0014 [.0012] | .0013 to .0014 | .365 | .515 | 0.71, 0.704 to 0.714 [0.71] |
| 2 | .0013 [.0014] | .0012 to .0014 | .356 | .506 | 0.70, 0.699 to 0.709 [0.69] |

Calls do carry a little information, above the null on every seed, but only a thirty-fifth to a sixtieth of the bar's .05 bits. It runs the wrong way for an alarm: grazers call *less* when a hunter is next to them. A0, with no kin reward, is the same: .0010, .00002 and .0026 bits, ratios 0.77, 0.96 and 0.64. The call rate never left about .5 in training in any arm (A1 seed 0: .485 at update 1, .498 at 300).

**Bar 2, listeners use it (A1): FAIL on every seed.** Listener risk is death within 50 steps for a grazer that hears a call and has no hunter within a cell.

| seed | real / muted (≤ .9) | real / shuffled (≤ .9) | A1 deaths / A2 deaths (≤ .9) |
|---|---|---|---|
| 0 | 0.81, .73 to .92 [0.85] | 1.04, .92 to 1.16 [0.98, no result] | 1.11, 1.00 to 1.22 [1.09] |
| 1 | 0.76, .69 to .85 [0.73] | 1.02, .91 to 1.15 [1.03] | 0.89, .80 to .98 [0.87] |
| 2 | 0.76, .67 to .85 [0.90] | 1.02, .91 to 1.12 [1.05] | 0.76, .69 to .83 [0.91, no result] |

Muting reception raises listener risk by about a quarter. Shuffling, which keeps the number of calls and where they come from, changes nothing. A1 against A2 goes both ways: A1 dies more on seed 0 and less on seeds 1 and 2.

**Bar 3, eavesdropping (B): FAIL on both seeds.**
- **Hearing against deaf hunters (bar ≥ 1.15):**
  - seed 0: 0.975, .972 to .978 [0.980]
  - seed 1: 1.036, 1.033 to 1.039 [1.041]
- **Muted against real hearing (bar ≤ .9):** .992 and .991 [.994, .991].
- **Shuffled against real hearing (bar ≤ .9):** 1.004 and 1.002 [.999, .999].

Both B arms learned to hunt A1 grazers well:
- catches per hunter-step .046 to .047, against .032 to .033 for `pp-st-0`
- training catches per world-step .62 at update 1, 1.44 at 300
- no hunter extinctions in the tests

Hearing added nothing to that.

**Bar 4, counter-move (C): FAIL on the first half (wrong direction); the second half passes.**
- **Danger-conditioned calling, C1 against C0 on identical states (bar ≤ 0.8):**
  - seed 0: 1.31, 1.31 to 1.32 [1.31]
  - seed 1: 1.16 [1.15]
  - C1 calls *more* than C0 next to hunters: probe chances .45 against .34, and .53 against .46.
  - The same comparison on each policy's own rollouts: 1.18 and 1.09.
- **Deaths per grazer-step, C1 against A1, both facing B1 (bar ≤ 0.9):** 0.030 and 0.023 [0.030, 0.030], about .0004 against .015.
  - C0 dies just as rarely, at .0004.
  - So the 33-fold gain comes from continued training against B1, not from anything about calls.

**Reported, not bars:**
- **Clustering:** nearest-neighbour distance A0/A1/A2 was 2.44/2.57/2.44 (seed 0), 2.51/2.50/2.48 (seed 1) and 2.52/2.54/2.57 (seed 2). A1 doesn't cluster tighter than A2, so there's no herding to explain away.
- **Moving away from a call:** listeners moved away from the nearest call .365, .341 and .339 of the time with real calls, against .363, .350 and .339 shuffled.
- **Survival, births and starvation:**
  - Starvation is almost nil (about 1e-5 per grazer-step).
  - Grazers sit near the 96-slot cap with no grazer extinctions.
- **Extinctions:** against the frozen deaf `pp-st-0` hunters, hunters died out by step 500 in 98 to 100% of A1 test worlds, with 18,843 to 21,177 total catches. Against B1, none did.

### Inferred (my reading, not tested)

- The call bit never came under pressure. With grazers at the slot cap and food plentiful, a call's energy cost of .3 hardly reaches the reward, so the bit stayed near the tiled start of .5 and drifted. That would explain why C1 and C0 differ in calling even though hunters never use calls (bar 3): two training runs drifting apart, not a response to being overheard.
- The muted drop is an input the brain never saw silent. In training half the neighbours always call, so an empty channel was new to the brain. The occupancy-preserving shuffle, which the spec required for exactly this reason, removes the whole effect.
- The prediction's most likely failure was credit, and it happened in a different form than predicted: calls didn't collapse toward zero, they never acquired a meaning at all. The kin reward didn't buy herding either.

### What can and can't be said

The spec allows "word for danger" and "eavesdropping" only where the matching bars pass, and none did, so neither phrase applies. What can be said: grazers given a free channel and a family reward didn't invent an alarm call, and hunters given their calls didn't use them.

### Clips (episode files, for reference, not evidence)

Paired from the same saved starting state and random stream (seed 12345), as trained and with reception muted. Local copies are in `runs/al21clips/`; the originals are on the Spark at `~/autolearn/runs/al21clips/`.
- `A1-s0_vs_st0_trained.json` / `A1-s0_vs_st0_muted_prey.json`
- `B1-s0_vs_A1-s0_trained.json` / `B1-s0_vs_A1-s0_muted_pred.json`
- `C1-s0_vs_B1-s0_trained.json` / `C1-s0_vs_B1-s0_muted_prey.json`
- `C0-s0_vs_B1-s0_trained.json` / `C0-s0_vs_B1-s0_muted_prey.json`

Each file lists its `moments`.
- **Moment counts:**
  - The spec's "warning" pattern (caller next to a hunter, an unaware grazer 2 to 4 cells away moves off) hits the 200 cap in A1 and B1.
  - B1 also has 58 eavesdrop-shaped moments (a hunter turns toward a call 4 cells away and eats within 5 steps).
- **They happen by chance:** calls are on half the time and carry no information, so such moments occur constantly.
- **Don't use them as "they invented a word" or "the hunters learned it".** An honest clip would show one of these moments beside the shuffled number, as a coincidence.

**Cost:**
- **Pilot:** 5 runs.
- **Full stage:** 17 runs.
  - A and C runs took 1,198 to 1,259 s each, B runs 732 to 749 s, so about 5.5 node-hours of training over 22:10 to 01:11 on two nodes.
  - The tests took under half an hour of GPU time.
