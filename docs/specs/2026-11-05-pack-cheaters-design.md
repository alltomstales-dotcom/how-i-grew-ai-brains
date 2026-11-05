# Stage 23: packs with cheaters. Does equal sharing select lower investment than contribution-based sharing?

Date: 2026-11-05. Written and committed before any code or run. The design comes from two reviews: Fable's proposal in (review, not published) (section 3), and Astra's corrections in (review, not published). Where they conflict, Astra's corrections win.

## The question

When a pack shares each kill equally, whatever each hunter put in, does selection lower the hunters' investment compared with a pack that shares in proportion to effort?

This is the question the bars judge. Whether equal sharing goes further and **collapses** the pack (effort near .15, catches falling, cheaters starving once workers are gone) is a separate hypothesis. It is reported, not barred.

## Why this world can ask it

- Stage 20 found that worlds filled to the hunter cap could not select, because free slots go by slot order, not merit. That was inferred, not shown. So this design makes reproduction visible to selection and checks it in a pilot before any full run: breeding order is random, the hunter cap is raised, and the pilot gate below must pass.
- Shared PPO doesn't stop a gene being selected (stages 7 to 13), but as the brain adapts it changes the fitness landscape. All arms of a seed start from the same brain, and the brain never sees effort.
- Stage 6a found that pack catches came from chasers meeting by chance, not from learned teamwork. That's fine here. The trait under selection is effort, a gene, not a tactic.

## The world

Base: `lasting_stealth` (stage 10f), the world stages 20 to 22 used, with `pack_size` 2. A grazer sees a hunter only from the next cell, and plants last (`seed_rate` .2).

### Catching (Astra's rule)

- **Assignment is unchanged.** As in `_pack_predation`, each hunter is assigned to the lowest-index living grazer within one cell (wrapped Chebyshev). A hunter is never credited to two hunts.
- **Catch chance.** A grazer whose assigned group has at least 2 hunters is caught with probability 1 − exp(−2·S), where S is the group's summed effort. A group of 1 never catches, and S = 0 never catches. This removes Fable's cliff at S ≥ 1, where .49 + .5 failed outright, and keeps diminishing returns. Two founders at .5 have S = 1 and catch with probability .865.
- **The meal.** The total meal is 50 (`pack_size` × `catch_energy`, as now), split one of two ways:
  - **E**, equally: 50 / n each;
  - **P**, in proportion to effort: 50 · e_i / S each.
- **Cost.** Effort costs .1 · e energy per step, every step, on top of body upkeep. It is charged after the energy cap, as camouflage was.
- **Clipping.** A hunter's energy is still capped at 60. A meal that takes it over the cap is partly lost. The meal a hunter actually keeps after clipping is measured, because the cap can erase a proportional reward.

### N, the neutral arm

N carries the same gene, mutating the same way. But the mechanics use .5 in its place everywhere: in the catch chance, in the cost (.05 a step) and in the split (which is then equal). The gene does nothing and costs nothing that depends on it. N is a reference, not a calibrated drift distribution: E and P change turnover and density, so N can't separate drift from every side effect.

### The gene

- **Slot.** Gene index 5, the old `diet` slot, carries effort. It is inert and unobserved when `omnivore` is off. It is reused only behind an explicit stage-specific flag, `effort` ("", "equal", "prop" or "neutral"; "" is off). The flag also turns on a mutation mask and labels the gene "effort" in metrics and replays. With it on, `omnivore` and `brains` must be off, and the world refuses to start otherwise. No old checkpoint or gene file is relabelled. Index 5 means effort only in runs whose config says `effort`.
- **Founders** have effort .5 exactly. Mutation sigma is .05, clipped to [0, 1].
- **"Blind."** The brain has no effort input. It still sees its own energy, as every brain does.

### Freezing

- **Every other gene is frozen.** For hunters, every gene except effort is held at the founder body (speed .2, sight .5) at initialisation, at birth and at every reset. For grazers, every gene is held there. This is a mask in the world, not the generic body freezing of `agents/freeze.py`, which would erase effort too.
- **Plants are fixed** and identical across arms: `plant_sigma` 0 and `founder_noise` 0, so every plant has the founder genes.
- **Grazer brains are frozen.** Only hunters learn (`--learners pred`).

### Breeding order and capacity

- **Random priority.** When more hunters are ready to breed than there are free slots, today's code fills slots in slot order. Under the `breed_random` flag (on in all three arms, off by default) eligible parents are ranked in a random order drawn each step, so a low slot number buys nothing. This is the same in every arm.
- **Raised hunter cap.** `max_pred` is 64, not 32. With these brains and founder bodies, stage 22b's home hunters held about 29 of 32 slots. Slack keeps the cap from deciding who breeds. Starting numbers are unchanged: 8 hunters and 24 grazers.

### New settings

All are off by default, so every other world is unchanged:
- `effort: str = ""`
- `effort_cost: float = 0.1`
- `effort_k: float = 2.0`
- `breed_random: bool = False`

Each step also records per-hunter effort, assigned grazer, group size, contribution share (e_i / S) and meal kept after clipping, but only when `effort` is on.

**Worlds.** `pack_effort_equal`, `pack_effort_prop` and `pack_effort_neutral` are `lasting_stealth` with `pack_size` 2, `max_pred` 64, `founder_noise` 0, `plant_sigma` 0, `gene_sigma` .05, `breed_random` on and `effort` set to the arm.

## Starting brains: why `pp-st-s`

- **Hunters.** Every arm of seed s starts from `pp-st-s`'s hunter. That's stage 10f's `lasting_stealth`: seeds 0 and 1, 150 updates, 768 worlds. These brains are the reference of stages 20 to 22. With founder bodies on open ground they're measured at .045 catches per hunter-step (stage 22b). They've never hunted under the pack rule, so PPO adapts during training, in every arm alike.
- **Why not stage 6a's pack runs.** `pp-pack-001` and `-002` did learn under the pack rule. But they come from the oldest code, with a different 10-channel world (no stealth, `seed_rate` .03). Their hunters evolved speed .66, so freezing them at founder speed .2 would put them far from what they trained on.
- **Grazers.** Each seed's frozen grazer brain is `pp-st-s`'s, with the founder body. The independently trained `pp-st-(1−s)` grazer is the held-out grazer in the evaluation, reported beside the main numbers.
- **The pilot** uses `pp-st-0`'s brains with seed 2, as stage 22's pilot did.

## Arms and runs

| arm | world | sharing | what effort does |
|---|---|---|---|
| E | `pack_effort_equal` | equal | catch chance, cost |
| P | `pack_effort_prop` | ∝ effort | catch chance, cost, share |
| N | `pack_effort_neutral` | equal | nothing (.5 used everywhere) |

- **Runs.** Seeds 0 and 1, 600 updates, `--num-envs 768 --sample-frac 0.35 --ckpt-every 25 --no-clips --learners pred --init-from runs/pp-st-s/checkpoints/latest.msgpack`, `--mem-gb 16`.
- **Names:** `pp-pk23-{E,P,N}-s{0,1}`, six runs in all.

## Measured during training (each update, from the rollout)

- **Effort:** mean and std of effort over living hunters (`pred_effort`, `pred_effort_std`).
- **Breeding:**
  - eligible birth attempts (hunter-steps at or above the breeding threshold);
  - attempts blocked for lack of a free slot;
  - births.
- **Selection:**
  - The breeding differential: over births, the parent's effort minus the mean effort of living hunters in its world that step.
  - The within-world correlation of effort and energy over living hunters, pooled from within-world deviations.
- **Hunts:**
  - assigned groups of two or more (attempts), and hunters in them;
  - catches, and hunters in successful groups (the mean group size of a catch);
  - lone hunters assigned to a grazer.
- **Meals:**
  - meal handed out, and meal kept after clipping;
  - meal kept by free-riders.
- **Free-rider:** a hunter whose share of its group's kept meal is greater than its share of the group's effort (e_i / S). The definition is Astra's. Under N's mechanics nobody is one.
- **Demography:**
  - hunters alive;
  - mean hunter generation;
  - starvations (a hunter's only death);
  - restarts by hunter extinction and by grazer extinction.

## The pilot gate (seed 2, about 1 node-hour)

**Pilot runs.** `pp-pk23-pilot-{E,P,N}` use the full settings and `pp-st-0`'s brains, at 200 updates each. The gate is judged on updates 101 to 200. Intervals come from the ten 10-update blocks there: the 2.5th and 97.5th percentiles of 2,000 resamples of those blocks.

All three must hold:

1. **Free slots aren't the limit.** In every arm, fewer than 5% of eligible birth attempts are blocked for lack of a free slot.
2. **Sustained multigeneration turnover.** In every arm:
   - the mean generation of living hunters averages at least 10 over updates 191 to 200;
   - it is at least 2 generations higher than over updates 141 to 150;
   - hunters keep breeding in every block: births per living hunter-step are above 0.
3. **Energy differences reach reproduction in mixed-effort cohorts.**
   - In E and P, the effort std among living hunters averages at least .03, so cohorts are mixed.
   - In E and P, the within-world effort–energy correlation has an interval that excludes 0.
   - In E and P, the breeding differential has the same sign as that arm's correlation.
   - In at least one of E and P, the breeding differential's interval excludes 0.
   - N's two numbers are reported as the reference.

**If the gate fails:** stop, write a dated addendum with the numbers and report back. Nothing is tuned to make the pack collapse. In particular, `effort_cost`, `effort_k` and the meal are not changed to chase a result.

**If it passes:** an addendum gives the pilot's numbers and the measured time per run before any full run. The pilot's E policy and effort distribution also set the evaluation spread (below).

## Evaluation (held-out, matched state)

`scripts/packs.py eval`, for each seed s:

- **Policies:** the last checkpoint of E's run (π_E) and of P's run (π_P).
- **Effort distributions:** the effort genes of E's and P's last gene snapshot (D_E, D_P).
- **The 2 × 2 cross.** π_E × D_E, π_E × D_P, π_P × D_E and π_P × D_P.
  - Each policy plays under its own arm's sharing rule. Within a policy the rule is fixed, so only the effort distribution changes.
- **Held-out and frozen.**
  - Fresh worlds: evaluation seed 12345, 256 worlds × 500 steps, no restarts.
  - Nothing learns, and mutation is off (`gene_sigma` 0). Children copy their parent's effort.
- **Matched state.**
  - Every cell starts from the same reset keys: the same positions, founder bodies, plants and densities (8 hunters, 24 grazers).
  - Each hunter slot gets the effort at the same quantile of D_E or D_P, from one shared uniform draw. So the cells differ only in the policy and in the effort distribution.
- **Grazers.** The training grazer `pp-st-s` decides the bars. The held-out grazer `pp-st-(1−s)` is reported beside it, and any disagreement in direction is flagged.
- **Per world:**
  - catches;
  - living hunter-steps;
  - starvations and births;
  - extinction and its step;
  - attempts, and hunters in successful groups;
  - grazer steps and deaths.
- **Measure:** catches per living hunter-step, summed over worlds.
- **Intervals:** 95%, from 1,000 paired resamples of the 256 worlds.
- **Evaluation spread.** The pilot's E policy and distribution are evaluated on seeds 12345, 1 and 2. The spread is the relative std of the catch rate over those three, and a ratio's spread is √2 times that.

## Bars (fixed now; each must hold on both seeds)

1. **Equal sharing selects lower effort.** E's mean effort over the final 100 updates (501 to 600) is at least .10 below P's and at least .10 below N's.
   - **Interval:** for each difference, the 2.5th to 97.5th percentiles of 2,000 resamples of the ten 10-update blocks in each arm.
   - **Pass:** both differences are ≥ .10 and both intervals exclude 0.
   - **No result:** a difference lies in [.08, .10), or is ≥ .10 with an interval that reaches 0.
   - **Fail:** anything else.
2. **Lower effort costs catches, and the effort is the cause.** Catches per living hunter-step, (π_E, D_E) / (π_P, D_P), are at most .80, and the 95% paired interval is below 1.
   - **Attribution** must also hold: within each policy, the effort distribution carries the deficit. (π_E, D_E) / (π_E, D_P) and (π_P, D_E) / (π_P, D_P) are both below 1, with intervals below 1.
   - **Pass:** the ratio is ≤ .80, its interval top is < 1, and attribution holds.
   - **Attribution fails:** the ratio passes but either within-policy ratio's interval reaches 1. This counts as a failure of bar 2 as an effort effect, and is reported as such.
   - **No result:** the point is within 2 evaluation spreads above .80, or ≤ .80 with the interval reaching 1.
   - **Fail:** anything else.

**Across seeds:** a bar passes only if it passes on both seeds, fails if it fails on either, and is otherwise no result. Fable's absolute bars (E below .3; P at least .5) are dropped, as Astra advised.

## Reported, not bars

- **Effort over time** per arm and seed, with std, and the final effort distributions.
- **Collapse (the separate hypothesis).** Does E's effort end at or below .15? Do catches per hunter-step fall over training? Does effort oscillate (worker loss, then cheater starvation)? These are reported per seed, with no bar.
- **Survival:**
  - hunters alive;
  - living hunter-steps;
  - hunter-extinction restarts per 1,000 world-steps;
  - grazer-extinction restarts.
- **Catches:**
  - total catches;
  - catches per hunter-step;
  - attempts, and the catch share of attempts;
  - mean group size of a catch, and of an attempt.
- **Starvations** per hunter-step.
- **Free-riding:**
  - the share of kept meal taken by free-riders;
  - the mean meal-minus-contribution gap;
  - meal lost to clipping, per arm.
- **The pilot gate's numbers** for every arm, at the end of the full runs.
- **Bar 2's decomposition:**
  - the policy effect: (π_E, D_P) / (π_P, D_P) and (π_E, D_E) / (π_P, D_E);
  - the held-out grazer's ratios.

## Readings (bars judged across seeds)

| bar 1 | bar 2 | reading |
|---|---|---|
| pass | pass | Equal sharing selects lower investment than either proportional sharing or drift, and the lower effort costs the pack a fifth or more of its catches. Effort is the cause, within fixed policies. |
| pass | fail or attribution fails | Equal sharing selects lower investment, but at this size it doesn't cost a fifth of the catches, or the loss can't be pinned on effort. Cheaters coexist with workers; no collapse. |
| fail: E ≥ .10 below P but not below N | any | Proportional sharing selects workers. Equal sharing doesn't select cheaters below the neutral reference. The difference is P rising, not E falling. |
| fail: E within .10 of P | any | At this cost and catch curve, the sharing rule doesn't change investment. |
| seeds disagree | | Report both seeds and say which reading each supports. No single reading. |

## Predictions (fixed before any run)

**Astra's:** E modestly below P, with coexistence more plausible than collapse to .15.

**Mine.**
- **Arms.** Final-100-update effort: P .85 to .95, E .40 to .60, N .45 to .55.
- **Bar 1:** E − P passes, E − N fails, so bar 1 fails. That's reading row 3: proportional sharing raises investment, and equal sharing doesn't push it far below neutral.
- **Bar 2 fails:** E/P near .85 to .95.
- **Why.** Under equal sharing a hunter's own effort still raises its own group's catch chance, and the catcher still gets its 50 / n. Take a pair at S = 1. A unit of effort adds about .27 to the catch chance, worth about 6.8 energy to the hunter per attempt. It costs .1 a step. So the balance depends on how often a hunter is in an attempt. At one attempt per 50 hunter-steps, E's optimum is S ≈ 1.15 for a pair (e ≈ .58). At one per 100, it's S ≈ .8 (e ≈ .4). Under P the same unit is worth about 28 per attempt, so P runs to the top of the range.
- **No collapse.** E won't fall to .15. Large crowds lower E's private return (50 / n), so E falls more where groups are big.

I hold the E range loosely, and the P direction firmly.

## Dynasties (for the studio)

**The tracer.** A new tracer, `scripts/pack_lineage.py`, reuses `scripts/lineage.py`'s uids, line tables and replay format. `lineage.py` itself isn't changed, because another export uses it.
- **Replays.** It replays E runs with the run's one hunter brain, and the final gene pool as founders.
- **Per animal, it records:**
  - effort (labelled "effort", never "diet");
  - steps assigned to a grazer, and attempts;
  - partner uids;
  - catches shared;
  - summed contribution share;
  - meal handed out, and meal kept after clipping;
  - free-ride surplus: kept meal minus contribution share × the group's kept meal.
- **Transfers between lines.** In each catch, the positive surpluses are paid for by the negative ones, in proportion. This gives a line-to-line transfer table: who fed whom.
- **Rankings.** `pk23_lines.md` ranks:
  - lines by free-riding (surplus received);
  - lines by working (surplus given);
  - worker → cheater line pairs by energy transferred;
  - lines by demographic drama, as `lineage.py` scores it.
- **Effort within lines.** The effort distribution of each line's living members is recorded over time, since mutating descendants need not share the founder's effort.
- **Output:** a few E replays go to `runs/lineage/pk23/`, in the 3-D viewer's format, with the effort, contribution and meal fields on every animal.
- **Limits.** These are lines within one episode, not training-wide dynasties. "Cheater" means meal share above contribution share, nothing more.

## Cost

| step | node-hours |
|---|---|
| pilot: 3 runs × 200 updates | about 0.5 |
| full: 6 runs × 600 updates, about 25 to 40 minutes each | about 3 to 4 |
| evaluation: 2 seeds × 4 cells × 2 grazers, plus the spread | about 0.3 |
| tracer | about 0.2 |
| **total** | **about 4 to 5** |

Compute is the second Spark and the Spark through `deploy/remote.sh`, with the memory guard and `--mem-gb 16`, one trainer per node. Nobody else's GPU work is stopped.

## Process

1. This spec and Astra's review are committed before any code.
2. Build with tests:
   - with every flag off, a step is identical to before;
   - the catch chance and the two splits are exact on hand-placed states;
   - N uses .5 everywhere;
   - the mask freezes every gene but effort;
   - random breeding order is fair.
3. Pilot, then an addendum with the pilot's numbers and the time per run, before the full runs.
4. Full runs, evaluation and tracer. Then a Result section that keeps the measured apart from the inferred.

## Addendum (2026-11-05): the pilot gate FAILS. Stopped before any full run.

**Runs.** `pp-pk23-pilot-E` ran on the second Spark (711 s) and `pp-pk23-pilot-P` on the Spark (667 s). Both used seed 2 and `pp-st-0`'s brains, with the full settings, for 200 updates. So one 600-update run would take about 33 to 36 minutes. N was not run. Criterion 1 must hold in every arm, and it failed in both E and P by a wide margin, so N's result could not change the outcome. That saved a third run.

**The gate**, judged on updates 101 to 200 by `python -m scripts.packs gate`. Intervals come from resampling the ten 10-update blocks. Files: `runs/pp-pk23-pilot-{E,P}/` (metrics, config, last checkpoint and gene snapshot), copied from the nodes.

| criterion | E | P | needed | holds? |
|---|---|---|---|---|
| 1. eligible birth attempts blocked for lack of a free slot | **99.71%** | **99.70%** | < 5% | **no** |
| 2. mean hunter generation, updates 191 to 200 | 32.0 | 32.7 | ≥ 10 | yes |
| 2. ... rise from updates 141 to 150 | +6.5 | +6.6 | ≥ 2 | yes |
| 2. births in every 10-update block | min 3,006 | min 4,871 | > 0 | yes |
| 3. effort std among living hunters | .207 | .181 | ≥ .03 | yes |
| 3. within-world effort–energy correlation, 95% | .0028 (.0013 to .0043) | .053 (.050 to .058) | excludes 0 | yes |
| 3. breeding differential (parent effort − world mean), 95% | +.0063 (.0056 to .0072) | +.0089 (.0080 to .0096) | same sign as corr; one excludes 0 | yes |

**Verdict: the gate fails on criterion 1.** By the spec, no full run is made, nothing is tuned, and this is reported back for a redesign.

### What happened (measured)

- **Hunters hit the raised cap.** Hunters filled the 64 slots by about update 25 and stayed there: 62.5 a world on average over updates 101 to 200, in both arms. They started at 8 founders, about 28 a world on update 1. Grazers sat at 95.9 of 96.
- **Hunting under the pack rule was easy.** Assigned groups caught on 89 to 96% of attempts, with 2.4 to 2.6 hunters in a successful group. Catches per living hunter-step were .033 (E) and .035 (P), against .045 for solo hunting in stage 22b. Over updates 101 to 200, starvation was .00037 (E) and .00053 (P) per hunter-step, and no world restarted for extinction of either species.
- **Most hunters were ready to breed and couldn't.** 72% (E) and 75% (P) of living hunter-steps were at or above the breeding threshold. Births were only .0021 and .0022 per hunter-step. Turnover came from a few deaths, and from restarts at step 500 that drew founders from the pool.
- **The cap ate meals.** Over updates 101 to 200, 38% (E) and 41% (P) of meal energy handed out was lost to the energy cap of 60.
- **Free-riding by the definition.** Free-riders kept 58% (E) and 58% (P) of the kept meal at update 200. In P that's almost entirely the cap. A high-effort hunter already near 60 loses part of its larger share, so its partner's kept share exceeds its effort share. This is the effect Astra warned the clipping measure would catch.
- **Effort still moved.** In both arms mean effort rose: E .50 → .62 and P .50 → .75 by update 200. P rose faster. Within worlds, higher-effort hunters held slightly more energy in both arms, and the parents that bred had above-average effort (the table).

### What this means (inferred)

- **Selection had slots to work with in only 0.3% of attempts.** Breeding order was random among ready parents, so slot order didn't decide who bred. But with nearly three quarters of hunter-steps ready and almost no free slots, who bred was mostly a lottery among the ready. Survival differences were tiny, because starvation was rare. Whatever selection there was ran through which parents happened to be ready when a slot opened. That's the condition stage 20 inferred and this stage meant to exclude. The effort shifts above are therefore not a test of the question, and nothing is read from them about E against P.
- **The rise in E is not evidence against free-riding.** It could be drift with clipping at the [0, 1] bound, hitchhiking on restart sampling, or real selection through readiness. N was not run, so none of these can be told apart here.

### What a redesign would need (not done here; for review)

The pilot shows hunters in this world are far above break-even under the pack rule. The meal is 50 for a pair against upkeep of about .9 a step, catches are easy on a full grazer cap, and the energy cap clips 40% of meals. Making room for selection means either:
- many more slots than hunters can fill, or
- hunters near break-even, with upkeep or the meal changed so that starvation, not slot vacancy, sets numbers.

The energy cap would also need to stop erasing P's proportional reward.

Each of these changes the economics that set effort's value. So they're a new design that needs review. They aren't a tuning step taken here.

### Cost

Two pilot runs of about 11 to 12 minutes each, on the second Spark and the Spark at the same time: about 0.4 node-hours. Nobody else's work was stopped.
