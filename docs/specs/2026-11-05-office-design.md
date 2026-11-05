# Stage 22: new places, the office. Does outdoor hunting carry through corridors, and can retraining win it back?

Date: 2026-11-05. Written before any code or run. The design is Astra's, from (review, not published) ("Next: new places, office first"). It follows Tom's ask of 2026-11-04 to move the brains into other places. The animals stay hunters and grazers. The office is walls, corridors and doorways built from the existing WALL terrain. Nobody here does office work.

## Why

So far every trained brain has lived on open ground, or among a few random wall segments (stage 3a). The closest earlier test is stage 13d's transplant, which moved brains between climates and found training repeats within a few percent, with one-way hunter effects of 9 to 22%. That was a change of bodies and food. The office is a change of shape: rooms, doorways and corridors a hunter has to go through to reach grazers it can see. Walls take away moves that open ground always allowed.

The question has two halves. Does a hunter trained outdoors lose skill in the office? If it does, does training in the office win the skill back, compared with spending the same training on more home practice?

## The office

A new module, `simulations/office.py`, holds a fixed, seeded generator. `office_map(size, seed)` returns a (size, size) map of wall cells. Seeds pick layouts, so the same seed always gives the same office.

- **Bays.** Along each axis the 48-cell ring is cut into three bays, each 12 to 20 cells wide. A bay is a corridor 2 cells wide, a wall, then one or two rooms with a wall between them and a wall after the last. The pattern is rolled to a random start.
- **Cells.** A cell is open if it's in a corridor on either axis. It's a wall if it's on a wall line on either axis. Otherwise it's room floor. So corridors form a grid that wraps around the torus, and rooms fill the blocks between them, 3 to 14 cells across.
- **Doorways.** Every room that shares a wall with a corridor gets one doorway, 1 or 2 cells wide, in one such wall. Each wall between two rooms gets a doorway half the time. A last pass opens a door from any room still cut off into a reachable neighbour, so every open cell can be reached from every other.
- **What walls do.** Walls already block movement for both species and show in the wall input channel (`evo.observe`). Plants never grow on them, and animals are never placed on them. They do not block sight, so this is an office with glass partitions. That is a limit of the existing terrain, and I'm keeping it rather than changing how sight works.
- **Wall density.** A sketch of the generator gives 21 to 26% walls over 32 seeds.

**World settings.** The new `EvoConfig` fields are `office: int = 0` (how many layouts are in the bank; 0 is off) and `office_seed: int = 0` (the first layout's seed). With `office` on, each reset picks one of layouts `office_seed` to `office_seed + office - 1` at random, from a key folded out of the reset key. Restarts during training therefore draw new layouts from the bank. With `office` 0, nothing in `reset` changes, so every other world is unchanged.

**Layouts.**
- **Training:** seeds 0 to 15.
- **Held-out test:** seeds 1000 to 1015. No arm ever trains on these.

**New worlds.**
- **`office_home`** is `lasting_stealth` with bodies and plants frozen: `gene_sigma` 0, `founder_noise` 0, `plant_sigma` 0. Every animal has the founder body: speed .2 and sight .5, so radius 3. The grazer sees a hunter only from the next cell.
- **`office`** is `office_home` with `office` 16 and `office_seed` 0. The held-out test is `office` with `office_seed` 1000.

**Generator tests** (`tests/test_office.py`):
- Every open cell can be reached from every other, by north, south, east and west steps on the torus, for all 32 bank seeds.
- The same seed gives the same map, different seeds give different maps, and the training and test banks share no map.
- Wall density is between 15% and 35% on every bank seed.
- Every map has doorways: a gap in a wall line between a room and a corridor.
- `reset` with `office` 0 gives exactly the map it gave before. With `office` on, the map is one of the bank's, and no animal or plant sits on a wall.

## Arms

Bodies and plants are frozen. Training uses the shared PPO trainer (`scripts.train`) with `--learners pred`, so only hunters learn and the grazers are fixed opponents.

- **Starting brains.** These are stage 10f's `lasting_stealth` runs `pp-st-0` and `pp-st-1`: seeds 0 and 1, 150 updates, 768 worlds, the usual 115,750-weight net, trained on open ground with evolving bodies.
- **Seed s.** The training grazer is `pp-st-s`'s grazer, and the independently trained held-out grazer is `pp-st-(1−s)`'s.

| arm | hunter | training |
|---|---|---|
| 1, home | `pp-st-s`'s hunter, unchanged | none |
| 2, home continuation | from `pp-st-s` | 300 updates in `office_home` |
| 3, office retraining | from `pp-st-s` | 300 updates in `office`, training layouts |

- **Arms 2 and 3 have equal budgets.** Both run 300 updates, `--num-envs 768 --sample-frac 0.35 --ckpt-every 25 --no-clips`, `--mem-gb 16`, seed s, with both brains taken from `pp-st-s` by `--init-from`. Stage 21 used the same settings.
- **Run names:** `pp-of-home-s<s>` and `pp-of-office-s<s>`. That's four training runs.
- **Why arm 2 is there.** Arm 2 adapts to founder bodies, frozen plants and the extra practice, exactly as arm 3 does, but on open ground. So arm 3 against arm 2 isolates what training in the office adds.

## Tests

`scripts/office.py`:
- Each test is 256 worlds × 500 steps, evaluation seed 12345, with no restarts and founder bodies.
- Nothing learns.
- Each test uses the last checkpoint of the fixed budget.

Every arm is tested on:
- **home:** `office_home`, open ground
- **office, held out:** `office` at `office_seed` 1000, the 16 layouts never trained on. This is where the bars are judged.
- **office, training layouts:** `office` at `office_seed` 0. Reported only, to show how much of arm 3's gain is memorising its 16 layouts.

Each is played against the training grazer and against the held-out grazer. All arms of a seed see identical opponents, starting states and random streams: the same reset keys give the same layout, founders and placements.

**Per world, recorded:**
- catches
- living hunter-steps
- hunter starvations (a hunter's only death)
- hunter births
- hunters extinct by step 500, and the step it happened
- grazer living steps, deaths and births

**Measure:** catches per living hunter-step, summed over worlds.

**Intervals:**
- 95% intervals come from 1,000 resamples of the 256 worlds.
- Resampling is paired: the same resampled world indices are used for every arm and layout set compared.

## Bars (fixed now, on both seeds, judged against the training grazer)

1. **Transfer loss ≥ 20%.** Arm 1's catches per living hunter-step on held-out office layouts are at most 0.80 of its own on home, with the 95% interval of that ratio below 1.
2. **Retraining recovers.** Arm 3 catches at least 1.25× arm 2 per living hunter-step on held-out office layouts, with the 95% paired interval above 1.

**Verdict for each bar and seed:**
- **Pass:** the point estimate is at or beyond the threshold, and the interval excludes 1 on the effect's side.
- **No result:** the point estimate misses the threshold by no more than 2 evaluation spreads, or it is beyond the threshold but the interval includes 1.
- **Fail:** anything else.

**A bar across seeds:**
- It passes only if it passes on both seeds.
- It is "no result" if it does not fail on either seed but doesn't pass on both.
- It fails if it fails on either seed.

**Held-out grazer.** The same two ratios against the held-out grazer are reported beside each bar. Any disagreement in direction is flagged in the result.

**Evaluation spread.** The pilot run is tested on evaluation seeds 12345, 1 and 2, on home and on held-out office layouts. For each, the relative standard deviation of the catch rate over the three seeds is computed. The spread of a ratio is √2 × the larger of the two. This matches stage 21.

**Reported, not bars:**
- **Totals:** catches and living hunter-steps for every arm, layout set and grazer.
- **Starvation:** hunter starvations per living hunter-step.
- **Extinction:** the share of worlds whose hunters died out by step 500, and the median step.
- **Grazers:** deaths and births.
- **Home retention:** arm 3's home catch rate over arm 2's, and over arm 1's. This asks whether learning the office cost skill on open ground.
- **Recovery share:** (arm 3 office − arm 2 office) / (arm 2 home − arm 2 office). That's the share of the continued hunter's own office loss that office training won back. It is only meaningful when arm 2 also loses in the office.
- **Arm 2's transfer loss:** its own office / home ratio, beside arm 1's.
- **Seen and held-out layouts:** arm 3 on its training layouts against held-out ones.
- **Random hunters:** a random-policy hunter against the training grazer, on home and on held-out office layouts, as a floor.

## Reading each outcome

| bar 1 (loss ≥ 20%) | bar 2 (≥ 1.25×) | reading |
|---|---|---|
| pass | pass | Corridors cost outdoor hunters at least a fifth of their catches, and practice in the office won back a real share. The recovery share says whether it's partial or full. |
| pass | fail | Corridors cost, and 300 updates of office practice didn't win it back over more home practice. The office is harder to learn, or needs more than this budget. "Can't learn the office" isn't earned on one budget. |
| fail, office below home | pass | Outdoor hunting mostly carries over, yet office practice still adds a quarter or more. The office holds a skill of its own beyond what open ground teaches. |
| fail, office below home | fail | Outdoor hunting carries through corridors, and there is little to recover. To these brains the office isn't a very different place. |
| fail, office at or above home | either | Walls help hunters. Rooms make corners (stage 3a's idea), so the "loss" is a gain. Report it as that, not as transfer. |
| no result or split seeds | | Say no result for that bar, with both seeds' numbers. |

If home retention is below 0.8 on both seeds, office training also cost skill outdoors. That gets reported as forgetting, whatever the bars say.

## Prediction (fixed before any run)

- **Loss:** 20 to 35% for arm 1 in the held-out office, so bar 1 passes. Hunters see grazers through glass walls and walk into them. They waste steps at walls and don't route through doors.
- **Recovery:** partial. Arm 3 reaches about 1.3× arm 2 in the office, and bar 2 passes narrowly, with a recovery share between a third and two thirds.
- **Home retention:** at least 0.9.

I hold this loosely. The main risk is the corner effect: grazers see hunters only from the next cell, and a grazer in a room has fewer ways out. That could make the office easier for hunters, not harder. A second risk: with frozen founder bodies, the home hunter may starve in many worlds, which leaves few hunter-steps to measure. Extinction is reported beside every rate for that reason.

## Pilot (rules fixed now)

- **Run.** One office retraining run on pilot seed 2 (`pp-of-pilot`, from `pp-st-0`, against `pp-st-0`'s grazer), 300 updates. Seed 2 isn't a full-run seed.
- **What it gives.** The measured time per run, the peak memory, and the evaluation spread (seeds 12345, 1 and 2 on home and held-out office).
- **Generator.** Its parameters are fixed in this spec, and the pilot doesn't change them unless the generator tests fail.
- **Fixed regardless.** No pilot number changes an arm, a bar or the budget. If the 16 GB cap runs out, the pilot may change `--mem-gb` or `--sample-frac`, and the same values then go into every run.
- **Addendum.** Written before any full run, with the pilot values and the time per run.

## Replays for the studio

- **What.** One office `episode3d.json` per arm and seed, made by `scripts/lineage.py`, so every animal carries a lineage uid, parent, line, name and cause of death. Each is played on held-out layout 1000 against that seed's training grazer, 8 episodes × 500 steps, at evaluation seed 12345.
- **How.** Each replay is made from a small replay run folder. Its config is `office` with `office` 1 and `office_seed` 1000, and its checkpoint holds the training grazer and the arm's hunter. `lineage.py` writes one terrain map per replay, from the first world, so every episode in a replay folder uses the same layout.
- **Where.** `runs/lineage/of22-<arm>-s<s>/`.
- **What not to say.** These are shared-PPO runs: every hunter in a run has the same brain. Lineages are lines of bodies and families, so names like Flint and Slate label founders and their descendants, not individual minds. Don't say a hunter learned the office or remembers a door.

## Cost

- **Training:** four runs plus the pilot, at about 12 to 13 minutes each. Stage 21's hunter-only B runs took 732 to 749 s at the same settings.
- **Tests:** 3 arms × 2 seeds × 3 layout sets × 2 grazers is 36 tests, at about 20 s each after compiling.
- **Total:** about 1.5 node-hours. Measured in the pilot.

Compute goes on the Spark and the second Spark through `deploy/remote.sh`, with the memory guard and `EVAL_GB`. Stage 21b is running on the same nodes at the same time, so runs are queued and the nodes shared.

## Addendum 1, pilot (2026-11-05, before any full run)

**Run.** `pp-of-pilot`: office retraining from `pp-st-0`, against `pp-st-0`'s frozen grazer, seed 2, 300 updates, on the second Spark.
- Settings: `--num-envs 768 --sample-frac 0.35 --mem-gb 16`, as the spec fixed them.
- Peak device memory was 7.81 GB under the 17.18 GB cap, so nothing changes.
- Training catches per world-step rose from .56 at update 25 to 1.23 at update 300.

**Measured cost.**
- One 300-update run takes **700 s, about 12 minutes**, on the second Spark with the GPU to itself.
- The four full runs come to about 47 minutes on one node.
- One 256-world test takes about 20 s after a compile of about a minute.

**Evaluation spread.** The pilot hunter was tested against `pp-st-0`'s grazer on evaluation seeds 12345, 1 and 2, 256 worlds each.

| layout | catches per living hunter-step | relative SD |
|---|---|---|
| home | .03824, .03830, .03836 | .0016 |
| office, held out | .04628, .04656, .04685 | .0061 |

The spread of a ratio is √2 × .0061 = **.0086**. A point estimate that misses its threshold by no more than 2 × .0086 of the threshold is "no result". That is a ratio of 0.80 to 0.814 for bar 1, and 1.229 to 1.25 for bar 2.

**What the pilot shows (reported, not a bar).** The office-trained pilot hunter catches more per living hunter-step on held-out office layouts than on open ground, .046 against .038. Hunters died out by step 500 in 40 to 54% of test worlds. This is seed 2, and it isn't any arm's score.

**Full runs.** Unchanged from the spec: `pp-of-home-s<s>` and `pp-of-office-s<s>`, seeds 0 and 1. Seed 0 trains on the Spark and seed 1 on the second Spark. Either node takes the other's runs if it frees up first.

## Result (2026-11-05): PASS on both bars, on both seeds and against both grazers

Four training runs on the second Spark took 635 to 675 s each. Every test was 256 worlds × 500 steps, evaluation seed 12345, founder bodies, no restarts. Intervals are 95%, from 1,000 paired world resamples. The rate is catches per living hunter-step. Numbers are against the training grazer, with the held-out grazer in brackets. Score files: `runs/of22eval/main/score_s0.json` and `score_s1.json` (with the per-world `.npz` files), on the second Spark and copied locally.

### Measured

**Bar 1, transfer loss ≥ 20%: PASS.** This is arm 1, the unchanged home hunter, on held-out office layouts over its own rate on open ground.

| seed | office / home, 95% | loss |
|---|---|---|
| 0 | 0.710, 0.699 to 0.721 [0.743, 0.734 to 0.752] | 29.0% [25.7%] |
| 1 | 0.728, 0.718 to 0.738 [0.708, 0.695 to 0.719] | 27.2% [29.2%] |

**Bar 2, office retraining ≥ 1.25× home continuation: PASS.** This is arm 3 over arm 2, both on held-out office layouts.

| seed | arm 3 / arm 2, 95% |
|---|---|
| 0 | 1.501, 1.470 to 1.533 [1.406, 1.385 to 1.431] |
| 1 | 1.421, 1.397 to 1.448 [1.447, 1.417 to 1.478] |

The held-out grazer agrees in direction on both bars and both seeds. Every point estimate is far outside the no-result band of 2 × .0086.

**Rates and totals** (training grazer; seed 0 / seed 1):

| hunter | layout | rate | catches | living hunter-steps | starved per hunter-step | worlds with hunters extinct by 500 (median step) |
|---|---|---|---|---|---|---|
| arm 1, home | home | .0449 / .0455 | 159,613 / 162,268 | 3.56M / 3.57M | .0104 / .0104 | .02 / .02 |
| arm 1, home | office, held out | .0318 / .0331 | 17,525 / 20,874 | 0.55M / 0.63M | .0228 / .0226 | .99 (201) / 1.00 (209) |
| arm 2, continuation | home | .0471 / .0492 | 186,281 / 194,502 | 3.95M / 3.95M | .0056 / .0051 | 0 / 0 |
| arm 2, continuation | office, held out | .0308 / .0323 | 15,102 / 18,579 | 0.49M / 0.57M | .0234 / .0229 | .99 (185) / 1.00 (206) |
| arm 3, office | home | .0390 / .0404 | 72,773 / 91,660 | 1.86M / 2.27M | .0166 / .0157 | .35 (352) / .25 (341) |
| arm 3, office | office, held out | .0462 / .0460 | 118,487 / 113,592 | 2.56M / 2.47M | .0170 / .0171 | .49 (258) / .50 (239) |
| arm 3, office | office, its training layouts | .0469 / .0467 | 127,072 / 123,110 | 2.71M / 2.63M | .0165 / .0166 | .49 / .43 |
| random | home | .0107 / .0099 | 1,843 / 1,653 | 0.17M / 0.17M | .0272 / .0277 | 1.00 (79) / 1.00 (79) |
| random | office, held out | .0132 / .0140 | 2,465 / 2,680 | 0.19M / 0.19M | .0268 / .0266 | 1.00 (88) / 1.00 (88) |

**Reported, not bars:**
- **Arm 2's own transfer loss:** office/home .653 and .657 [.739, .680]. More home practice makes the hunter a little better outdoors and no better, or slightly worse, in the office.
- **Recovery share:** .94 and .81 [1.15, .95]. Office training won back most of the continued hunter's office loss, and against the held-out grazer on seed 0 all of it. In the office, arm 3 reaches .98x and .94x of arm 2's rate on open ground, and 1.03x and 1.01x of arm 1's.
- **Home retention:** arm 3 at home over arm 2 at home was .828 and .821 [.838, .897], and over arm 1 .870 and .890 [.826, .919]. That is above the spec's forgetting line of 0.8 on both seeds, but close. Office-trained hunters died out by step 500 in 25 to 35% of open-ground worlds, against 0 for arm 2.
- **Seen and held-out layouts:** arm 3 caught 1.014 and 1.017 [1.016, 1.022] as much on its 16 training layouts as on the 16 held-out ones. So the gain carries to offices it never saw, and it isn't memorised layouts.
- **Random hunters catch more in the office than at home:** 1.23x and 1.41x. So the office is not harder for hunting as such. It is harder for the home strategy.
- **Grazers starve in the office.** These grazers are the frozen home brains. At home, grazer deaths roughly equal catches (159,762 against 159,613 on seed 0), so starvation is near nil.
  - In the office with arm 1 hunters, grazers died 235,731 times against 17,525 catches, so about 218,000 starved. Births kept up (244,527).
  - With random hunters, it was 204,536 deaths against 2,465 catches.
  - Living grazer-steps in the office are about two thirds of home's (8.2M against 12.1M). With arm 3 hunters they are about two fifths (5.0M).

### Prediction against the result

- **Loss:** I said 20 to 35%. It was 27 to 29% (26 to 29% held out). Right.
- **Recovery:** I said partial, about 1.3×, with a share of a third to two thirds. It was 1.42 to 1.50× with a share of .81 to .94, close to full. Wrong: it was larger than I predicted.
- **Home retention:** I said at least 0.9. It was .82 to .83 against arm 2 (.84 and .90 held out). Wrong: office training cost more skill outdoors than I predicted.
- **The corner effect** I named as the main risk did not turn the loss into a gain for trained hunters. It may be why random hunters do better indoors (inferred, below).

### Inferred (my reading, not tested)

- **Some of the home hunter's office loss may be scarcer prey, not lost skill.** The frozen home grazers starve in the office and there are about a third fewer of them. That doesn't explain the whole effect, because random hunters, facing the same thinner grazers, catch more indoors, not less. So I read the loss as mostly the home strategy failing in corridors, but the two are not separated here. A test with office-trained grazers, or the same grazer density in both places, would separate them.
- **Why grazers starve indoors.** A guess: they get stuck in rooms whose plants they eat out, because plants spread only to neighbouring open cells and walls cut rooms off. Not measured.
- **What the office hunters learned isn't shown.** Nothing here shows that they route through doorways. The tests show only that they catch more indoors and less outdoors. Movement through doors was not measured.
- **The trade-off.** Retraining bought the office at a cost of about a sixth of home skill. With one budget and one learner, I can't say whether a hunter could hold both.

### What can and can't be said

Measured, allowed on screen:
- "Hunters trained outdoors caught about 28% less in an office they'd never seen."
- "Hunters retrained in other offices caught about 1.4 to 1.5 times as much as hunters given the same practice outdoors, in offices neither had seen, and got back nearly all of what was lost."
- "They lost some of their outdoor skill doing it, about a sixth."

Not earned:
- "they learned to use doors"
- "they remember the layout" (the held-out layouts rule that out as an explanation, and nothing shows memory)
- any claim about an individual hunter's brain. Every hunter in a run shares one brain.

Caveat to carry with the first line: the office also starves the grazers, so the outdoors-to-office comparison is not against the same prey density.

### Cost

- **Pilot:** 1 run, 700 s.
- **Full stage:** 4 runs, 635 to 675 s each, about 45 minutes on the second Spark between 06:58 and 07:43 UTC.
- **Tests:** 48 test files in about 5 minutes.
- **Total:** about 1 node-hour, inside the spec's estimate of 1.5. The Spark was busy with stage 21b's trainer throughout, so everything ran on the second Spark.

### Replays for the studio (illustration, not evidence)

- **Made by:** `scripts/lineage.py`, 8 episodes × 500 steps at evaluation seed 12345.
- **Setting:** held-out layout 1000 (22% walls), against each seed's training grazer.
- **Where:** `runs/lineage/of22/of22-<arm>-s<seed>/`, with arm `home` (arm 1), `cont` (arm 2) or `office` (arm 3). These are gitignored and local.
- **What each folder holds:**
  - `episode3d.json`: the episode with the most dramatic line, which the 3-D viewer plays with walls
  - `lineage.json`: all 8 episodes
- **Rankings:** `runs/lineage/of22/lines.md`. Cause of death is exact in all six.
- **The replay run folders** (config plus combined checkpoint) are in `runs/of22replay/`.

| replay | episode | hunters who lived | the line to follow |
|---|---|---|---|
| `of22-home-s0` | 3 | 63 | Basalt, 15 descendants, 71% of hunters at step 155, gone by 188. Flint (episode 3) takes all the hunters at 188, and they die out by 298 |
| `of22-home-s1` | 2 | 53 | Ash, 9 descendants, gone by 159 |
| `of22-cont-s0` | 3 | 80 | Ember, 13 descendants, gone by 155. Flint reaches 100% at 155 and dies out at 222 |
| `of22-cont-s1` | 6 | 76 | Slate, 33 descendants, takes all the hunters at 173 after a comeback, gone by 234 |
| `of22-office-s0` | 2 | 247 | Shale, 64 descendants, 69% of hunters, lasts to step 468 |
| `of22-office-s1` | 0 | 258 | Jasper, 104 descendants, takes all the hunters and is alive at step 500. Ember rises to 81% and falls |

The contrast is the same as in the tests. Home and continued hunters die out in the office by about step 200 to 300. Office-trained hunters' families last most or all of the episode, with three to four times as many hunters ever alive.

**For the studio:**
- These names are labels that `lineage.py` gives to founders in order within each replay. This Flint and Slate are not stage 20's Flint and Slate (`pp-hof-a50-0`, an individual-brain run), and not their descendants.
- Every hunter in a stage 22 run shares one brain. So a line is a family of bodies under one shared brain, not a mind that learned the office.
- Say "the Slate family took over", not "Slate learned to use doors".

## Follow-up: stage 22b (2026-11-05)

The prey-scarcity confound on bar 1 was tested in `2026-11-05-office-prey-design.md`. With grazers that can't starve, prey numbers are level indoors and out (.998). Arm 1's office/home ratio is 0.644 and 0.681 [0.695, 0.652], so the drop is not just scarce prey. Bar 2 under the same rule is 3.2× and 2.9×. The caveat line above is superseded by 22b's allowed line.
