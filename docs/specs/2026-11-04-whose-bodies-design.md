# Stage 20b: whose changing bodies held evolution back, and does it keep improving?

Date: 2026-11-04. Written before any of these runs or evaluations.

## Why

Stage 20 froze every body gene of both species as a method fix. Its unanchored arm D, evolution alone from random brains, then reached 1.77x and 2.16x random against held-out trained grazers, where stage 17's matching arm, with bodies evolving, sat at about 0.93x and 0.99x. The interpretation review (review, not published) rejects the causal claim "changing bodies held evolution back" as it stands, for three reasons:

1. **The two tests used different hunter bodies.** D's hunters were tested in the founder body. Stage 17's were tested in their own evolved bodies, which are close to blind: sight gene .094 and .066 against the founder's .5, speed .12 and .13 against .2 (last training row of `pp-evo-full-0`, `-1`). They pay .53 to .54 a step in upkeep, against .873 for the founder body. So D's advantage could sit in the evaluation body, not the brain.
2. **Both species were frozen at once,** so whose bodies mattered is unknown.
3. **Training catches rising isn't transferable skill rising.** That needs successive checkpoints tested against the same fixed opponents.

This stage does what the review asks, in its order: a cheap evaluation-only diagnostic (part 1), then a matched two-by-two (part 2) with checkpoints.

## Part 1: same-body diagnostic (evaluation only)

**Question:** in the same hunter body, against the same grazers, do D's brains still beat stage 17's?

Brains: stage 17's full arm `pp-evo-full-<s>` and stage 20's arm D `pp-vis-D-<s>`, seeds 0 and 1. Grazers: the held-out pools, `pp-cul-genes-<1−s>`, as in stage 20. Test settings exactly stage 20's: `scripts.cross_indiv`, 64 worlds of 500 steps, no restarts, hunters frozen (children copy the brain), `--world-set gene_sigma=0 --world-set plant_sigma=0.05`, usual upkeep. Evaluation seeds: the six of stage 20's noise measurement, 12345 and 1 to 5.

| brains | bodies | how |
|---|---|---|
| D | founder | `--bodies runs/pp-vis-D-<s>` (every gene is the founder value) |
| stage 17 | founder | `--bodies runs/pp-vis-D-<s>` |
| random actions | founder | `--random --bodies runs/pp-vis-D-<s>` |
| D | stage 17's evolved | `--bodies runs/pp-evo-full-<s>` |
| stage 17 | stage 17's evolved | stage 20's noise runs (`noise_pp-evo-full-<s>_<e>.json`), same command; reused, not rerun |
| random actions | stage 17's evolved | `--random --bodies runs/pp-evo-full-<s>` |

The main number is the **same-body ratio** R_same = D / stage 17, both in the founder body, catches per hunter-step averaged over the six evaluation seeds. Beside it: the own-body ratio R_own = D (founder body, its own) / stage 17 (its own evolved body), on the same six seeds, and the **share kept**, ln R_same / ln R_own. The reverse cross, D / stage 17 both in stage 17's body, is reported as R_17body.

**Noise:** stage 20's s_eval = .031 (pooled SD of the log of a paired ratio on one evaluation seed), so the band is e^(2s) = 1.064. Using it on a six-seed mean is conservative.

**Readings, fixed now:**
- **The advantage is in the brains:** R_same at least 1.5 on both seeds and share kept at least 0.75.
- **The advantage was the evaluation body:** R_same within the band of 1 (at most 1.064) on both seeds.
- **Both:** anything between; the share kept says how much is brains.
- **Secondary, brain and body as a package:** if R_same is large but R_17body is within the band of 1, D's brains are better only in a body that can see, so the comparison of brains can't be separated from the body they're tested in. Reported, not a bar.

## Part 2: a matched two-by-two

**Question:** whose changing body held evolution back, hunters', grazers', both or neither, and in the frozen world does transferable skill keep rising?

All cells: `tiny_stealth`, 4,702 weights a brain, 64 worlds, 1,000 updates, seeds 0 and 1, `darwin`, unanchored (grazers co-evolve, as stage 17 and arm D), usual upkeep (`pred_base_cost` .5), random founders (no `--founders`; the same random brains as stages 17, 19 and 20, since the seeds are the same), mutation on all weights, sigma .03. **Identical initialisation and plants in every cell:** `--world-set founder_noise=0 --world-set gene_sigma=0.05 --world-set plant_sigma=0.05`, so every animal of both species starts in exactly the founder body, every plant starts uniform, and plants mutate at .05 in every cell. The cells differ only in `--freeze-bodies` (new, `agents/freeze.py`): after every step the named species' body genes are set back to the founder body, so a child never acts in a mutated body. Gene mutation still draws its random numbers, so the cells share every random stream; freezing both this way gives the same worlds as `gene_sigma=0` (tested in `tests/test_freeze.py`).

| cell | hunter bodies | grazer bodies | `--freeze-bodies` | run |
|---|---|---|---|---|
| **N, neither** | evolve | evolve | none | `pp-bf-N-<s>` |
| **H, hunters frozen** | founder | evolve | pred | `pp-bf-H-<s>` |
| **G, grazers frozen** | evolve | founder | prey | `pp-bf-G-<s>` |
| **B, both** | founder | founder | both | `pp-bf-B-<s>` |

All four are run fresh. N isn't stage 17: stage 17 started bodies spread by `founder_noise` .05 and plants at `gene_sigma`; N starts them uniform like every other cell. B should repeat stage 20's D on the same seed; a difference outside the band is reported as a check on the machinery (seed 1 runs on a different node from D).

Every run keeps its population at updates 200, 400, 600 and 800 (`--keep-at`), and 1,000 is the final one.

### Test

As part 1: held-out grazers `pp-cul-genes-<1−s>`, the stage 20 test settings, **all hunters in the founder body** (`--bodies runs/pp-vis-D-<s>`), whatever body they evolved, so the cells are compared in common bodies.

- **Final (update 1,000):** six evaluation seeds, 12345 and 1 to 5; the mean is the cell's number.
- **Checkpoints (200, 400, 600, 800, and 1,000):** evaluation seeds 12345, 1 and 2, against the same fixed grazers. The 1,000 point in the trend uses the same three seeds, so every trend point is like for like.
- Reported, not bars: N's and G's hunters in their own evolved bodies at update 1,000 (seed 12345); hunter body genes at the end of every run; hunters per world, hunter generations and training catches per hunter-step by window of 100 updates.

References: random actions in the founder body against the same grazers (from part 1), and trained genes hunters in the founder body (.0424 and .0437 at evaluation seed 12345, stage 20).

### Bars (can fail)

Ratios of held-out catches per hunter-step, founder bodies, final six-seed mean, per seed.

1. **Changing bodies held evolution back (total):** B / N at least **1.25** on both seeds.
2. **Hunters' changing bodies held it back:** H / N and B / G each at least **1.25**, on both seeds (the hunter-body effect with grazer bodies evolving and with them frozen).
3. **Grazers' changing bodies held it back:** G / N and B / H each at least **1.25**, on both seeds.
4. **Transferable skill keeps rising in the frozen world:** in B, catches at update 1,000 at least **1.10x** those at update 600 (three-seed means), on both seeds. Stage 20 claimed nothing about D's trend; this tests whether it kept improving or levelled off.

Reported beside the bars: the interaction (B x N) / (H x G), each cell against random and against trained hunters, and the trend for every cell, including B at 1,000 against 200.

**Noise rule, as stage 20.** For a ratio against a bar t on one seed: past the bar is a pass; short of the bar but within e^(2s) of it is **no result**; further short is a fail. s = .031 (s_eval) for every bar, so e^(2s) = 1.064. A pass within 1.064 of the bar is "pass, inside noise". A bar passes when every ratio in it passes on both seeds, fails when any fails, and is otherwise no result.

| bar | pass | no result | fail |
|---|---|---|---|
| 1 to 3 | at least 1.25 | 1.175 to 1.25 | under 1.175 |
| 4 | at least 1.10 | 1.034 to 1.10 | under 1.034 |

A ratio within 1.064 of 1.0 is also called "no measurable difference". The band is evaluation noise only. Two training seeds can't measure run-to-run noise between independent evolutions; a big seed-to-seed difference is reported as that.

## Prediction

**Part 1:** the advantage is in the brains. Stage 17's brains evolved in near-blind bodies and use no sight they never had, so in the founder body I expect them near random (about 1.0x), and R_same about 1.8 and 2.1, share kept at least 0.75. In stage 17's near-blind body I expect D's brains to lose most of their edge (R_17body about 1.1 to 1.2), since what they learned needs sight.

**Part 2:** the hunters' bodies are what mattered. In co-evolution hunters trade sight for cheaper upkeep, go nearly blind, and their brains then specialise in bumping into nearly blind grazers, which doesn't carry to trained grazers. So bar 1 passes (B / N about 1.8), bar 2 passes (H / N and B / G about 1.6), bar 3 fails (G / N and B / H about 1.1). N lands near random in the founder body (about 1.0 to 1.1x), B near D (1.77x, 2.16x), H about 1.6x, G about 1.1x. Bar 4 is my least confident call: I expect B to have risen steeply to about update 600 and then slowly, about 1.08x from 600 to 1,000, so no result.

## What each pattern would mean

- **Part 1 says brains, bar 1 passes:** with initialisation matched and bodies common at test, freezing bodies builds better brains. That's the review's claim 3, now measured.
- **Part 1 says evaluation body:** D's advantage over stage 17 was largely the body it was tested in; part 2 is then the only fair comparison, and bar 1 is the whole answer.
- **Bars 1 and 2 pass, 3 fails:** the hunters' own changing bodies held them back; grazers' bodies don't matter measurably. On screen: "When we stopped the hunters' bodies changing, evolution built better hunters."
- **Bars 1 and 3 pass, 2 fails:** the moving target was the grazers' bodies, a Red Queen through bodies, not brains.
- **Bar 1 passes, 2 and 3 both pass:** each species' changing body holds evolution back; the interaction says whether they add.
- **Bar 1 passes, 2 and 3 both fail or no result:** only freezing both helps; neither alone does. Report the interaction; that's an interaction, not a single cause.
- **Bar 1 fails:** with matched initialisation and common test bodies, freezing bodies doesn't build better brains; stage 20's D versus stage 17 was the evaluation body, the initial spread, or both. The review's rejection of claim 3 stands, and becomes a finding.
- **Bar 4 passes:** transferable skill was still rising at 1,000 updates in the frozen world; next, run longer. **Fails:** a new plateau in the frozen world, at whatever level B reached.
- **Any bar no result:** say so, with the ratio and band, and don't count it either way.

## Commands

Code runs on the nodes from a snapshot of this commit (`git archive` plus nothing else) in `~/autolearn-20b`, not from `~/autolearn`, so other stages' work in progress can't reach these runs. Its `runs/` holds hard links to the input runs on the second Spark and copies of the three needed on the Spark. Seed 0 runs on the second Spark, seed 1 on the Spark, so every within-seed comparison is on one node. Two trainers at a time per node, through `deploy/remote.sh` and its memory guard, plus the part 1 evaluations on the second Spark.

```
EVAL_GB=5 bash deploy/remote.sh train_indiv --world tiny_stealth --mode darwin --run-id pp-bf-<cell>-<s> --seed <s> --updates 1000 --num-envs 64 --sigma 0.03 \
  --world-set founder_noise=0 --world-set gene_sigma=0.05 --world-set plant_sigma=0.05 --freeze-bodies <none|pred|prey|both> --keep-at 200,400,600,800
EVAL_GB=5 bash deploy/remote.sh cross_indiv --hunters runs/pp-bf-<cell>-<s> [--hunter-ckpt pop_u0200.npz] --grazers runs/pp-cul-genes-<1-s> \
  --world-set gene_sigma=0 --world-set plant_sigma=0.05 --bodies runs/pp-vis-D-<s> --eval-seed <e> --out bf_<cell>-<s>_<ckpt>_<e>.json
```

Part 1: `--hunters runs/pp-vis-D-<s>` or `runs/pp-evo-full-<s>` (or `--random`), `--bodies` as in the table, `--out sb_<brains>_<bodies>-<s>_<e>.json`.

## Result (2026-11-04): Part 1 says brains. Freezing both species' bodies passes; freezing either one alone fails. Skill kept rising

Code snapshot 8dec5ea in `~/autolearn-20b` on both nodes; seed 0 and part 1 on the second Spark, seed 1's part 2 on the Spark. Queues `queue-20b-part1.sh` and `queue-20b-train.sh` there, logs `~/autolearn-20b-*.log`. Results sit in that snapshot's `runs/` (`pp-bf-*`, and `sb_*` and `bf_*` beside the held-out pools `pp-cul-genes-*`), not in `~/autolearn/runs`. All 220 test files are present. The second Spark's memory guard refused five tests while another stage's trainer was running, and each retry succeeded. Every founder-body test charged the founder upkeep, .8733 a step.

### Part 1: same-body diagnostic

Held-out grazers, catches per hunter-step, means over the six evaluation seeds:

| seed | body | D brains | stage 17 brains | random | D / stage 17 |
|---|---|---|---|---|---|
| 0 | founder | .0252 (1.82x random) | .0129 (0.93x) | .0138 | **R_same 1.96** (per evaluation seed 1.83 to 2.11) |
| 0 | stage 17's evolved | .0153 (1.10x) | .0126 (0.91x) | .0138 | R_17body 1.22 |
| 1 | founder | .0305 (2.11x) | .0130 (0.90x) | .0144 | **R_same 2.34** (2.26 to 2.55) |
| 1 | stage 17's evolved | .0178 (1.28x) | .0137 (0.98x) | .0140 | R_17body 1.30 |

R_own (D in its own body against stage 17 in its own) is 2.01 and 2.22. **Share kept is 0.96 and 1.06, so the advantage is in the brains.** By the rule fixed above, R_same is at least 1.5 and the share kept at least 0.75, on both seeds. Stage 17's brains are at random level in either body. D's brains keep a smaller edge in stage 17's near-blind body (1.22x, 1.30x, both outside the band of 1). So the package reading doesn't apply: they're better there too, by less.

### Part 2: the matched two-by-two

Held-out grazers, all hunters in the founder body, update 1,000, means over six evaluation seeds:

| cell | seed 0 | seed 1 | x random |
|---|---|---|---|
| N, neither frozen | .0143 | .0169 | 1.03x, 1.17x |
| H, hunter bodies frozen | .0164 | .0167 | 1.19x, 1.16x |
| G, grazer bodies frozen | .0149 | .0146 | 1.08x, 1.01x |
| B, both frozen | .0252 | .0305 | 1.82x, 2.11x |

| ratio | seed 0 | seed 1 | per evaluation seed | bar reading |
|---|---|---|---|---|
| B / N | **1.77** | **1.80** | 1.69 to 1.89, 1.74 to 1.87 | pass |
| H / N | 1.15 | 0.99 | 1.06 to 1.27, 0.93 to 1.05 | fail |
| B / G | 1.70 | 2.09 | | pass |
| G / N | 1.04 | 0.86 | 0.99 to 1.06, 0.82 to 0.89 | fail |
| B / H | 1.54 | 1.82 | | pass |
| interaction (B x N) / (H x G) | 1.47 | 2.11 | | reported |

1. **Changing bodies held evolution back (total): PASS.** B / N = 1.77 and 1.80, past 1.25 and far outside the band.
2. **Hunters' changing bodies held it back: FAIL.** B / G passes (1.70, 2.09), but H / N is 1.15 and 0.99, under the fail line of 1.175 on both seeds. Seed 1's is no measurable difference.
3. **Grazers' changing bodies held it back: FAIL.** B / H passes (1.54, 1.82), but G / N is 1.04 and 0.86. Seed 0's is no measurable difference, and seed 1's is lower than N.
4. **Transferable skill keeps rising in the frozen world: PASS** (seed 0 inside noise). In B, update 1,000 against 600 is 1.146 and 1.246 (bar 1.10; seed 0 is under 1.17, so it's inside noise).

**Checkpoints, held-out, founder bodies, means over evaluation seeds 12345, 1 and 2, updates 200 / 400 / 600 / 800 / 1,000:**

| cell | seed 0 | seed 1 |
|---|---|---|
| N | .0155 .0161 .0156 .0148 .0143 | .0155 .0147 .0143 .0143 .0170 |
| H | .0168 .0158 .0142 .0167 .0160 | .0166 .0162 .0159 .0190 .0165 |
| G | .0136 .0148 .0159 .0145 .0148 | .0146 .0145 .0141 .0147 .0143 |
| B | .0176 .0184 .0216 .0242 .0247 | .0167 .0192 .0244 .0268 .0304 |

B rises at every checkpoint on both seeds (1,000 against 200: 1.41x, 1.82x). N, H and G stay within about 1.0x to 1.2x random at every checkpoint, with no trend.

**Reported beside the bars:**
- **Check on the machinery:** B repeats stage 20's D to the printed precision on both seeds and on both nodes. Held-out .0252 and .0305, the same as D in part 1. Hunter generations are 661 in both, and training windows match.
- **Own bodies:** N's and G's hunters in their own evolved bodies (evaluation seed 12345) catch .0129, .0160 and .0138, .0144, also at random level.
- **Bodies at the end of training:**
  - Wherever hunter bodies evolved (N, G), hunters went nearly blind: sight .068 to .075, speed .11 to .12, as in stage 17.
  - Wherever grazer bodies evolved (N, H), grazers went nearly blind and fast: sight .11 to .15, speed .30 to .41.
- **Training, catches per hunter-step, updates 1-100 → 901-1,000:**

  | cell | seed 0 | seed 1 |
  |---|---|---|
  | N | .029 → .033 | .027 → .033 |
  | H | .029 → .040 | .028 → .039 |
  | G | .028 → .033 | .028 → .036 |
  | B | .026 → .043 | .026 → .048 |

- **Hunters a world, last 100 updates:** N 30, 30; H 17, 14; G 30, 30; B 25, 29.
- **Total catches, six evaluation seeds:** N 4,095 and 5,406; H 5,147 and 5,320; G 4,372 and 4,240; B 12,532 and 23,061.
- **Both species alive at step 500:** 0 in every cell except B (.008, .042).

**My prediction:**
- Part 1: right. I predicted the advantage was in the brains, R_same about 1.8 and 2.1 (got 1.96, 2.34), and stage 17's brains near random in the founder body (0.93x, 0.90x). R_17body of about 1.1 to 1.2 was slightly low (got 1.22, 1.30).
- Bar 1: right (predicted about 1.8; got 1.77, 1.80).
- Bar 2: wrong. I predicted the hunters' bodies were what mattered, with H / N about 1.6. H was barely better than N (1.15, 0.99).
- Bar 3: right that it fails (predicted about 1.1; got 1.04, 0.86). But I predicted it as half of a hunter-body story, and the real pattern is an interaction.
- Bar 4: wrong in the cautious direction. I predicted no result at about 1.08; it passed (1.15, 1.25).
- N near random: right. B near D: right. H about 1.6x random: wrong (1.19x, 1.16x).

**The pattern is the "only freezing both helps" reading fixed above.** Freezing either species' body alone did nothing clear of the bar. Freezing both gave hunters about 1.8x as good, in the same body against the same unseen grazers. This is an interaction, not a single cause.

**What's measured:**
- In the same body against the same unseen grazers, stage 20 D's brains catch about twice what stage 17's do. So D's advantage over stage 17 was in the brains, not the evaluation body.
- With initialisation, plants, random streams and test bodies all matched, freezing both species' bodies gave hunters 1.77x and 1.80x as good as letting both evolve.
- Freezing only the hunters' bodies, or only the grazers', gave no gain past the bar against the all-evolving cell (H / N 1.15, 0.99; G / N 1.04, 0.86).
- In the both-frozen cell, skill against fixed unseen grazers rose at every kept checkpoint up to update 1,000. From 600 to 1,000 it rose 1.15x and 1.25x.
- In the other three cells it stayed near random at every checkpoint.

**What's inferred, not tested:**
- *Why one moving body is enough to stop it.* Whichever species could change its body did so the same way, toward near blindness: hunters to sight about .07, grazers to about .13 and faster. A world where either side is nearly blind may be a different game from the test, where trained grazers keep their pool bodies and hunters have the founder body's sight. Then brains that win at home don't carry over.
  - H's hunters kept full sight but faced blind, fast grazers, and still caught more at home (.039) without transferring.
  - That fits, but no run separates it. A test against grazers frozen at the co-evolved bodies would.
- *Whether it keeps rising past 1,000 updates.* Only two seeds were run, and the rise from 800 to 1,000 on seed 0 was small (.0242 → .0247).

**Limits:**
- Two seeds, each one fixed random start shared by all four cells.
- The band is evaluation noise only. Run-to-run noise between independent evolutions isn't measured, though seeds 0 and 1 agree in direction on every bar.
- One sigma (.03), 1,000 updates.
- The frozen body is the founder body for both species, which isn't the body any trained grazer had.
- Test grazers keep their own pools' evolved bodies.
- Training metrics show a small gene spread in frozen species (about .004), because each step's newborns are counted before the clamp. Saved genes and test upkeep confirm the frozen bodies were exactly the founder's.
