# Roadmap: growing and evolving intelligence

Written 2026-10-14, after pp-001 and while pp-evo-001 was being prepared. This is a proposal: each
stage becomes its own sub-project, with a spec, a plan and a test that can fail, as sub-projects 1
and 2 did.

## What carries forward from run to run

A run leaves four things behind. Each stage below should reuse all four.

| Artefact | Where it lives | How a later world reuses it |
|---|---|---|
| **Brains** (PPO weights) | `runs/<id>/checkpoints/*.msgpack` | Weight transplant when the input layout matches; distillation (teacher to student) when it doesn't |
| **Bodies** (gene pools) | `runs/<id>/checkpoints/genes_*.npz` | Seed a new world's founders, with no need to re-evolve from founder genes |
| **Opponents** | every checkpoint | A league of past brains to train against, so new skills don't erase old ones |
| **Tests** | `scripts/eval.py`, `eval.json` | Every new world keeps the old tests and adds its own |

Today, brains only transfer between worlds whose observations match. pp-001's brain reads 9×9×4;
the evo world's reads 9×9×8. Making brains portable is the first piece of work below.

## Lessons so far

1. **Equal bodies on a wrap-around grid stalemate.** In pp-001 the prey learned to dodge almost
   perfectly within about 6 minutes. Predators then starved within 30 steps and learning stopped.
   Intelligence only keeps growing while the world keeps producing problems. Asymmetry (speed,
   terrain, group hunting) and moving targets (co-evolving opponents) are what keep the pressure
   on.
2. **Selection takes whatever is free.** With food plentiful and no cost to giving birth, breeding
   thresholds collapsed while sight, the trait that costs energy, faded. Every trait needs a real
   cost, and every test has to show which trait carries the advantage, gene by gene.
3. **Tests can pass for the wrong reason.** The lineage test passed with random brains before
   anything had trained. Control conditions (random brains, founder bodies, single-gene swaps) are
   part of every test from now on.

## The road

### Stage 3: portable brains and a league (in progress)

- **One body interface.** A fixed registry of sensor channels, where each channel always means the
  same thing and missing ones read zero, plus a superset action space. Any checkpoint then loads
  into any world. Add `--init-from` for warm starts and a distillation script for architecture
  changes.
- **A league.** Train each species against a pool of frozen past opponents, as AlphaStar did, and
  track Elo. This breaks stalemates and stops forgetting.
- **Terrain.** Walls, burrows only prey fit into, and water. This ends the open-torus symmetry
  behind lesson 1.
- **Test:** a warm-started brain reaches pp-001's level in the new world in less than half the
  steps it takes from scratch, and Elo against the league keeps rising.

**Progress, 2026-10-14** (15-minute runs, at the user's request):

- *Built:* the terrain world (walls, and burrows only prey can enter); fixed 10-channel observations;
  `--init-from` with zero-padded inputs, so older brains load unchanged; a burrow test; Elo over a
  run's checkpoints (`scripts/league.py`); league training (`--league N`); `--set` world overrides;
  living-first PPO sampling, which doubled learning speed.
- *Found:*
  - Carried-over brains start far ahead in a new world, so learning persists.
  - Walls help predators: they catch 3–4× as often as in the open.
  - Burrow use has not emerged in 15 minutes.
  - Elo shows prey improving and predators not, even declining. That is the reason to try league training next.
- *Not done:* distillation for changes to the network's shape; water.
- *Follow-ups (30-minute runs):*
  - pp-terrain-003 had 64 burrows and started from pp-terrain-001's brains. It failed the burrow
    test again: trained prey hid 1.05× as much near a predator, against 1.20× for random prey.
  - pp-evo-004-league trained a quarter of its worlds against past brains. It passed both evolution
    tests but did not fix the predators.
  - Elo was re-rated with the bodies each brain trained with. Predators never reached the +100 bar
    in any run (−11 to +77), while prey gained +200 to +280. pp-001, with no evolution at all, shows
    the same.
- *Verdict (stage closed):* carrying brains over works. The league does not stop the predators
  stalling, and burrow use did not emerge. Why predator brains stop improving is an open question,
  carried forward.
- *Leads on the predator stall (from the training logs):*
  - In pp-001, predators almost stopped breeding: about 0.01 births per world-step early on, falling to
    0.0001 by update 625. (Corrected 2026-10-29; this line once said zero births.) Their only rewards were +0.01 for
    each step alive and −1 for dying.
  - In the evo runs, predator births fall over training (0.19 → 0.07 per step), their reward stays
    flat, their policy entropy falls (1.4 → 0.6), and their sight evolves toward blindness.
  - Hypotheses:
    1. A ceiling. At equal speed, geometry caps the catch rate, and predators reach it by update 25
       (lesson 1).
    2. Credit assignment. The reward is too thin to keep improving the hunt.
  - The test that tells them apart: train predators against prey that move at random and never
    learn. If predator Elo still stays flat, it is credit assignment, not the prey.
  - *Done 17:41 UTC:* pp-pred-solo (`--learners pred`, prey frozen at their random start) against
    pp-pred-both (both learn), 30 minutes each on the evo world.
    - Solo predators improved until about update 58 (reward 0.0051 → 0.0072), then went flat:
      Elo +52.
    - In the normal run, predators gained +92 and prey +357.
    - So prey adapting is not what stops the predators. They reach their limit within about
      1.4M world-steps even against prey that never learn.
    - Still open: a ceiling (geometry and bodies) or a limit in learning. Next check: a hand-written
      chase rule against the same prey.
  - *Answered 17:55 UTC (`scripts/chase.py`, 48 episodes):* mostly a ceiling.
    - In the evo world, learned predators catch as well as a hand-written "step toward the nearest
      prey" rule, or better: the chaser does 0.83–1.06× as well.
    - In pp-001, the chaser beats the learned predators against random prey (1.45×). Against
      trained prey neither catches anything (0.0001–0.0002 per predator-step).
    - On an equal-speed map that wraps around, trained prey simply escape. More learning will not
      free the predators; the world has to give them a way to win:
      - ambush from terrain;
      - a speed edge;
      - hunting together (Stage 6).

### Stage 4: memory

- **What needs remembering.** Seasons and day–night cycles, food that moves, and predators that
  can't be seen in the dark.
- **Brains with memory.** Recurrent or state-space brains (LSTM or S5) in place of the
  feed-forward network.
- **Test:** a memory brain beats a feed-forward brain on the seasonal world, with bodies held
  equal.

**Progress, 2026-10-14:** built the seasons world (night, and plants taking turns on the two halves
of the map), GRU brains (`--memory H`), recurrent PPO over whole animal histories, and
`scripts/compare.py`, which compares two runs at matched world-steps. pp-mem-001 (GRU, 64 units) and
pp-mem-ff (feed-forward) are training, 30 minutes each, with evolution off.

**Result (stage closed, test FAILED):**

- The first pair (30 minutes) was void. A review found that newborns inherited a dead animal's
  memory.
- The fixed pair, pp-mem-003 against pp-mem-ff3, ran 60 minutes. At matched world-steps the
  memory prey lived 0.99× as long, and the memory predators caught 1.13× as often (target 1.2×).
- Memory predators learn faster per world-step: 1.21× the catch rate at 4.7M steps. The
  feed-forward brain catches up by 7.6M steps, and it runs 1.6× faster, so against the clock the
  two are level.
- Prey are capped by space and food, so this world gives them nothing that memory can win.
- A fair next test needs a world where only memory pays: food that disappears and comes back, or
  predators that hide at dusk.

### Stage 5: a richer ecology

- **Food web.** Three or four levels: plants, grazers, mid-predators, apex predators and
  scavengers.
- **Larger maps.** 128–256 cells across, which needs JAX sharding across the GB10.
- **New body traits:** size, armour, camouflage and metabolism.
- **Sexual reproduction.** Crossover between two parents, which makes speciation measurable
  (clusters in gene space).
- **Test:** at least two stable lineages split out of one founder pool, and the food web survives
  without hand-tuning.

**Progress, 2026-10-14 (Stage 5a):** built a size gene (big animals digest tough plants, only small
prey fit into burrows, predators catch only prey up to their own size plus 0.25, and size costs
upkeep), two-parent breeding with crossover, the `niches` world, and `scripts/species.py`. A
no-learning probe fooled the first version of the split measure, which saw clumps of cloned genomes
as species. The measure now works on the size gene, with a spread floor and a 0.3 size gap.
Spec: `docs/specs/2026-10-14-speciation-design.md`.

v1, v2 and v3 all failed the split test. v1 and v2 found loopholes: big prey were safe, so prey
grew big and predators shrank. v3 made size matter only for feeding, but grass fed every size
fully, so being big never cost a grazer anything at the trough (prey size gap 0.13, one wide
group). v4 (`size_tradeoff`) makes small bodies digest grass best and big ones shrubs, so no size
is best at both. v4 failed as well, and the plant log showed why v3 and v4 never had a chance: the
grass died out within minutes (toughness 1.0 everywhere), so there was only one food. v5
(`kinds_persist`) keeps both kinds. v5 failed too, but without mating two even groups of prey
pulled apart (size gap 0.30 at the end, still widening; D 1.8). With mating, prey moved together
toward small bodies. v6 (60 minutes) was a near miss: without mating, two even size groups held
0.35–0.40 apart for 25 minutes with D 1.92–2.09, failing only on one snapshot at D 1.95. With
strict size matching, prey shrank together. Two foods hold two body sizes apart when lineages
breed alone; two-parent breeding prevents it. The bar stays where it was.

**Stage 5b, meals worth what the prey stored (FAIL twice, closed, 2026-10-30).** From a viewer's comment: a predator's edge is the energy density of its meal. A catch is now worth 1.25x what the grazer held, and hunters can see who is fat. Control: the same sight, with every meal the same. Test (`scripts.meals`, both seeds): hunters eat fatter grazers than the population by at least 1.15x the control's ratio, and meal value falls at least 10% when they cannot see fat. Spec: `docs/specs/2026-10-30-fatprey-design.md`. Result: selectivity 1.10x and 0.97x the control against a bar of 1.15x, and blinding hunters to fat cost 3.4% and -0.2% of a meal against a bar of 10%. Hunters did not choose fat prey. Grazers ended leaner on both seeds, but not because of hunters choosing, and meals here pay about 11 against the control's 25, which confounds that comparison. Spec: `docs/specs/2026-10-30-fatprey-design.md`. That first test was unfair to the idea: I assumed prey held about 20 energy and they held about 9.5. The fair retest (`fatprey_rich`, meals back at about 25, same bars) also failed: selectivity 1.07x and 1.00x, blind drop 2.9% and 1.9%. Grazers vary plenty (energy 9 with an sd of 5), so the difference exists and hunters still do not use it, for reasons not measured. Most of the leaner-grazer effect went away with the confound.

### Stage 6: social intelligence and communication

- **A signal action.** An animal can emit a token that its neighbours see.
- **Kin recognition** from genes.
- **Prey only a group can take down**, so cooperative hunting pays.
- **Test:** signals carry information, measured as mutual information between the signal and a
  hidden fact (predator nearby, food nearby). Groups catch large prey that a lone predator cannot.

**Progress, 2026-10-14 (Stage 6a):** the `pack` world, where a prey is eaten only when two
predators within one cell hunt it together. First runs (pp-pack-001/002): catches per step went
from about 0.1 to 0.7 by update 159, but the test failed: against trained prey, learned
predators catch only 1.02–1.04× as often as lone hand-written chasers (bar 1.5×). The rise came from
evolution (predator speed 0.66, prey 0.14), and chasers that each run to the nearest prey meet
there anyway. Spec: `docs/specs/2026-10-14-pack-hunting-design.md`.

**Stage 6b (FAIL):** predators signal 37–38% of the time, but the signal carries 0.01–0.03 bits
about nearby prey, and blanking it changes catches by under 1% (bar: 10%). With nothing to
coordinate in the pack world, a signal has no job. Built: the `signals` world adds a signal to every move (0.05 energy), seen by the
same species on the next step. `scripts/signals.py` measures signal rate, mutual information with
"the other species is within 2 cells", and the catches lost when predators cannot see signals
(pass: 10% or more on both seeds). Spec: `docs/specs/2026-10-14-signals-design.md`.

**Stage 6c (built):** the `flank` world, where a prey is eaten only when its hunters stand on
opposite sides of it, which two chasers coming up behind it never do. `flank_signals` adds the
signal. Spec: `docs/specs/2026-10-14-flanking-design.md`.

**Stage 6c (PASS on two seeds):** learned predators catch 3.0–3.8× as often as lone chasers
against trained prey, on seeds 0 and 1, in founder and evolved bodies, with and without signals
(bar 1.5×). This is the first stage where learning clearly beats a simple rule. Signals help the
run's evolved predators on both seeds (blanked: −17 to −22%; another world's signals: −27 to
−28%), but with founder bodies only on seed 0. A signal says almost nothing about nearby prey, so
it seems to act as a beacon for packmates.

### Stage 7: brains evolve too

- **Brain size as a gene that costs energy.** Bigger brains cost food, so evolution has to justify
  intelligence.
- **Evolve learning settings** (rates, entropy, network width) with population-based training.
- **Learning within a lifetime** (meta-RL or plastic weights), so each animal learns during its own
  life, not just the shared species brain. This is the Baldwin effect: evolution shapes what is
  easy to learn.
- **Test:** evolved learners adapt to a change in the world (a new predator, a moved food source)
  within one lifetime, faster than fixed brains.

**Stage 7a (built):** a brain gene sets how many of the species brain's 128 hidden units an animal
uses (8 + 120g), at 0.5g energy per step. The test compares the evolved predator brain size in the
flank world, where learning beats a simple chase, with the pack world, where it does not (bar: 0.1
bigger in flank, on two seeds). Spec: `docs/specs/2026-10-14-brain-size-design.md`.

**Stage 7a, first pair:** every brain shrank to the floor in both worlds (predators 0.06 and
0.09, prey 0.07). A code review found why: units no animal uses never train, so a bigger brain
bought untrained units at a higher energy cost. v2 spreads founders' brain genes over [0, 1] and
adds a pair where brains cost nothing, to measure drift and hitchhiking.

**Stage 7a v2 (FAIL):** with spread founders, brains fell to the floor in both worlds again at a
cost of 0.5 (flank 0.06, pack 0.12). With no cost, every brain grew to 0.7–0.8, so brains do help,
but at 0.5 they never pay. The richer pack predators kept more brain, speed and sight. Flank
predators beat chasers 2.4× with about 15 hidden units. v3 tries a fifth of the cost (0.1).

**Stage 7a v3 (FAIL, closed):** at cost 0.1 brains settle mid-range and the worlds differ the
wrong way: predator brains 0.19 in flank and 0.50 in pack. Pack brains rose only after evolved
speed and sight made pack predators rich. Brain size follows food, not the value of thinking.
Not tuned further; a wealth-matched comparison is left for later.

**Stage 7b (PASS on two seeds, closed):** the wealth-matched test 7a left open. A brain tax replaces the
flat cost: an animal with brain gene g keeps `1 - tax * g` of every meal, so a poor flank predator and
a rich pack predator pay the same share of their food for the same brain. Worlds `pack_tax` and
`flank_tax` (tax 0.3, spread founders, no flat cost). Runs at tax 0.3 and 0.1 on seed 0, with the bar
fixed before any run. Result, seed 0: predator brains 0.53 (flank) against 0.18 (pack) at tax 0.3, and 0.69 against 0.38
at tax 0.1: gaps of +0.35 and +0.31, bar +0.12. Seed 1 at tax 0.3: 0.47 against 0.14, a gap of +0.33. With a fair cost, brain size follows the value of
thinking, reversing 7a. Pack brains fell only after pack predators evolved speed and sight.
Spec: `docs/specs/2026-10-29-brain-tax-design.md`. Was blocked
on 2026-10-29 (fixed the same day by regenerating the CDI spec): new GPU containers on the Spark failed because `/var/run/cdi/nvidia.yaml` still names
`libnvidia-egl-wayland.so.1.1.21` after apt upgraded it to 1.1.23. Fix (needs sudo):
`sudo nvidia-ctk cdi generate --output=/var/run/cdi/nvidia.yaml`.

**Stage 7c, learning within one lifetime (PASS on two seeds, 2026-10-30).** Memory brains take 85% / 76% good bites 10-40 steps after a flip, against 64% with no memory; with the
memory wiped they fall to 66% / 60%. After a bad bite the same animal's next bite is good 67% / 55% of the time, against 50% with no memory. No-memory brains also recover, later, so memory's gift is speed. The `fickle` world: one plant kind feeds and the other poisons,
nothing shows which, and it flips at random, so only tasting and remembering can work. A memory brain (GRU, reset at birth) against a
feed-forward brain, matched at 150 updates. Test (`scripts.lifetime`, both seeds): at 10-40 steps after a flip the memory brain's good-bite
share beats the other's by at least 0.10, and beats its own at 0-10 steps by at least 0.05. Spec:
`docs/specs/2026-10-30-lifetime-design.md`.

**Stage 7d, is the map the memory? (FAIL, 2026-10-30).** No-memory brains seemed to recover after a flip, with the good kind climbing to 0.8 around them. Scattering the grazers barely hurt them (0.035 and 0.076, bar 0.15), so location was not the memory. Survivorship explains the plants around them: their plant pools are 85% shrubs, worlds where the rare kind feeds die, and about nine in ten living worlds at 80+ steps are the lucky kind. It explains only 9-52% of the rise in good bites, depending on seed and how the rise is split; the rest happens inside each kind of world and is unexplained. The 7c no-memory curve should not be read as recovery; the memory result stands on the wiped control, which used the same plants. Spec: `docs/specs/2026-10-30-scatter-design.md`.

**Stage 9a, hunters who can eat plants (FAIL on the second bar, 2026-10-30).** Stage 5b showed hunting pays about 0.19 energy per step, well under what hunters burn. A diet gene now splits what a hunter gets from catches and from plants. Diet fell from 0.5 to 0.107 and 0.098, against 0.475 and 0.445 where the gene does nothing, so the first bar passed easily. Plants give them about all their energy, and a fixed chaser catches about 10x what they do. They stopped being hunters. The second bar failed the wrong way: a gut at 0.3 was meant to keep them hunting, and diet went lower still (0.050 and 0.049) while hunters fell from 31.6 to about 5. Even where the gene does nothing, plants are 0.97 of their energy. Spec: `docs/specs/2026-10-30-omnivore-design.md`.

**Stage 9b, how bad must a gut be? (PASS on two seeds, 2026-10-31).** A rough sum said hunting and plants pay the same at a gut of about 0.07. Diet ended at 0.90 and 0.91 at a gut of 0.03 (bar at least 0.50), 0.86 and 0.88 at 0.08, and 0.07 at 0.20 (bar at most 0.25). So a bad enough gut keeps a hunter hunting, and hunting well again: a fixed chaser catches 0.67x and 0.64x what they do. The switch is sharp, between 0.08 and 0.20. The price is numbers: about 2 to 3 hunters per world in evaluation against about 32 at a normal gut. Spec: `docs/specs/2026-10-31-gut-ladder-design.md`.

**Stage 9c, do winters make real omnivores? (FAIL on two seeds, 2026-10-31).** Plants were thin food half the time, so summer paid above the diet switch and winter below it. Hunters went further toward plants instead: diet 0.109 and 0.116, plants 0.93 of their energy (bar 0.20 to 0.80). Winter also cut grazers to about 11 from about 31. The surprise was the no-winter control, where one seed's diet genes formed two groups. Spec: `docs/specs/2026-10-31-winter-design.md`.

**Stage 9d, do hunters split into two kinds? (FAIL by the rule, 2026-10-31, with the project's first real split).** No new rules, fresh seeds, twice as long. At a gut of 0.11 hunters' diet genes split on both seeds, steadily: D 5.92 and 6.73, smaller group 0.33 and 0.39, gap 0.75, where drift reaches only D 2.66. The second bar, both kinds alive at step 400, failed because no world had any hunters by then, a step I chose without checking how long hunters last. An exploratory curve shows they take turns: plant eaters boom and crash by step 100, meat eaters outlast them, and hunters die out by about step 300. So they are two kinds in the pool, not two species side by side. Spec: `docs/specs/2026-10-31-two-diets-design.md`.

**Stage 9e, is the split made by the rich start? (FAIL by the rule, 2026-10-31).** The split repeated in kind (D above 5, gap about 0.75, in all four rich runs), though seed 4's meat group dipped to 0.16 in the last snapshot, under the 0.2 bar. Starting with fewer plants erased it on both seeds (D 1.70 and 1.62, gap about 0.17). My mechanism was backwards: plant eaters keep up to step 50 on a lean start, while the meat group never forms, and grazers never boom (about 24 at step 50 against 95). All hunters are gone by about step 100. Spec: `docs/specs/2026-10-31-bonanza-design.md`.

**Stage 10a, is there a world that lasts? (PASS with a caveat, 2026-10-31).** The first version, bigger meals, was withdrawn before training: its validity check showed grazers die out too, even hand-written ones with plants on the map, and plant cover falls 0.50 to 0.09 whoever is eating. Every world so far has been a boom and bust of about 150-300 steps, hidden by restarts. New test, no training (`scripts.lasting`): hand-written animals over a grid of plant and meal settings, both species alive at step 2,000 in at least half of worlds. All 12 settings with plants spreading at 0.2 pass, and none of the 24 slower ones do, so it is a sharp threshold. But those worlds sit near the slot caps of 96 grazers and 32 hunters, which may be what holds them up. Spec: `docs/specs/2026-10-31-persist-design.md`.

**Stage 10b, do trained brains last too? (FAIL on survival, 2026-10-31).** In the world where hand-written animals last, both species alive at step 2,000 in 0.000 of worlds on both seeds (bar 0.5). The world lasts and the hunters don't: grazers sit at the cap of 96, plants at 0.91 of the map, hunters mostly gone by step 200. Learned hunters still out-catch a fixed chaser (0.75x and 0.65x). Trained grazers flee and hand-written ones don't, which is the likely difference. Spec: `docs/specs/2026-10-31-lasting-brains-design.md`.

**Stage 10c, does a big enough meal let trained hunters last? (FAIL, 2026-10-31).** At a meal of 250, both species alive at step 2,000 in 0.066 and 0.098 of worlds (bar 0.5); at 100, none. Hunters are all-or-nothing: surviving worlds hold about 32, the cap. Learned hunters catch less as meals grow (0.0083 at 100, 0.0054 at 250). A probe swapping in hand-written animals settles the cause on these runs: against hand-written grazers trained hunters last in every world, and hand-written hunters die against trained grazers. Trained grazers are what kill hunters. Spec: `docs/specs/2026-10-31-big-meals-design.md`.

**Stage 10d, can a viewer's idea make a world last? (FAIL, 2026-10-31).** Stage 8b's walking costs and thickets, with plants that last, at meals of 25 and 100: both species alive at step 2,000 in 0.000 of worlds on both seeds, grazers still at the cap. Learned hunters beat a fixed chaser only at a meal of 100 (0.72x and 0.86x), where they catch about twice as often and still die out. Spec: `docs/specs/2026-10-31-costly-dodging-design.md`.

**Stage 10e, do hunters built for the chase last? (FAIL, 2026-10-31).** Speed cost hunters nothing, and they evolved it (0.82 and 0.78, against 0.18 and 0.16), but both species alive at step 2,000 stayed at 0.000 at a meal of 25 and 0.008 at 100. Against my prediction. Spec: `docs/specs/2026-10-31-sprint-design.md`.

**Stage 10f, can grazers dodge what they can't see coming? (FAIL by the rule, 2026-10-31, and the first trained brains in a world that lasts).** Hunters are camouflaged: a grazer sees one only from the next cell. At meals of 25, both species alive at step 2,000 in 0.957 and 0.961 of worlds, where the best before was 0.098. Hunters hold about 30, catch about seven times as often, and beat a fixed chaser (0.64x and 0.66x), and grazers are born and eaten at about 1.3 per step. It fails on grazers sitting at the cap of 96. Spec: `docs/specs/2026-10-31-camouflage-design.md`.

**Stage 10g, with room to move, what sets the numbers? (FAIL by the rule on hunters at their cap, 2026-10-31).** Three times the slots. Both species alive at step 2,000 in 0.984 and 0.996 of worlds, and hunters beat a fixed chaser (0.74x and 0.73x). Grazers now settle at about 113 of 288, so hunters hold them down and no slot limit sets their number, against my prediction. Hunters fill their cap (about 95 of 96), which fails the bar. Spec: `docs/specs/2026-10-31-room-design.md`.

**Stage 11a, do different climates make different animals? (FAIL by the rule on two of three bars, 2026-10-31, and climate clearly shaped bodies).** Tom's idea: three connected worlds of 256 x 256 on two machines, each with its own climate, brains and gene pool: open meadow, forest with thickets and walking costs, tundra with winters. On two sets of fresh seeds, grazer sight came out meadow 0.76 and 0.68, forest 0.32 and 0.32, tundra 0.12 and 0.10, and hunter speed tundra 0.52 and 0.51, meadow 0.29 and 0.31, forest 0.19 and 0.21. Forest's hunters die out with no restarts (0.000), so bar 1 fails. Bar 2, on migrants' descendants, failed, but its measure was flawed (see 11b), so how isolated these worlds were is unmeasured. Tundra's seasons give the first swings in numbers, with hunters far below their cap. Spec: `docs/specs/2026-10-31-continents-design.md`.

**Stage 11b, do climate's differences survive real migration? (FAIL by the rule on a flawed bar, 2026-10-31).** Migrants could replace residents and trainers kept pace. Grazer sight stayed meadow over forest over tundra (0.37 / 0.22 / 0.11 and 0.70 / 0.25 / 0.10). Bar 1 (5% descended from arrivals) failed, but its measure was flawed: every world's restart wiped the lineage tags, so "nearly isolated" in stage 11a is withdrawn. Spec: `docs/specs/2026-10-31-migration-design.md`.

**Stage 11c, how much do migrants really contribute? (FAIL by the rule on meadow's share, 2026-10-31, and the strongest result so far).** Lineage tags now survive restarts. On two sets, 72% and 96% of tundra's grazers descend from arrivals, and every one of forest's hunters does, but only about 10% of meadow's grazers, under the 20% bar. Grazer sight still comes out meadow 0.73 and 0.74, forest 0.24 and 0.23, tundra 0.11 and 0.11. So climate's differences hold against heavy migration, and that order has held on all six sets. Spec: `docs/specs/2026-10-31-lineage-design.md`.

**Stage 12a, does a body part that matters follow climate too? (FAIL on two of three bars, 2026-11-01).** Tom asked for more complexity in the organism, then spotted the result on the phone viewer: hunters collapse. Size on the three continents took meadow and tundra from lasting in every world to none, with hunters down about tenfold. Hunters grow big in meadow and tundra (0.73 to 0.91) and tiny in forest (0.05), where they all but stop hunting. Grazer sight kept its order, more narrowly. Spec: `docs/specs/2026-10-31-bulk-design.md`.

**Stage 12b, why does size kill the hunters? (bars met, 2026-11-01).** The collapse reproduces in the small world (0.000, 0.000). A body at a fifth of the cost barely helps (0.004, 0.004). Taking size out of predation restores lasting worlds (0.984, 0.852), with hunters shrinking to 0.08 where otherwise they climb to 0.92 and grazers follow. Spec: `docs/specs/2026-11-01-why-bulk-kills-design.md`.

**Stage 12c, the failed catch, the meal, or the arms race? (not separated, 2026-11-01).** No arm lasts: certain catches 0.469, 0.434; flat meals 0.000, 0.000; a fixed 0.73 chance 0.219, 0.055. My arms-race prediction wasn't confirmed. Changing one thing at a time, a 27% failure rate with no race and a meal about 11% smaller each do real harm. It also corrects stage 12b: the cheap body's saving was largely taken back by the race. Spec: `docs/specs/2026-11-01-arms-race-design.md`.

**Stage 12d, is it just the hunters' energy budget? (not separated, 2026-11-01).** Paying failed catches back in bigger meals works (0.887, 0.586). With certain catches, grazers shrank to keep meals small (0.570, 0.281). The race arm scored 0.000 twice, but the energy cap of 60 clipped its meals, a flaw I missed, so it didn't test the budget. Spec: `docs/specs/2026-11-01-pay-it-back-design.md`.

**Stage 12e, with the cap out of the way, is it burn or the race? (the race, 2026-11-01).** With meals that fit, no race lasts at low burn (0.996, 1.000) and high burn (0.723, 0.543); the race doesn't, at low burn (0.402, 0.297) or high (0.176, 0.008). The race world has about the same income as one that lasts in every world, so the average budget isn't the whole story, and I predicted all four would last. How the race does its harm isn't shown. Spec: `docs/specs/2026-11-01-burn-or-race-design.md`.

**Stage 12f, can a big enough meal carry the race? (PASS, 2026-11-01).** Meals of 50 last in 0.980 and 0.945 of worlds, and meals of 70 in 0.996 and 0.988, against 0.176 and 0.008 at 37. It's the first world where size decides catches and both species last, with hunters still ahead of the chaser. Both species run to the top of the size range and sit near their caps. Spec: `docs/specs/2026-11-01-price-of-the-race-design.md`.

**Stage 12g, with a race that lasts, does size follow climate? (FAIL on one number, 2026-11-01).** Grazer size follows climate strongly on both sets: meadow 0.89 and 0.90, forest 0.23 and 0.33, tundra 0.13 and 0.15, the opposite of my guess, and hunter size takes the same order. Meadow lasts in every world, and tundra in 0.344 and 0.875, which misses the bar of 0.5 on one set. Forest's hunters are carried by arrivals. Why tundra's grazers end up small isn't tested. Spec: `docs/specs/2026-11-01-feast-continents-design.md`.

**Stage 12h, why do tundra's grazers end up small? (PASS, 2026-11-01).** Winter shrinks grazers from 0.91 to 0.12 on both seeds in the small map, and with their size free they stay at 0.93. So winter plus upkeep is enough to explain it, with no crossing and no big map. Worlds also last far better with big grazers (0.934, 0.844 against 0.098, 0.098). Spec: `docs/specs/2026-11-01-small-in-winter-design.md`.

**Stage 12i, do winter's worlds fade because small grazers are small meals? (PASS, 2026-11-01).** With every meal a flat 71 and grazers just as small (0.12), worlds go from 0.098 to 1.000 on both seeds. But the fixed chaser now beats learned hunters (1.15x, 1.05x), so generous food costs hunting skill. Spec: `docs/specs/2026-11-01-small-meals-design.md`.

**Stage 13a, can camouflage evolve? (FAIL on two of four bars, 2026-11-01).** Free, it evolves quickly from hunters that hide nothing (0.94 on both seeds) and lasts about as well as my hand-set rule (0.746, 0.973). At a price of 0.25, the seeds split: 0.079 and 0.939, lasting in 0.000 and 0.172. Spec: `docs/specs/2026-11-01-camouflage-gene-design.md`.

**Stage 13b, a valley, and what camouflage can cost (FAIL on the valley, 2026-11-01).** Founders past the first step still split at 0.25 (0.937, 0.084), so a one-step valley doesn't explain it. At prices of 0.10 and 0.05 every run evolves the gene and worlds last (0.594, 0.891 and 0.711, 0.781): the first time hiding that evolved, at a price, has carried a trained world. Spec: `docs/specs/2026-11-01-camouflage-price-design.md`.

**Stage 13c, does camouflage follow climate? (FAIL on two of three bars, 2026-11-01).** The gene ends near 0.93 everywhere, so a gap of 0.04, and set K's worlds fade. But it spreads in the same order on both sets, meadow, forest, tundra, at updates 494, 516, 539 and 35, 53, 60, which is the direction animals walk. Lineages aren't traced yet. Set L, which swept early, lasts in every meadow and tundra world. Spec: `docs/specs/2026-11-01-camouflage-climate-design.md`.

**Stage 13d, are brains better at their own climate? (FAIL on both bars, 2026-11-01).** Tom's question. Training is repeatable: a same-climate brain from another run hunts within a few percent of the home brain. In tundra, tundra-trained hunters beat meadow-trained ones by 9% to 22% on all four targets, but meadow shows no steady home advantage, and grazers show none. So climate shapes bodies far more clearly than brains. Spec: `docs/specs/2026-11-01-transplant-design.md`.

**Stage 13e, does camouflage really travel between continents? (PASS, 2026-11-01).** With no crossing, meadow still gets there (updates 83 and 38), and tundra never does in 600 updates on either set (0.086, 0.140), and its worlds die. With crossing, tundra's hunters reach about 0.90. So a continent ended up with a trait it didn't reach alone. Lineages aren't traced, and crossing rescued tundra's worlds on only one set of two. Spec: `docs/specs/2026-11-01-no-crossing-design.md`.

**Stage 14a, individual brains that descendants inherit (FAIL on bars 2 and 3, 2026-11-01).** Tom's idea. Each animal holds its own network of 4,702 weights. Learning in life barely helps (0.98x), and Lamarckian inheritance does harm: 0.98x and 0.94x on catches, and in winter worlds last in 0.516 and 0.547 against more than 0.9. Why isn't settled, and this learning rule may just be weak. Spec: `docs/specs/2026-11-01-individual-brains-design.md`.

**Stage 14b, does evolution keep learning? (FAIL on bar 2, 2026-11-01).** A heritable learning rate. Evolution turns it down (0.264, 0.256 against a control of 0.425, 0.563), but worlds don't fully recover (0.531, 0.797 against a bar of 0.8). Baldwinian's falls too, so it isn't only about inheriting stale lessons. Spec: `docs/specs/2026-11-01-heritable-learning-design.md`.

**Stage 14c, does inherited knowledge build up, given time? (built 2026-11-01, training).** Stage 14a's arms for about 350 generations in place of 50. Spec: `docs/specs/2026-11-01-long-inheritance-design.md`.

**The plan after stage 14c (2026-11-01).** From a review of what's known about genes, memory and learning, at `docs/specs/2026-11-01-memory-and-inheritance-review.md`. Each stage gets its bars fixed in a spec before it runs.

- **Stage 15a, conditional expression.** A camouflage gene expressed only in winter, costing energy only then. Can tundra evolve it alone, which it didn't in stage 13e?
- **Stage 15b, a season input.** Does knowing the season reduce the harm Lamarckian inheritance does in winter?
- **Stage 15c, environmental mutation.** Tom's idea. Mutation rising by place, season or a parent's hunger, against controls with the same average mutation spread evenly. Can tundra reach camouflage alone, and does it beat even mutation?
- **Stage 16, culture.** Juveniles learn from a parent in early life, and no weights are inherited. This is the realistic form of Tom's idea.
- **Stage 17, a compressed genome.** A small genome decoded into a brain, with evolved learning rules. The first rung that's recursive at all, and still bounded by the simulation.

### Stage 8: toward realism

- **Physics.** Move to continuous 2-D, then 3-D, physics in JAX (Brax or MuJoCo MJX), which keeps
  everything GPU-native as now.
- **3-D replay (built 2026-10-29).** `python -m scripts.episode3d --run-dir runs/NAME --evolved
  --tries 12` records one episode with the run's brains and evolved bodies (the longest of 12 seeds)
  to `episode3d.json`. The dashboard's `world3d.html` (three.js, vendored in `viz/dashboard/vendor/`)
  replays it in 3-D: follow an animal, see its genes and sight square, colour by brain gene or energy,
  and show trails. Its hills are a visual-only preview; the agents still live on a flat grid.
- **Stage 8a, hills (FAIL 2026-10-29).** Zeroing the height channel costs predators 4.8% and -5.4% of
  catches and prey 2% of lifetime (bar 10%). But trained predators climb (78% of time on high ground
  against 64% at random) in plain hills and keep low (56%) in flank_hills, so terrain shapes behaviour
  even though the slope channel adds little to catching. Hills cut flanking's edge over a chaser from 3-3.8x to 2.3x.
  Built: Each cell has a fixed height in [0, 1]:
  climbing costs 4 energy per unit gained (about 0.3 per uphill step), high ground adds up to 2 to
  sight by day, and hills hide what is behind them (19-41% of in-range cells, depending on the
  radius). A new channel gives visible cells' heights relative to the animal's own. Worlds `hills`
  and `flank_hills`. Test (`scripts.hills`): with the height channel zeroed, predators catch at least
  10% less or prey live at least 10% less long, in both runs. The 3-D viewer draws the real map.
  Spec: `docs/specs/2026-10-29-hills-design.md`.
- **Stage 8b, ambush (PASS on two seeds, 2026-10-30).** Hunters spend 64-65% of their time in thickets (map 25%) and stand still 90-93% of the time (random 20%),
  and catch 68-72% less when thickets stop hiding them. A one-line chaser still catches more, but burns two thirds more energy per step, so per meal they are about even. From a viewer's comment on the first Short: predators hide to spend less energy
  chasing. The world could not reward that, since moving cost the same as standing still. Now moving costs 0.5 energy per cell, and
  a quarter of the map is thickets that hide an animal except from the next cell. Test (`scripts.ambush`, both seeds): at least 1.3x the
  map's share of time in thickets, at least 1.5x random's standing still, and at least 10% fewer catches when thickets stop hiding
  predators. Spec: `docs/specs/2026-10-30-ambush-design.md`.
- **Two-level control.** The grid-trained strategy layer sets goals, and a learned locomotion layer
  moves the body. That is how earlier stages carry over into physics.
- **Test:** the strategy layer, transferred from the grid, beats one trained from scratch inside
  the physics world.

### Stage 9: open-ended worlds and scale

- **Worlds that adapt to the agents.** Environment generators co-evolve with the agents (POET,
  PAIRED) and keep producing problems that are just beyond reach. That is the stalemate fix
  automated.
- **Scale out.**
  - Split acting from learning (IMPALA or SEED RL, with V-trace).
  - Use more Sparks, if more become available.
  - Track a skills suite, like Crafter's achievements.
- **Test:** the measured skill count keeps rising without anyone adding new worlds by hand.

## Order and why

Portable brains come first, because every later stage reuses old brains and that is impossible
until inputs are shared. The league and terrain come next, because lesson 1 says pressure is what
grows intelligence. Memory and ecology follow, then communication, then evolving the learners
themselves. Physics and open-endedness come last, once the cheap grid has shown which ideas work.
