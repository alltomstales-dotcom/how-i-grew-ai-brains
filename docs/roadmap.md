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

**Stage 9b, how bad must a gut be? (built 2026-10-31, training).** A rough sum says hunting and plants pay the same at a gut of about 0.07. Test (`scripts.omnivore --ladder`, both seeds): diet at least 0.50 at a gut of 0.03, and at most 0.25 at 0.20, with 0.08 reported. Spec: `docs/specs/2026-10-31-gut-ladder-design.md`.

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
