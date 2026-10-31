# Stage 3a: terrain and portable brains

Date: 2026-10-14. Builds on the evolution world (`simulations/evo.py`). The user asked for the
roadmap to be built in a loop, with the plan reviewed after each phase, so this spec is the record
rather than a gate.

## Why

pp-001 stalled because prey and predators move alike on an open torus: prey dodge every attack and
predators starve. pp-evo-001 then showed that bodies evolve, but predators still could not live off
trained prey. Terrain changes the geometry: walls make corners and corridors where predators can
trap prey, and burrows give prey a refuge that predators cannot enter but that holds no food. A prey
that hides cannot eat, so hiding has to be timed. That is a behaviour worth watching for, and a
test that can fail.

Brains also need to carry forward from world to world (the user asked whether learning persists),
so the observation layout becomes fixed and a saved brain can be loaded into a world with more
input channels.

## What changes

**Terrain.** Each world gets a static map, `State.terrain` (size × size, int8): 0 open, 1 wall,
2 burrow. It is drawn at every reset from `EvoConfig.walls` straight segments of `wall_len` cells
(horizontal or vertical at random, wrapping like the world) and `burrows` single cells laid on top.
Defaults are 0 and 0, so the plain evo world is unchanged. A new registered world, `terrain`
(`TerrainConfig`), sets `walls=24`, `wall_len=6`, `burrows=16` on the 48 × 48 map.

- **Movement.** A step into a wall leaves the animal where it is. Predators also treat burrows as
  walls. A fast animal's second step is checked the same way.
- **Plants** never grow, seed or germinate on walls or burrows.
- **Spawning.** Animals start on open cells only.
- **Predation** is unchanged: a predator catches prey on its own cell, so a prey in a burrow is safe.

**Observations.** The evo observation gains two channels at the end, wall and burrow, inside the
animal's sight like the other map channels: `[plant, toughness, prey, predators, energy, speed,
sight, breed, wall, burrow]`, `OBS_CHANNELS = 10`. They are all zero in the plain evo world.

**Portable brains.** `agents/checkpoint.load_brains(path, template)` loads a checkpoint into a
network with the same or more input channels. The first convolution's kernel is padded with zero
weights for the new channels, so the loaded brain behaves exactly as before until it sees terrain.
`scripts/train.py --init-from PATH` starts both species from that checkpoint (optimiser state
fresh); `config.json` records it.

**Metrics.** Each step records, for living prey, whether a predator is within 3 cells (Chebyshev,
wrapped) and whether the prey sits in a burrow. Training metrics add `burrow_use_near` (share of
threatened prey in burrows), `burrow_use_far` (share of unthreatened prey in burrows) and
`burrow_cover` (share of cells that are burrows, a constant for reference).

**Rendering.** Walls are drawn grey and burrows dark brown.

## Success test (can fail)

`scripts/eval.py` on a terrain run adds a burrow test. With trained predators, run trained prey and
then random prey for the same episodes and measure `near / far` burrow use for each.

- **Pass:** trained prey's ratio is at least 1.5 × random prey's ratio, and trained prey use
  burrows when threatened at least 1.5 × as often as random prey do.
- Random prey are the control: their burrow use does not depend on threat, so their ratio sits
  near 1.

The existing tests still run: brain ratios against random brains, evolved-versus-founder shares and
plant shift.

## Runs

- `pp-terrain-001`: terrain world, 1024 worlds, 15 minutes, fresh brains.
- `pp-terrain-002`: same, brains warm-started from the pp-evo-002 checkpoint
  (`--init-from`). Comparing the two curves shows whether carried-over learning helps.
- Run lengths go up only if the curves are still rising at 15 minutes.

## Out of scope (Stage 3b)

A league of frozen past brains with Elo ratings, and distillation across changes of network shape.
