# How it works

No background needed. The [specs](specs/README.md) have the exact rules.

## The world

A square grid, 48 cells on a side, that wraps at the edges: walk off the right and you come back on the left. Plants
grow on it. Prey (grazers) eat plants, and predators (hunters) eat prey. Each step, every animal looks at the 9×9
patch around itself, picks a move, and pays for being alive in energy. If its energy runs out it dies, and with enough
energy it can breed.

Each stage adds to this world: walls, burrows, seasons, hills, thickets, food that turns to poison.

## Brains: learning by trial and error

Each species starts with one blank neural network, shared by all its animals. Nobody writes hunting or fleeing into it.
The only feedback it gets is a small reward for each step alive, a bigger reward for having offspring, and a penalty
for dying. Hunting and fleeing are never rewarded directly.

Training uses PPO (Proximal Policy Optimization), a standard reinforcement-learning method. Thousands of animals play
out their lives, and the network shifts toward the choices that led to more reward than it expected. So a "species
brain" gets better across all the lives lived, and an animal doesn't learn anything of its own during its life,
unless a stage gives it memory.

## Bodies: evolution

From Stage 2, each animal has genes for speed, sight and breeding threshold, and later for size and brain size. A child
inherits them with small random changes, and each gene costs energy. Nobody picks winners: animals that survive and
breed pass on their genes. Plants have genes too, for how fast they spread and how tough they are.

So two things are changing at once, on two timescales: the species brain, by learning, and the bodies, by evolution.
Each one changes what the other is selected for. In Stage 2, a costly gene like sight only survived when the brain
could use it.

## Memory

Most brains see only the present moment. A memory brain (a GRU, a kind of recurrent network) also carries a small state
from step to step, which starts blank at birth. When Stage 7c says an animal "learned to taste", it means this state
changed after a bad bite, and the animal's choices changed with it. The network's weights stayed fixed. The ability to
do that was learned by the species over training.

## Tests that can fail

This is the part that matters most.

1. **The bar goes in first.** Each stage's [spec](specs/README.md) sets out the test and the pass mark before any run.
   A near miss stays a miss: Stage 5a failed at D 1.95 against a bar of 2.0.
2. **Controls.** Stage 2's first test passed with brains that had never trained, which showed the test was measuring the
   wrong thing. So results are now compared with what a control does: random brains, a one-line hand-written chaser,
   founder bodies, the same brain with one sense or its memory blanked.
3. **Two seeds.** One lucky run isn't a result. Since Stage 6c, a stage only closes as a pass after it holds on a second seed.
4. **Reviews.** Fresh code reviews have voided or weakened results. One found newborns inheriting a dead animal's memory,
   and another found signals that could only reach animals already in sight.

The comparison that has mattered most is the **one-line chaser**: a hand-written rule that steps toward the nearest prey.
If trained hunters can't beat it, they haven't learned to do anything clever, whatever the curves say.

## Why it runs fast

The world, the networks and the learning are all written in [JAX](https://github.com/jax-ml/jax) and compiled together
for the GPU, so hundreds of worlds run in parallel at thousands of steps per second on one NVIDIA DGX Spark (GB10).
Stage runs usually take about 30 minutes. See the [design spec](specs/2026-10-14-predprey-crucible-design.md).

## The code

| Path | What it is |
|---|---|
| `simulations/predprey.py` | The first world: reset, step, observe |
| `simulations/evo.py` | The world with genes, and every rule added since |
| `simulations/worlds.py` | The named worlds (`evo`, `flank`, `ambush`, `fickle`, ...) |
| `agents/` | The network, PPO, policies, checkpoints, evaluation |
| `scripts/train.py`, `scripts/eval.py` | Train a run, and run its base tests |
| `scripts/chase.py`, `signals.py`, `species.py`, `hills.py`, `ambush.py`, `lifetime.py` | Each stage's test |
| `scripts/episode3d.py` | Record one episode for the 3-D viewer |
| `viz/dashboard/` | The viewer and the run dashboard |
| `deploy/` | Docker image and scripts for a GPU machine |
| `site/` | The public web page and its replays |
| `tools/` | Publishing tools for this repo |
