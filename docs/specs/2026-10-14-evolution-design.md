# Sub-project 2: evolution of bodies and plants — design

Status: design approved by the user on 2026-10-14 ("Looks great. Build it, run it when planned and
update docs as it goes"). It builds on sub-project 1 (`2026-10-14-predprey-crucible-design.md`).

## Goal

Add Darwinian evolution to the predator–prey Crucible. Two things change:

- **Bodies evolve.** Animals' bodies are inherited from their parent with mutation, and each trait
  costs energy.
- **Plants evolve too.** Grass becomes a population of plants that reproduce by seeding and inherit
  their own traits.

Brains are not inherited. Each species keeps one PPO-trained network, which now also sees the
animal's own genes, so a single brain learns to drive many bodies. Purpose: **watchable
emergence**, meaning a person can see traits shift in the charts and clips.

## What the user decided

- Heritable animal traits: **speed, sight range, breeding threshold** (size was not chosen).
- **Plants evolve too.** This was the user's own addition. Plant traits are **spread** and
  **toughness**.
- Run `pp-evo-001` after pp-001 ends, with the same memory budget (12 GB cap, `autolearn-*`
  containers, port 9001 on Tailscale only).

## The world: `simulations/evo.py`

This is a new module. `simulations/predprey.py` keeps its behaviour, so sub-project 1's tests and
the pp-001 baseline stay valid. The evo module reuses the predprey helpers (`move`, `predation`,
`_first_claim`, `_count_map`, `_windows`, `nearest_dist`). `EvoConfig` subclasses `WorldConfig`.

### Genes

Every gene is a float in [0, 1].

| Who | Gene | Phenotype | Cost |
|---|---|---|---|
| Animal | speed `s` | A moving animal moves twice in its chosen direction with probability `s` | `speed_cost · s²` energy per step (default 1.0) |
| Animal | sight `g` | View radius `r = 1 + floor(3g + 0.5)` ∈ {1..4}; world channels outside Chebyshev radius `r` read 0 | `sight_cost · (r − 1) / 3` energy per step (default 0.5) |
| Animal | breed `b` | Breeds when energy ≥ `repro_energy · (0.5 + b)`; the child gets half the parent's energy | Implicit: many weak children versus few strong ones |
| Plant | spread `p` | Seeds an empty 4-neighbour cell with probability `seed_rate · p · (1 − tough_seed_cost · k)` | Natural death probability `die_rate · p` per step |
| Plant | tough `k` | Survives being grazed with probability `k`; the grazer gains `grass_energy · (1 − 0.5k)` | Seeds less often (see spread) |

Defaults:
- `seed_rate` 0.2, `tough_seed_cost` 0.8, `die_rate` 0.02.
- Background germination: 0.0005 per bare cell per step. The new plant's genes come from a random
  living plant in the same world, or from the founder distribution if the world has none.

Founders:
- Animals start with genes (speed 0.2, sight 0.5, breed 0.5), plants with (spread 0.5,
  tough 0.2). Each gets Gaussian noise of σ 0.05, clipped.
- At sight 0.5 the radius is 3. At breed 0.5 the thresholds equal sub-project 1's (30 and 45).

Mutation:
- A child, animal or plant, gets `clip(parent + N(0, gene_sigma))` with `gene_sigma = 0.05`.
- Animals also carry `gen`, which is the parent's plus 1, and a lineage `tag`, copied from the
  parent and used only by evaluation.

### Step order

1. Move, with speed doubling.
2. Predation (unchanged).
3. Grazing with toughness.
4. Metabolism: `1 + speed cost + sight cost`, then the energy cap.
5. Starvation.
6. Reproduction with per-animal thresholds and gene inheritance.
7. Plants, in this order: natural death, then seeding (a cell seeded from several neighbours takes
   the one with the highest random score), then germination.
8. Done check: `t ≥ max_steps` or either animal species extinct (unchanged).

Rewards are unchanged: alive, birth and death. Fitness is never rewarded directly; selection comes
only from who breeds.

### Observation

There are 9×9×8 channels, in bf16:
- `plant`, `plant toughness`, `prey count`, `predator count`. These are blanked outside the sight
  radius.
- `own energy / cap`, `own speed`, `own sight gene`, `own breed gene`.

### Evolution across resets

`batch_step(cfg, key, states, prey_act, pred_act)` works on the whole batch of worlds:
1. Step every world.
2. If any world is done, build a gene pool from all worlds' living animals and plants.
3. Reset the done worlds, drawing founders from that pool.

Founder genes, and their generation counts, are drawn with `jax.random.choice` weighted by
`alive`. They are copied without mutation. If a pool is empty, the founder distribution is used.
For predprey, `batch_step` is the old per-world autoreset.

## Training

- `make_trainer(wcfg, tcfg, world=predprey)` takes a world module with this interface:
  - `reset`, `observe`, `batch_step`, `advance`;
  - `OBS_CHANNELS`;
  - `extra_metrics(info)`, which predprey returns empty.
- `init_params` takes the channel count.
- Evo metrics, as means over living agents or plants and over the rollout:
  - `prey_speed|sight|breed` (plus `_std`) and the same for `pred_`;
  - `plant_spread|tough` (plus `_std`);
  - `plant_cover`, `prey_gen`, `pred_gen`.
- `scripts/train.py --world evo`:
  - Records `world_name` in `config.json`.
  - At each checkpoint, saves `checkpoints/genes_NNNNNN.npz` and `genes_latest.npz`, holding the
    living animals' and plants' genes and generations.
  - Starts each clip after update 0 from the current gene pool, so clips show evolved bodies.
- Rendering:
  - Animal brightness scales with speed.
  - Plants get darker green as toughness rises.
  - The header shows mean speed, sight and toughness.
- The dashboard adds evolution charts (gene means with ±1 std dashed lines, plant cover and
  generations). A chart appears only when the run logs its keys.

## Success test (`scripts/eval.py` on an evo run)

1. **Evolved bodies win.** In 256 mixed worlds, half of each species' founders take genes from
   `genes_latest.npz` (tag 1) and half from the founder distribution (tag 0). Plants come from the
   evolved pool. Both halves use the same trained brains. Over steps 250–499, the tag-1 share of
   living-agent-steps must be ≥ 0.60 for prey **or** for predators.
2. **Plants evolved.** At least one plant gene's mean in `genes_latest.npz` differs from the founder
   mean by ≥ 0.15.
3. **A person can see it.** The user can name a trait change in the clips or charts.

`eval.json` records `prey_evolved_share`, `pred_evolved_share`, `plant_shift` (per gene),
`evolved_pass` and `plant_pass`. The exit code is 0 only when both pass. Predprey runs are
evaluated exactly as before.

## Budget and operations

- 512 worlds by default. Observations double in size (8 channels), so memory is measured with a
  short GPU run before the long one. If its peak is over 10 GB, the run uses `--num-envs 256`.
- The 4-hour cap is unchanged.

## Out of scope

Evolving brains (lineage networks or neuroevolution), sexual reproduction, a size trait,
speciation metrics, and seeing other animals' genes.

## Changes after the final review and pp-001 (2026-10-14)

pp-001 was stopped at update 625 with the user's agreement. Both of its tests passed (predators
3.36×, prey 1.61×), but trained prey dodged almost every attack, predators starved within about
30 steps, and learning plateaued.

The final review then found that the success test would pass with untrained brains. Food was
plentiful, prey stayed at the 64-slot cap, and the breed gene collapsed to about 0.06, which alone
gave the evolved half a 0.72 share. The changes:

- **`pred_start_energy` 60** in the evo world only. The user chose this.
- **`seed_rate` 0.03 and `birth_cost` 8**, so food becomes the limit. These were chosen from
  random-brain simulations on CPU (64 worlds, 2,000 steps, settings A–G).
  - In setting G, prey settle at about 30 of 64 slots and plant cover is about 0.41.
  - The breed gene levels off near 0.28 instead of collapsing.
- **Random-brain control** from that simulation, at 2,000 steps: prey speed 0.28, sight 0.07,
  breed 0.28; plant spread 0.65, toughness 0.13. Gene trends in the trained run are read against
  this. For example, sight rising above 0.07 means the trained brains make sight worth its cost.
- **Evaluation:**
  - Lineage tags go to random slots.
  - `single_gene_share` reports the share when one gene at a time is evolved.
  - The genes are taken from the snapshot matching the evaluated checkpoint.
  - The brain ratios on an evo run are labelled as computed with founder bodies.
- **`max_prey` 96** in the evo world. In the smoke run, trained grazers filled 64 slots, but with 96 they settle around 66–70, held by food. Peak memory is 6.79 GB at 512 worlds, and throughput is about 2,545 steps/s.
- **Faster resets:** each reset draws from 4,096 living plants (`thin_plants`), not the full
  1.18M-cell pool.
- **Germination:** a germinated plant copies a random living plant and is also mutated.
- **Caveat:** under random brains, plant spread still drifts up by about 0.15 in 2,000 steps, so
  `plant_pass` is weak evidence on its own. Compare against the control above.
