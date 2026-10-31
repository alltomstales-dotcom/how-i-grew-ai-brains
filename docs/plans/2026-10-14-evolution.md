# Evolution (sub-project 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add heritable animal bodies (speed, sight, breeding) and evolving plants (spread, toughness) to the predator–prey Crucible, train it on a DGX Spark, and evaluate whether the evolved bodies beat the founder bodies.

**Architecture:**
- A new world module, `simulations/evo.py`, sits beside the untouched `predprey.py`.
- Trainer, evaluation and rendering take a world module (`simulations/worlds.py` registry) instead of importing predprey directly.
- Gene pools pass between worlds at reset time inside `batch_step`.

**Tech Stack:** JAX, Flax linen, Optax, imageio, Chart.js (as sub-project 1).

**Spec:** `docs/specs/2026-10-14-evolution-design.md`

**Code in this plan:** the code is written straight into the source files rather than copied into the plan first. The plan fixes the interfaces, the tests (names and assertions) and the order. The user chose a native build ("Build it…"); the spec holds every number.

## Global Constraints

- `simulations/predprey.py` behaviour is unchanged; all 46 existing tests must still pass.
- Gene values stay in [0, 1]. Default `gene_sigma` is 0.05. Founder genes are animals (0.2, 0.5, 0.5) and plants (0.5, 0.2).
- Observations are 9×9×8, bf16. Sight radius is `1 + floor(3g + 0.5)`.
- The memory budget is a 12 GB cap, `autolearn-*` containers only, and the dashboard on Tailscale only.
- No `jax.random.categorical` over pools. Use `jax.random.choice(..., p=w)`, because categorical's Gumbel noise is pool-size × samples.

## Review Focus

1. **All-extinct pool:** every world's prey extinct at reset. Founders must fall back to the founder distribution, not NaN.
2. **Seeding conflicts:** a bare cell seeded from 2+ neighbours in one step must get exactly one plant, with one parent's genes.
3. **Sight masking:** an animal outside the radius must not be visible in any world channel, and own-state channels must not be masked.
4. **Double move:** a "stay" action with speed 1 must not move, and wrapping must hold for 2-cell moves.
5. **Clips after update 0:** they must start from the evolved pool, not the founders.

---

### Task 1: The evo world: genes, reset, observe

**Files:** create `simulations/evo.py` and `tests/test_evo.py`.

**Produces:**
- Types and constants: `EvoConfig(WorldConfig)`, `Species(pos, energy, alive, genes, gen, tag)`, `State(prey, pred, plant, plant_genes, t)`, `GenePool(prey_genes, prey_gen, prey_w, pred_genes, pred_gen, pred_w, plant_genes, plant_w)`, `OBS_CHANNELS = 8`, `NUM_ACTIONS`.
- Functions:
  - `reset(cfg, key, pool=None)`
  - `make_state(cfg, prey=(), pred=(), plants=(), t=0)`, where animals are `(x, y, energy, speed, sight, breed)` and plants are `(x, y, spread, tough)`
  - `sight_radius(g)`, `observe(cfg, state)`, `gene_pool(states)`

**Tests:**
- `test_reset_founders_in_range`
- `test_reset_draws_founders_from_pool`
- `test_empty_pool_falls_back_to_founders`
- `test_sight_radius_mapping`
- `test_observe_masks_outside_sight`
- `test_observe_self_channels`

### Task 2: The evo world: step

**Files:** modify `simulations/evo.py` and `tests/test_evo.py`.

**Produces:** `advance(cfg, key, state, prey_act, pred_act) -> StepResult`, `batch_step(cfg, key, states, prey_act, pred_act)`, `extra_metrics(info)`, and `StepInfo`, which is the predprey fields plus gene sums.

**Tests:**
- `test_speed_one_moves_two_cells`
- `test_stay_never_moves`
- `test_metabolism_includes_trait_costs`
- `test_breed_threshold_per_animal`
- `test_child_inherits_genes_gen_and_tag`
- `test_mutation_stays_in_range`
- `test_tough_plant_survives_grazing`
- `test_plant_seeds_one_neighbour`
- `test_seed_conflict_one_parent`
- `test_plant_death`
- `test_germination_on_empty_world`
- `test_batch_step_reseeds_done_world_from_pool`
- `test_extra_metrics_finite`

### Task 3: World-generic trainer, policies and networks

**Files:**
- Create `simulations/worlds.py`.
- Modify `simulations/predprey.py` (add `advance`, `batch_step` and `extra_metrics` only), `agents/networks.py`, `agents/policies.py` and `agents/ppo.py`.
- Test file: `tests/test_evo.py`.

**Produces:** `get_world(name) -> (module, ConfigClass)`, `tiny_configs(world_name="predprey")`, `make_trainer(wcfg, tcfg, world=None)` and `init_params(net, key, view=9, channels=4)`.

**Tests:**
- `test_evo_trainer_smoke_tiny`, which checks that the gene metrics are present and finite.
- `test_predprey_batch_step_matches_autoreset`.
- The existing `tests/test_ppo.py` still passes.

### Task 4: Rendering, evaluation and lineage share

**Files:** modify `viz/render.py` and `agents/evaluate.py`; add `reset_mixed(cfg, key, pool)` to `simulations/evo.py`.

**Produces:**
- `render_frame` handles both worlds.
- `rollout_states` and `write_clip` gain `world` and `init_state`.
- `evaluate(..., world=None)`.
- `lineage_share(cfg, prey_pi, pred_pi, key, num_episodes, pool) -> {"prey": float, "pred": float}`.

**Tests:**
- `test_render_evo_frame_shape`
- `test_reset_mixed_tags_half`
- `test_lineage_share_bounds`
- The existing render and eval tests still pass.

### Task 5: Scripts and dashboard

**Files:** modify `scripts/train.py`, `scripts/eval.py`, `viz/dashboard/index.html` and `tests/test_scripts.py`.

**Changes:**
- `train.py`: `--world {predprey,evo}`; `world_name` in `config.json`; gene snapshots at each checkpoint; clips start from the gene pool.
- `eval.py`: dispatches on `world_name`.
- Dashboard: evolution charts with std bands, shown only when their keys exist.

**Tests:**
- `test_tiny_evo_run_writes_gene_snapshots`
- `test_eval_script_on_tiny_evo_run`
- The existing script tests still pass.

### Task 6: GPU memory check, deploy, run and report

**Steps:**
1. Sync to the Spark and rebuild the image (only if the dependencies changed; they don't).
2. After pp-001's eval has finished, run a 20-update evo smoke and read its peak memory. Pick 512 or 256 worlds.
3. Start `pp-evo-001` with 4 hours.
4. Record the results in the spec and the roadmap.
5. Run the eval when the run finishes, and report to the user.
