# Stage 4: memory

Date: 2026-10-14. Builds on the evo and terrain worlds and the portable 10-channel brains.
Written as the record for the build loop the user asked for, not as a gate.

## Why

Every brain so far is feed-forward: it acts on the 9 × 9 patch in front of it and forgets
everything after the step. That is enough on a world where everything that matters is in view.
This stage makes worlds where it is not, and gives brains a memory, so we can watch for behaviour
that depends on the past: heading back to where food was, staying hidden after a predator drops out
of sight, moving with the seasons.

## World: night and seasons

Two new evo settings, both off by default so earlier worlds are unchanged. A registered world,
`seasons`, turns both on.

- **Night.** `day_len` steps per day (0 = always day), of which the last `night_frac` are night.
  At night every animal's sight radius is 1, whatever its gene, so the patch beyond its
  neighbours reads zero. Sight still costs energy at night.
- **Seasons.** `season_len` steps per season (0 = none). The map's left and right halves take turns
  being in season. Plants out of season cannot seed or germinate, and they die at
  `offseason_death` times the normal rate. Food therefore drifts from one half to the other, and
  an animal can only know which half is due by remembering.
- A light channel tells the brain whether it is night: own-state channels stay as they are, and
  night shows up as the dark patch itself. No clock channel is given. Knowing the time is part of
  what memory has to do.

## Brains: a recurrent core

`ActorCritic(memory=H)` adds a GRU cell of size H after the dense layer (H = 0 keeps the current
network exactly). Each animal slot carries its own hidden state.

- The state is zeroed whenever a slot holds no living animal at the start of a step, so a newborn
  in a reused slot starts blank.
- `--init-from` still works when loading a feed-forward checkpoint into a memory brain. The GRU's
  parameters start fresh; it is a new layer.

**Recurrent PPO.** The rollout stores each slot's hidden state at the start of the rollout. When
learning, minibatches are made of whole slot sequences (all T steps of a slot), not single steps,
and the GRU is re-run over each sequence from its stored start state with resets where the slot was
empty. Living-first sampling picks slots by their number of living steps.

## Success test (can fail)

`pp-mem-001` (GRU, H = 64) and `pp-mem-ff` (feed-forward) on the `seasons` world. Everything else is
equal: same seed and same bodies (evolution off: `gene_sigma = 0`, founder genes fixed), each
trained for the same number of world-steps.

- **Pass:** at matched world-steps over the last third of training, the memory brain's prey live
  at least 1.2× as long as the feed-forward brain's prey, or its predators catch at least 1.2× as
  often. Both are measured by `scripts/compare.py` with trained opponents of the same kind: every
  checkpoint either run saved in the last third of the shorter run's world-steps plays its own
  prey against its own predators (the `scripts/eval.py` trained-vs-trained match), averaged per run.
  The memory brain runs slower, so "matched world-steps" is what keeps the comparison fair.
- **Behaviour to look for in clips:** prey moving towards the half that is coming into season
  before the plants there have grown, and prey not walking back into a predator that slipped out
  of sight at nightfall.

## Out of scope

Transformers and state-space models (S5); memory across lifetimes (Stage 7).
