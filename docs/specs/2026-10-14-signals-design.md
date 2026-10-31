# Stage 6b: a signal neighbours can see

Date: 2026-10-14. The second slice of Stage 6 (social intelligence), built on the pack world from
Stage 6a. Written as the record for the build loop the user asked for, not as a gate.

## Why

In the pack world a prey is eaten only when two predators hunt it together. Predators can already
see each other, so they can form packs by sight alone. A signal lets an animal say something that
sight cannot show, such as "prey here" or "predator here", and lets others act on it. The question
is whether the animals invent a use for a channel that means nothing at the start.

## World

`signals` (bool, off by default so every earlier world is unchanged). With it on:

- **Actions.** Each brain has 10 actions instead of 5: the same five moves, each with the signal
  off or on (`move = a % 5`, `signal = a >= 5`). A dead slot never signals.
- **Seeing signals.** One more observation channel, last, counts the animals of the viewer's own
  species that signalled on the previous step in each cell of its view. It respects sight and
  night like the other count channels. Prey and predators do not hear each other.
- **Cost.** A signal costs `signal_cost` (0.05) energy, about a tenth of a predator's base upkeep.
  Every trait needs a real cost; this one keeps the channel from filling with noise for free.
- **New-born animals** start silent. A slot that died and was reused does not carry the dead
  animal's signal.

A registered world, `signals`, is the pack world with signals on.

Brain shapes now depend on the world: `worlds.obs_channels(world, cfg)` and
`worlds.num_actions(world, cfg)` replace the module constants wherever a brain is built. Brains
from 5-action worlds do not load into 10-action brains; signal runs start from scratch.

## Success test (can fail)

`pp-sig-001` and `pp-sig-002` on `signals`, 30 minutes each, seeds 0 and 1.

- **Signals are used:** the signal rate of the trained predators is neither near 0 nor near 1
  (between 0.02 and 0.98), so it can carry information.
- **Signals carry meaning:** for the trained predators, the signal depends on whether a prey is
  within 2 cells of the signaller. Mutual information between the two is at least 0.05 bits.
- **Listeners use them** (the decisive part): `scripts/signals.py` evaluates each trained run twice
  on the same seeds, once as is and once with the signal channel zeroed in every observation.
  **Pass:** predators catch at least 10% less without the channel, on both seeds.
- **Control:** the pack runs from Stage 6a, compared at matched world steps with `compare.py`.
  This is reported, not gated: signals may help listeners without making the brains better
  overall in 30 minutes.

Prey get the same channel among themselves. Their use of it is reported in the same way, not gated.

## Result (2026-10-14): FAIL

`pp-sig-001` and `pp-sig-002`, 30 minutes each, 768 worlds.

| run | predators signal | information, bits | catches as trained / signal channel blanked | drop | prey signal |
|---|---|---|---|---|---|
| pp-sig-001 | 37% | 0.027 | 0.0139 / 0.0139 | −0.1% | 35%, 0.000 bits |
| pp-sig-002 | 38% | 0.012 | 0.0137 / 0.0138 | −0.7% | 27%, 0.000 bits |

Animals signal often but say almost nothing, and listeners ignore it. That fits Stage 6a: in the
pack world a lone chase already does as well as anything (learned predators against chasers:
1.03× and 1.15× here), so a signal has no problem to solve. The cost of 0.05 energy is too small to
silence the noise. Signals get a second chance in `flank_signals` (Stage 6c), where coordination
is needed.

*Review finding (2026-10-14, after this result).* An independent code review found that the
signal channel was masked by the listener's sight, so a predator only heard signallers it could
already see (2–3 cells for founders, out of a 9-cell view). A signal could therefore say little
that sight did not. This weakens the result: the test above measured a signal that could hardly
carry news. The fix (commit after a99d0c9) makes signals calls, heard across the whole view by
day or night. The pack-world runs were not repeated, since nothing there needed coordinating.
The flank-with-signals run is the fair test.

## Out of scope

Signals heard across species (eavesdropping), more than one signal token, kin recognition.
