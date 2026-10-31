# Stage 6c: closing in from both sides

Date: 2026-10-14. The third slice of Stage 6 (social intelligence). Written as the record for the
build loop the user asked for, not as a gate.

## Why

Stage 6a failed in an instructive way. "Two predators must hunt together" did not need teamwork:
chasers that each run to the nearest prey arrive in pairs anyway, and learned predators did no
better than they did. Coordination only pays when the simple rule fails. Real pack hunters do
something a lone chaser cannot: they cut off the escape by coming from the other side.

## World

`flank` (bool, used with `pack_size`). A prey is eaten only when two of its hunters stand exactly
opposite each other through it: west and east, north and south, or two diagonal corners (wrapped
offsets within one cell). A hunter on the prey's own cell counts for no side. Two chasers coming up
behind the same prey are on the same side, so they catch nothing.

*Review finding (2026-10-14).* The first version checked for "any hunter east and any west, or any
north and any south". That let two hunters on the same side count: north-west plus south-west
passed as north plus south. 16 of the 28 pairs of neighbouring cells counted as a flank instead of
the 4 truly opposite pairs. The first flank pair was stopped at update 68 and set aside, and the
rule now needs exact opposites.

Registered worlds:

- `flank`: the pack world with flanking.
- `flank_signals`: the same with the Stage 6b signal, heard across the whole view.

## Success test (can fail)

`pp-flank-001` on `flank` and `pp-flank-sig-001` on `flank_signals`, run as a pair, 30 minutes
each.

- **Pass:** against the run's own trained prey, learned predators catch at least 1.5× as often as
  the hand-written chasers (`chase.py`), with founder bodies and again with the run's evolved
  bodies (`chase.py --evolved`).
- **Signals (reported):** `signals.py` on `pp-flank-sig-001`. If blanking the signal channel costs
  at least 10% of catches, signals help the flank.
- If either run passes, repeat it on a second seed before calling it a result.

## Result, seed 0 (2026-10-14): PASS

`pp-flank-001` and `pp-flank-sig-001` (corrected rule), 30 minutes each, 768 worlds. Catches per
step rose from 0.009 to about 0.03, and predators drew together (nearest other predator 4.0 → 2.4
cells). Unlike the pack world, both species evolved *slower* bodies (predator speed 0.06, prey
0.10), so the gain did not come from speed.

Catches per predator-step against trained prey:

| run | bodies | lone chaser | learned | learned ÷ chaser |
|---|---|---|---|---|
| pp-flank-001 | founder | 0.0009 | 0.0036 | 3.8× |
| pp-flank-001 | evolved | 0.0012 | 0.0037 | 3.2× |
| pp-flank-sig-001 | founder | 0.0009 | 0.0033 | 3.7× |
| pp-flank-sig-001 | evolved | 0.0010 | 0.0039 | 3.7× |

Each figure covers about 1.3 million predator-steps, so the learned rate stands on several
thousand catches. Against random prey the edge is smaller (1.1–2.3×): random prey wander into
flanks by chance, while trained prey evade chasers that come from one side.

Signals in `flank_signals` (seed 0): blanking the channel cost 33% of catches with founder bodies
and 17% with evolved ones (bar 10%), so listeners use them. But a signal says almost nothing about
whether prey is within 2 cells (0.003 bits). It seems to work as a beacon ("a packmate is here"),
heard beyond sight. Silence is a state the brain never met in training, so part of the drop may be
surprise rather than lost information. A shuffled control (signals heard from another world) and
the second seed decide it.

**Shuffled control (seed 0, 22:12).** Predators heard the signals of another world, which keeps
the signal statistics and removes the link to their own packmates:

| bodies | as trained | channel blanked | another world's signals |
|---|---|---|---|
| founder | 0.0032 | 0.0026 (−18.3%) | 0.0024 (−24.5%) |
| evolved | 0.0041 | 0.0031 (−24.5%) | 0.0030 (−26.8%) |

Wrong signals hurt as much as silence, or more, so the drop is not mere surprise: listeners use
what their own packmates signal. The blanked-channel drops differ from the first evaluation (33%
and 17%) because the five-gene code draws different random worlds for the same seed. Single
estimates swing by about ±10 points, but every one is above the 10% bar.

## Result, seed 1 (2026-10-14): PASS, so 6c is a result

`pp-flank-002` and `pp-flank-sig-002`, the same rules and lengths on seed 1:

| run | bodies | lone chaser | learned | learned ÷ chaser |
|---|---|---|---|---|
| pp-flank-002 | founder | 0.0011 | 0.0036 | 3.3× |
| pp-flank-002 | evolved | 0.0011 | 0.0040 | 3.7× |
| pp-flank-sig-002 | founder | 0.0008 | 0.0025 | 3.0× |
| pp-flank-sig-002 | evolved | 0.0011 | 0.0039 | 3.7× |

All eight readings across both seeds sit between 3.0× and 3.8×, against a bar of 1.5×. In the
flank world, learned predators beat a hand-written chase on both seeds, in founder and evolved
bodies, with and without signals.

Signals on seed 1 are mixed:

| bodies | as trained | channel blanked | another world's signals |
|---|---|---|---|
| founder | 0.0025 | 0.0026 (+4.8%, FAIL) | 0.0024 (−6.2%, FAIL) |
| evolved | 0.0039 | 0.0031 (−21.9%) | 0.0028 (−28.4%) |

In evolved bodies, the bodies the run actually ends with, signals mattered on both seeds (drops of
17–28%, shuffled 27–28%). In founder bodies they mattered on seed 0 and not on seed 1. So the
signal result holds for the run's own animals but is not as firm as the flank result itself. The
flank result does not depend on signals: `pp-flank-002`, with no signal at all, does as well.

## Out of scope

More than two hunters, and roles (driver and blocker) named in the rules.
