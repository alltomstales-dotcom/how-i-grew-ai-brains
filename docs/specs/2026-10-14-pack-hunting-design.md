# Stage 6a: hunting together

Date: 2026-10-14. The first slice of Stage 6 (social intelligence). Written as the record for the
build loop the user asked for, not as a gate.

## Why

Stage 3 found that predator brains stop improving. The tests on 2026-10-14 showed this is a
ceiling, not a learning failure:

- Predators plateau even against prey that never learn.
- In the evo world they catch as well as a hand-written "step toward the nearest prey" rule.
- On an equal-speed map that wraps around, a trained prey escapes any single chaser.

The only ways past that are terrain, a speed edge, or more than one hunter. This slice takes the
last: two predators can corner what one cannot catch.

## World

`pack_size` (0 = off, as before). With `pack_size = k`:

- At the end of movement, each predator joins the hunt on the lowest-index prey within one cell
  of it (wrapped Chebyshev distance, its own cell included).
- A prey is eaten when at least k predators joined its hunt.
- Every hunter of an eaten prey shares a meal of `k * catch_energy`, so each gets `catch_energy`
  when exactly k hunt.
- A lone predator on a prey's cell catches nothing.

The `catches` metric counts prey eaten rather than predators that ate. With `pack_size` off the
two are equal, so earlier runs are unchanged.

A registered world, `pack`, is the evo world with `pack_size = 2`.

Nothing else changes: same brains, same observations (each predator already sees a count of the
predators near it), no signals yet. Signals are Stage 6b. Their test will be whether they help
packs form beyond what sight alone gives.

## Success test (can fail)

`pp-pack-001` on `pack`, 30 minutes, extended if catches are still rising.
`scripts/chase.py` compares the learned predators with hand-written chasers, each chasing alone
toward the nearest prey, against the same prey.

- **Pass:** against the run's own trained prey, learned predators catch at least 1.5× as often
  as the chasers. Chasers that ignore each other only reach a two-hunter catch by luck. Beating
  them by half again means the predators converge on the same prey: a hunt together.
- **Behaviour to look for in clips:** two predators closing on one prey from opposite sides.

## Result (2026-10-14): FAIL

`pp-pack-001` and `pp-pack-002`, 30 minutes each (extended from 15 because catches were still
rising), 768 worlds. Catches per step rose from about 0.1 to 0.70 and 0.73 by update 159, and
predators kept close together (nearest other predator 1.3–1.5 cells). But `chase.py` gives:

| run | vs random prey: chaser / learned | vs trained prey: chaser / learned |
|---|---|---|
| pp-pack-001 | 0.0106 / 0.0133 (0.80×) | 0.0143 / 0.0138 (1.04×) |
| pp-pack-002 | 0.0106 / 0.0130 (0.82×) | 0.0146 / 0.0144 (1.02×) |

Against trained prey, learned predators catch no more than lone chasers, far from the 1.5× bar.
The rise during training came from bodies: predators evolved speed 0.66 against 0.2 for founders,
and prey slowed to 0.14. A chaser that runs to the nearest prey also tends to arrive in pairs,
because nearby chasers pick the same nearest prey. So in this world, pairs form without any
coordination, and the brains learned nothing beyond that.

In the runs' evolved bodies (`chase.py --evolved`), both kinds of predator catch about 2.3 times
as often (0.033 against 0.014 per predator-step against trained prey), but the chasers still
match the learned brains: 0.97× and 0.98× (and the same for the Stage 6b signal runs). Bodies
made the difference, not brains.

## Out of scope

Signals and kin recognition (Stage 6b); prey herding.
