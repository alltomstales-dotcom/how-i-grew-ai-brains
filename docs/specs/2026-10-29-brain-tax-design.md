# Stage 7b: brain size at equal wealth (a brain tax)

Date: 2026-10-29. The follow-up Stage 7a left open. Written before any run, so the bar cannot move.

## Why

Stage 7a asked whether evolution buys more brain where thinking pays. In flank, learned predators
beat a simple chase threefold; in pack they do not. It failed both ways: at a flat cost of
0.5 energy per step every brain hit the floor, and at 0.1 pack predators kept *more* brain (0.50
against 0.19). Pack predators catch about twenty times as often, and a flat cost per step bites a
poor animal far harder than a rich one. So brain size followed food, not the value of thinking.

The fix is to charge the brain as a share of income instead of a flat fee. Then rich and poor
animals pay the same fraction for the same brain.

## World

`brain_tax` (float, 0 by default, which keeps every earlier run unchanged). An animal with brain
gene g keeps `1 - brain_tax * g` of every meal: grass for prey, its share of a catch for predators.
An animal that eats nothing pays nothing.

With a tax and no flat cost, a bigger brain pays when it raises the animal's food intake by more
than the share it gives up. That depends on how much thinking helps, not on how rich the animal is.

Registered worlds: `pack_tax` and `flank_tax`. These are `pack_brains` and `flank_brains` (spread
founders) with `brain_cost = 0` and `brain_tax = 0.3`.

## Runs

Seed 0, 30 minutes each, run two at a time (12 GiB each):

| run | world | tax |
|---|---|---|
| pp-tax-flank-30 | flank_tax | 0.3 |
| pp-tax-pack-30 | pack_tax | 0.3 |
| pp-tax-flank-10 | flank_tax (`--set brain_tax=0.1`) | 0.1 |
| pp-tax-pack-10 | pack_tax (`--set brain_tax=0.1`) | 0.1 |

The cost-free pair from 7a v2 (`pp-brain-flank0`, `pp-brain-pack0`, seed 0) is the drift and
hitchhiking control: its gap (flank minus pack predator brain) was +0.02.

Why these taxes: a flat 0.5 per step was roughly 40% of a pack predator's income, and every brain
hit the floor. A flat 0.1 was roughly 8%, and brains settled mid-range. A 10% and a 30% tax bracket
the range where brains should settle mid-range.

## Success test (can fail)

Measured on the predators' mean brain gene at the end of each run.

- A tax level is **informative** if both its runs end with the predator brain in [0.1, 0.7]: off
  the floor and below the cost-free ceiling.
- **Pass:** at some informative tax, flank minus pack is at least +0.12 (0.1 above the cost-free
  gap), and no informative tax shows a gap of −0.1 or worse. A pass is then repeated on seed 1 at
  that tax.
- **Fail:** every informative tax shows a gap below +0.12.
- **Inconclusive:** no tax is informative. Then the tax is not tuned further.

Reported: brain trends over the run, the prey brains, speed and sight, catches per step, and
`chase.py --evolved` for each run.

## Seed 0 result (2026-10-29): PASS at both taxes

Gene means are the average of the last 20 updates (about 155 updates per run). Learned ÷ chaser is
`chase.py --evolved` against trained prey.

| run | tax | predator brain | prey brain | predator speed / sight | catches per step | learned ÷ chaser |
|---|---|---|---|---|---|---|
| pp-tax-flank-30 | 0.3 | 0.53 | 0.08 | 0.05 / 0.07 | 0.035 | 3.7× |
| pp-tax-pack-30 | 0.3 | 0.18 | 0.09 | 0.70 / 0.79 | 0.63 | 1.04× |
| pp-tax-flank-10 | 0.1 | 0.69 | 0.16 | 0.05 / 0.09 | 0.038 | 3.7× |
| pp-tax-pack-10 | 0.1 | 0.38 | 0.16 | 0.74 / 0.82 | 0.66 | 1.03× |

Both taxes are informative (every predator brain in [0.1, 0.7]; flank at tax 0.1 only just, at 0.69).
Gap (flank minus pack): **+0.35 at tax 0.3** and **+0.31 at tax 0.1**. The bar was +0.12, and the
cost-free gap from 7a v2 was +0.02. **PASS.**

What it shows:

- **The 7a result reverses once the cost is fair.** Under a flat cost, pack predators kept more brain
  (0.50 against 0.19). Under a share of income, flank predators keep about three times as much.
  Where learning beats a simple chase almost fourfold, evolution pays for brain; where it adds
  nothing, brain is cut.
- **Pack brains fell as pack predators got rich.** Pack predator brains held near 0.45-0.6 until
  predators evolved speed and sight (about update 90), then fell to 0.17 (tax 0.3) and 0.38 (tax
  0.1). Speed and sight do the catching there, so brain is the trait to cut. Flank predators never
  evolved speed or sight, and their brains rose steadily.
- **Prey brains fell low in every run** (0.08-0.16). Grazing needs little thought, and prey pay the
  tax on every bite.

## Seed 1 result (2026-10-29): PASS, Stage 7b closed

Tax 0.3, seed 1, 30 minutes each, same measures as above.

| run | tax | predator brain | prey brain | predator speed / sight | catches per step | learned ÷ chaser |
|---|---|---|---|---|---|---|
| pp-tax-flank-30-s1 | 0.3 | 0.47 | 0.08 | 0.06 / 0.07 | 0.033 | 3.7× |
| pp-tax-pack-30-s1 | 0.3 | 0.14 | 0.09 | 0.77 / 0.82 | 0.69 | 1.04× |

Gap (flank minus pack): **+0.33**, bar +0.12. The pass holds on a second seed.
