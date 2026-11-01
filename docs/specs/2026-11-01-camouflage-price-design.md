# Stage 13b: a valley, and what camouflage can cost

Date: 2026-11-01. Written before any of these runs.

## Why

Stage 13a: at a price of 0.25, one seed never evolved camouflage (0.079) and the other did (0.939), and even that one lasted in only 0.172 of worlds. Free camouflage evolved on both and lasted in 0.746 and 0.973. Two separate questions follow. Was seed 0 stuck in a small fitness valley, since the first benefit comes at a gene of about 0.17 and founders sit at 0.1? And what price can a world live with?

## World

As stage 13a, 48 x 48 on `lasting`, with no crossing:

- `camo_start`: price 0.25, with founders at 0.2, just past the first step. Tests the valley.
- `camo_10`: price 0.10, founders at 0.1.
- `camo_05`: price 0.05, founders at 0.1.

Seeds 0 and 1, 150 updates, with one seed on each machine.

## Success test (can fail)

Over the last 15 updates, on both seeds:

**The valley:** `camo_start` hunters' gene at least 0.5 on both seeds. If so, founders past the first step all get there, and stage 13a's seed 0 was stuck for that reason. If not, something else stopped it.

**The price:** a cheaper arm passes if, on both seeds, its hunters' gene is at least 0.5, both species are alive at step 1,999 in at least 0.5 of worlds, and a fixed chaser catches at most 1.0x what learned hunters do.

The stage passes if the valley bar and at least one price arm pass.

Prediction: the valley bar passes, with `camo_start` still lasting poorly, since the price is unchanged. Both cheaper arms evolve the gene on seeds where they get past the valley, and that's the risk: the valley is there at any price above zero, though a cheaper price makes the slope gentler. So I expect `camo_05` to pass and I'm unsure about `camo_10`.

Reported, not bars: gene curves, hunter numbers, and grazer sight.

## Result (2026-11-01): FAIL on the valley bar, and both cheaper prices pass

| | hunters' camo gene, seeds 0, 1 | gene over time, seed that stuck | both alive at step 1,999 | hunters | chaser against learned hunters |
|---|---|---|---|---|---|
| `camo_start`, 0.25, founders 0.2 | 0.937, **0.084** | 0.18, 0.11, 0.08, 0.08, 0.08 | 0.031, 0.000 | 9.2, 7.7 | 0.69x, 0.64x |
| `camo_10`, 0.10 | **0.942, 0.942** | | **0.594, 0.891** | 17.3, 19.5 | 0.67x, 0.69x |
| `camo_05`, 0.05 | **0.942, 0.942** | | **0.711, 0.781** | 21.2, 22.3 | 0.69x, 0.66x |

**The valley bar fails,** so the stage fails by its rule, and my prediction there was wrong. Founders past the first step didn't all get there: seed 1 started at 0.18 and fell back to 0.08. So a one-step valley at the first benefit isn't enough to explain stage 13a's stuck seed. Perhaps the first step of hiding buys too little to pay for itself at this price, so the slope runs downhill for longer than one step, which I haven't tested.

**At 0.25 it looks like a coin flip.** Across stages 13a and 13b, two of four runs got there and two didn't, and the seed that gets there changed with the founders. Those that do reach it last poorly (0.172, 0.031).

**Both cheaper prices pass.** Every run evolves the gene by about update 60 and worlds last, with hunters ahead of the chaser. This is the first time hiding that evolved, at a price, has carried a trained world. Worlds at 0.10 and 0.05 aren't clearly ordered. Seed 1 lasts better at the higher price, so don't read the difference between them as a trend.

Grazers' sight stays near 0.5 in every arm that got camouflage, so they don't appear to give up looking, within these 150 updates.

