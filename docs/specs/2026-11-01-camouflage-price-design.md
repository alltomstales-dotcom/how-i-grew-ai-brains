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
