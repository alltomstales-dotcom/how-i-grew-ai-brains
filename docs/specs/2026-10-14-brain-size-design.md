# Stage 7a: brain size as a gene

Date: 2026-10-14. The first slice of Stage 7 (brains evolve too). Written as the record for the
build loop the user asked for, not as a gate.

## Why

Stage 6 found one world where learning clearly beats a simple rule (flanking: learned predators
catch 3.2–3.8× as often as lone chasers) and one where it does not (pack hunting: about 1×). If
thinking costs energy, evolution should buy more of it only where it pays. That is the question
behind "evolution has to justify intelligence".

## World

`brains` (bool, off by default). With it on:

- **A fifth animal gene, `brain`**, in [0, 1], founders at 0.5. It is inherited and mutates like
  the others, and with mating it crosses over. Every evo world carries the gene (as with size), but
  it does nothing unless `brains` is on.
- **Capacity.** Each species still shares one network. An animal with gene g uses only the first
  `units(g) = 8 + round(g * 120)` of the 128 hidden units in the network's dense layer; the rest
  are zeroed for that animal. The conv layers are shared in full. So a small-brained animal runs
  a narrower slice of the same brain.
- **Cost.** `brain_cost * g` energy per step (0.5 by default, the same as full sight).
- **Observation.** The gene is appended as the last observation channel (a constant plane), which
  the network reads to build the mask. It comes after the signal channel when both are on.

Brains with memory are not supported with `brains` (the trainer refuses the combination).

Registered worlds: `pack_brains` (pack + brains) and `flank_brains` (flank + brains).

## Success test (can fail)

`pp-brain-flank` on `flank_brains` and `pp-brain-pack` on `pack_brains`, run as a pair, 30 minutes
each, seed 0; then a second seed if the first passes.

- **Pass:** at the end, the predators' mean brain gene in `flank_brains` exceeds that in
  `pack_brains` by at least 0.1, on both seeds.
- Reported: the prey brain genes, the trend over the run, and the chase comparison for both runs.

## v2, after a code review (2026-10-14)

An Opus review of the brain code, run while the first pair was training, found no bug but two
design problems that could make the test meaningless:

- **Big brains could not pay.** A hidden unit outside an animal's mask outputs exactly zero, so it
  gets no gradient. Founders all start near 0.5 (about 68 units), so units above about 76 never
  train. A mutant with a bigger brain gets the same policy plus untrained noise and pays more
  energy. The gene could only drift down, and the test would measure how fast it falls, not
  whether thinking pays.
- **The gene can move without being chosen.** Breeding in these worlds is asexual, so a lineage
  that wins for speed or sight carries its brain value along with it. Hunger differs between the
  worlds too: flank predators catch less, so the same brain cost bites harder.

Changes:

- **Founders spread the brain gene** (`brain_spread`, on in both brain worlds). Founders draw it
  uniformly from [0, 1], so every width of the brain is trained from the start. Other genes keep
  their founder values. Animals drawn from the gene pool are unchanged.
- **A cost-free control pair.** The same two worlds with `brain_cost = 0`. Without a cost, the
  gene can only drift or hitchhike, so the gap between the worlds in this pair is what chance and
  linkage alone produce.
- `signals.py` now blanks or shuffles the signal channel by its index, not the last channel, which
  is the brain gene when both are on. No registered world had both, so no result was affected.

Test (replaces the one above): `pp-brain-flank2` and `pp-brain-pack2` (spread founders, with
cost), then `pp-brain-flank0` and `pp-brain-pack0` (spread founders, no cost), 30 minutes each.

- **Pass:** at the end, the predator brain gene in flank exceeds pack by at least 0.1 with cost,
  and that gap is at least 0.1 larger than the gap in the cost-free pair. Then a second seed.
- Reported: the trend of the brain gene and of the other genes, prey brains, and `chase.py
  --evolved` for both costed runs.

The first pair (`pp-brain-flank` and `pp-brain-pack`, narrow founders) ran before the change and
is kept as a reference for what the narrow start does. After 154 updates:

| run | predator brain | prey brain | learned ÷ chaser (trained prey, founder bodies) |
|---|---|---|---|
| pp-brain-flank | 0.07 (from 0.49) | 0.075 | 2.9× |
| pp-brain-pack | 0.13 (from 0.50, low point 0.08 at update 61) | 0.086 | 0.93× |

Brains shrank in both worlds, and further in flank, the opposite of the hypothesis. Flank
predators still beat chasers almost threefold with about 16 hidden units, so flanking may need
little capacity. Flank predators also catch far less (0.004 against 0.017 per predator-step), so
the same cost takes a bigger share of their food. v2 tells whether this survives a fair start.

## v2 result (2026-10-14): FAIL, and the cost decides everything

Spread founders, seed 0, 30 minutes each (about 155 updates). Gene means at the end:

| run | brain cost | predator brain | prey brain | predator speed / sight | catches per step |
|---|---|---|---|---|---|
| pp-brain-flank2 | 0.5 | 0.06 | 0.07 | 0.06 / 0.07 | 0.03 |
| pp-brain-pack2 | 0.5 | 0.12 | 0.08 | 0.58 / 0.63 | 0.45 |
| pp-brain-flank0 | 0 | 0.79 | 0.75 | 0.05 / 0.10 | 0.04 |
| pp-brain-pack0 | 0 | 0.78 | 0.70 | 0.74 / 0.83 | 0.71 |

Gap (flank minus pack) in the predator brain: −0.06 with cost, +0.02 without. The bar was +0.1 and
then 0.1 more than the cost-free gap. **FAIL.**

What it shows:

- **Brains are worth having.** With no cost, every brain grew, from 0.5 to 0.7–0.8, in both
  species and both worlds, steadily over the run. Drift alone would not move four populations the
  same way. So more hidden units do help.
- **At a cost of 0.5, no brain pays for itself.** Every brain fell to the floor within 20 updates.
  There is no middle ground at either cost, so this pair cannot show where thinking pays more.
- **Rich predators keep more of everything.** Pack predators catch about 15 times as often as
  flank predators, and with cost they kept more brain, speed and sight. Flank predators, short of
  food, cut all three. The difference between the worlds tracks wealth, not the value of
  thinking.
- **Flanking needs little brain.** In evolved bodies with brains at 0.06 (about 15 hidden units),
  flank predators still caught 2.4 times as often as chasers; pack predators 1.04 times.

Next (v3): the same pair at `brain_cost = 0.1`, a fifth of the cost, looking for a cost where
brains settle in the middle, so that a difference between the worlds could show.

## v3 result (2026-10-15): FAIL, the opposite way; Stage 7a closed

`brain_cost = 0.1`, spread founders, seed 0, 30 minutes each.

| run | predator brain | prey brain | predator speed / sight | catches per step | learned ÷ chaser (evolved bodies) |
|---|---|---|---|---|---|
| pp-brain-flank-c01 | 0.19 | 0.20 | 0.06 / 0.07 | 0.03 | 2.9× |
| pp-brain-pack-c01 | 0.50 | 0.26 | 0.75 / 0.83 | 0.69 | 1.04× |

At this cost brains settle in the middle, so a difference can show, and it runs the wrong way:
predators keep 0.31 *less* brain where learning beats a simple chase threefold than where it
adds nothing. The cost-free gap was +0.02, so this is not drift.

The pack run shows the order of events. Predator brains sat at 0.35–0.38 until predators evolved
speed and sight (updates 58–96), catches rose tenfold, and only then did brains climb to 0.50.
Brain size followed food.

**Finding.** In these worlds the brain an animal can afford is set by its energy budget, not by
how much thinking pays. Flanking pays off hugely, but flank predators are poor (one twentieth of
the catches), so they cannot buy brain; and the flanking skill fits in about 20–25 hidden units.
The bar is not moved after the fact, and the cost is not tuned further, which would be fishing.
A fair version would need to compare worlds of equal wealth (for instance, brain cost as a share
of what an animal eats), which is left for later.

## Out of scope

Evolving learning settings (rates, entropy), learning within a lifetime, per-animal networks.
