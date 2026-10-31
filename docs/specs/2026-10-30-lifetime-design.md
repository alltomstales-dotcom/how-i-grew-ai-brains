# Stage 7c: learning within one lifetime

Date: 2026-10-30. Written before any run.

## Why

So far a species shares one brain, trained across all its animals, and nothing an animal experiences changes what it
does later in its own life. Stage 4 tested memory and found no gain, and concluded that a fair test needs a world where
only memory pays. This is that world.

## World

`fickle` (EvoConfig plus two plant kinds that persist, as in niches v5):

- **One kind feeds, the other poisons.** A bite of the good kind gives `grass_energy` (5). A bite of the bad kind costs
  `poison` (3). Toughness no longer discounts a meal.
- **Nothing shows which.** The good kind is hidden state in the world (`State.good`), never in any observation. The
  animal can see the two kinds, and can feel its own energy change after a bite.
- **It flips.** Each step, with chance `flip_rate` (1/150), the good kind swaps. So no fixed preference can work, and the
  species brain cannot simply learn "eat soft plants".

A feed-forward brain sees only this moment, so it has no way to carry "that last bite made me sicker" forward. A brain with
memory (`--memory 64`, the GRU from Stage 4; memory resets at birth, so nothing is inherited) could taste and adapt.
This is learning by memory inside a trained network (as in RL²), not changing weights during life. Plastic weights
are left for later.

## Success test (can fail)

Seed 0 and seed 1, a memory run and a feed-forward run each, matched at 150 updates, so both see the same world-steps
(the memory brain is slower per step, so the cap is updates, not time).

`python -m scripts.lifetime --run-dir runs/GRU --baseline runs/FF --evolved` on 256 fixed seeds reports the share of bites
from the good kind, by steps since the last flip (0-10, 10-40, 40+). Both must hold on both seeds:

1. **Memory wins:** at 10-40 steps after a flip, the memory brain's good-bite share beats the feed-forward brain's by at
   least 0.10.
2. **It recovers:** the memory brain's share at 10-40 steps beats its own at 0-10 steps by at least 0.05.

Reported: all three bins for both brains, prey lifetime, gene means, and a 3-D replay.

## Result (2026-10-30): PASS on both seeds

Matched at 150 updates, evolved bodies, 256 fixed seeds. Share of bites from the kind that feeds, by steps since a flip:

| run | 0-10 | 10-40 | 40+ | prey lifetime |
|---|---|---|---|---|
| pp-life-gru-0 (memory) | 0.455 | 0.853 | 0.826 | 40.7 |
| pp-life-ff-0 (no memory) | 0.479 | 0.635 | 0.750 | 34.5 |
| pp-life-gru-1 (memory) | 0.487 | 0.759 | 0.737 | 35.7 |
| pp-life-ff-1 (no memory) | 0.477 | 0.637 | 0.746 | 36.0 |

Memory beats no memory at 10-40 steps by **+0.218 and +0.122** (bar +0.10; seed 1 only narrowly). Memory recovers by **+0.398 and +0.272** (bar +0.05). **PASS.**

**Controls, added after the result because brains with no memory also improve after a flip, so the pass alone could not show that memory was the cause
(`scripts.lifetime --controls`, not part of the bars):**

| | 10-40 share, seed 0 / 1 | next bite good after a bad bite, seed 0 / 1 |
|---|---|---|
| memory | 0.853 / 0.759 | 0.674 / 0.549 |
| the same brain, memory wiped every step | 0.658 / 0.603 | 0.482 / 0.455 |
| no memory | 0.635 / 0.637 | 0.498 / 0.498 |

Wiping the memory takes the same brain down to the no-memory level, and after a bad bite those brains are at a coin flip. The memory brain's animals switch.
So the gain comes from memory, and it shows inside one animal's life.

Limits:

- **Memory's gift is speed.** By 40+ steps, no-memory brains reach 0.75, against 0.83 and 0.74 for memory. Why they recover at all was not measured; animals with the
  wrong habit dying first, and plant cover shifting, are both possible.
- **Seed 1 is weaker,** and it shows no gain in prey lifetime (35.7 against 36.0).
- This is memory inside a trained network, with weights fixed during life. The ability to taste and remember was learned by the species across training.
  Plastic weights remain for later.

## Why do brains with no memory recover? (2026-10-30, `scripts.lifetime --why`)

The bins above count steps since the clock last restarted, which is a flip or the start of a round (when the good kind is drawn fresh). Counting bites after real flips only, memory's lead is larger: at 10-40 steps memory 0.888 / 0.788 against no memory 0.576 / 0.607.

Share of the good kind among edible plants, seeds 0 / 1:

| | whole map, 40+ | within 2 cells of grazers, 0-10 | 10-40 | 40+ | good-bite share, 40+ |
|---|---|---|---|---|---|
| no memory | 0.495 / 0.492 | 0.455 / 0.465 | 0.628 / 0.618 | 0.806 / 0.775 | 0.750 / 0.746 |
| memory | 0.336 / 0.396 | 0.567 / 0.532 | 0.501 / 0.525 | 0.374 / 0.462 | 0.827 / 0.737 |

- **No memory:** the map stays at half and half, yet what surrounds the grazers climbs to about 0.8, and their bites simply match it. The population ends up sitting on the good kind. Grazers born after a flip already do better than those born before it (0.68 against 0.46 at 0-10).
- **Memory:** the good kind around them gets scarcer (they eat it down), yet they still take it about 0.8 of the time, roughly twice what is nearby. That is choice.
- Removing the sense of their own energy costs no-memory brains 0.05-0.07 at 10-40, so energy acts as a small memory too. Low and high-energy grazers do not differ.

Hypothesis: without memory, the map is the memory. This is a correlation until Stage 7d (below in the roadmap) tests it by scattering the grazers.
