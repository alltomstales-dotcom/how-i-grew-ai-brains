# Stage 18a: dumber than a mealworm?

Date: 2026-11-02. Written before any of these runs. From a viewer's comment on Part 3: the net has fewer neurons than a mealworm, and it's just random garbage signals in and out.

## Why

There are two claims in that, and only one is testable here. I don't have a sourced neuron count for a mealworm, and the net may well be smaller. So the size claim stays unsettled. "Random garbage" I can measure, and so can a related question it raises: would a much bigger brain do better?

## World

`lasting_stealth`, with hunters' camouflage by rule, meals of 25, 48 x 48. Three sizes of brain, each trained by the shared trainer for 150 updates, seeds 0 and 1:

- small: `tiny_stealth`, already run (`pp-ts-0`, `pp-ts-1`), 4,702 weights.
- usual: `lasting_stealth`, retrained on current code (`pp-mw-1x-*`), 115,750 weights.
- big: `brain_big`, 4x the conv width and hidden size, 1,802,374 weights, about 16x the usual (`pp-mw-16x-*`).

## Test

No extra training. `scripts.cross_shared`: hunters from one run, with their genes and brain, against grazers from another, with theirs, frozen, 64 worlds of 500 steps. **All hunters face the usual brain's grazers from the same seed,** following lesson 15. Random hunters are the floor, with the same grazers and the usual hunters' genes.

## Success test (can fail)

Catches per hunter-step, on both seeds:

1. **Not random garbage:** usual hunters catch at least 3x what random hunters catch.
2. **A big brain helps:** big hunters catch at least 1.10x what usual hunters catch.
3. **A small brain costs:** usual hunters catch at least 1.10x what small hunters catch.

Each is its own question, so there's no single stage pass. I'll report all three. Bar 1 is what the Short leads on.

Prediction: bar 1 is met by a lot. Bars 2 and 3 both miss: this world doesn't ask much, stage 7a found brain size tracks food and not the value of thinking, and the small brain has already kept up. So the funny answer may be true: a brain about 25 times smaller than the usual one, and about 380 times smaller than the big one, does about as well, because the problem is simple, which is lesson one. Reported, not bars: both species alive at step 1,999 in each size's own world, and a named hunter to follow from each.

## Result (2026-11-02): bar 1 passes, bar 2 fails, bar 3 passes, and my prediction was half wrong

Catches per hunter-step against the usual brain's grazers from the same seed, 64 worlds of 500 steps, seeds 0 and 1:

| hunters | catches | against the usual brain |
|---|---|---|
| random | .0112, .0130 | |
| small, 4,702 weights | .0349, .0328 | 0.78x, 0.71x |
| usual, 115,750 weights | .0445, .0459 | |
| big, 1,802,374 weights | .0465, .0490 | 1.04x, 1.07x |

1. **Not random garbage: pass.** Usual hunters catch 3.97x and 3.53x what random hunters catch (bar 3x).
2. **A big brain helps: fail.** 1.04x and 1.07x (bar 1.10x). About 16 times the weights buys 4 to 7% more, for about three times the training time.
3. **A small brain costs: pass.** Usual catches 1.28x and 1.40x what small catches (bar 1.10x).

**I predicted bars 2 and 3 would both miss, so "the small brain does about as well" was wrong.** That line also leaned on earlier comparisons where hunters faced different grazers; lesson 15 applies to this mistake too. The diminishing return is real, though: going from 4,702 to 115,750 weights buys 28 to 40%, and from 115,750 to 1,802,374 buys 4 to 7%.

Both species alive at step 1,999: usual .988, .867; big .867, .977. The size claim stays unsettled: I still have no sourced neuron count for a mealworm, and weights aren't neurons.

**A process note:** the big brain ran out of device memory at startup at 16 GB, three tries each, and was rerun one seed at a time at 32 GB with the same settings.
