# Stage 15f: place or mix, with enough sets to tell

Date: 2026-11-01. Written before any of these runs.

## Why

Stage 15e: mutation zones' mix applied at random scored 1.000 and 0.000, against zones' 1.000 and 1.000. The bar missed, which says place matters, but one set swung the result from one end to the other, so two sets can't settle it.

## World

As stages 15d and 15e: isolated tundra, `mut_zone_tundra` and `mut_random_tundra`, 600 updates, on four fresh seeds, 60 to 63. Sets C2 to F2.

## Success test (can fail)

Both species alive at step 1,999, `scripts.persist --evolved --steps 2000 --episodes 32`, averaged over the four sets:

1. **Place matters:** zones at least 0.25 above the random mix.

If it's met, place does something the mix alone doesn't. If it isn't, then on this measure the mix is about as good, and what helps is that mutation is uneven, not that it's environmental. Either way I'll report every set, not just the means, and pool with stage 15e's two sets as a reported figure, not a bar.

Prediction: not met. Zones have looked steady so far (0.969, 0.656, 1.000, 1.000), and that's the case for place, but I still expect the mix to account for most of it. Reported, not bars: hunters' gene and numbers on every set.

## Result (2026-11-02): FAIL, and it's the mix

All eight trainers reached update 600. Both species alive at step 1,999:

| | C2 | D2 | E2 | F2 | mean |
|---|---|---|---|---|---|
| `mut_zone_tundra` | 1.000 | 0.969 | 0.938 | 1.000 | 0.977 |
| `mut_random_tundra` | 0.969 | 1.000 | 0.938 | 1.000 | 0.977 |

**Place matters: missed.** The difference is 0.000 on the mean, against a bar of 0.25, and my prediction held. Hunters' gene (0.883 to 0.892) and numbers (261 to 353) don't separate them either.

Pooled with stage 15e's two sets, as reported: zones 0.984, random 0.802, where the gap is all from one run at 0.000. One in six random runs failing that way may mean something, and with six runs it can't be told from chance. Even mutation over its own four sets averages 0.273, on different seeds.

**So on this measure, what helps is that mutation is uneven, not that it's environmental.** Most children mutate a little and a few mutate a lot, and worlds last far better than when every child mutates the same amount at the same average. Where it happens made no difference that this test can see. That's the answer to Tom's question as tested here: environmental mutation helps, for a different reason than it sounds.

Why the mix helps isn't tested. My guess is that most animals carry less mutation damage while a few supply variation. It's all in isolated tundra at one price, with one size of mix.

