# Stage 13d: are brains better at their own climate?

Date: 2026-11-01. Written before any of these runs. Tom asked whether the continents are making brains better suited to their climate, or just adding noise. Bodies clearly differ. For brains I've never tested it.

## Why

There's one brain per species per continent, trained there. The only measure so far, learned hunters against a fixed chaser, is about the same in every climate, which can't tell a climate-specific brain from a general one. The standard test for local adaptation is a reciprocal transplant: move each one to the other's home.

## The control this needs

A brain from another run might do worse anywhere, just for not having trained against those particular partners. So comparing a home brain with a visitor mixes climate with that. The fair comparison is between two foreign brains, both from the other set: one from the same climate and one from the other climate. If the same-climate one does better, climate is what's making the difference.

## Test

No training. `scripts/transplant.py` runs `agents.evaluate.evaluate` with 32 worlds of 500 steps each, on the target run's world and gene pool, so bodies and climate are always the target's. Only the focal species' brain changes, and the other species keeps the target's brain.

Runs: sets I2 and J2 of stage 12g, meadow and tundra. Forest can't take part: thickets give its brains an extra input, so they don't fit.

For each of four targets (meadow and tundra, in each of two sets) and for each species, three brains: its own, the other set's from the same climate, and the other set's from the other climate.

Measures: for hunters, catches per hunter-step, where higher is better. For grazers, catches per hunter-step against them, where lower is better, and their lifetime.

## Success test (can fail)

**Local adaptation in hunters:** the same-climate foreign brain catches at least 1.10x what the other-climate foreign brain catches, in all four targets.

**Local adaptation in grazers:** the same-climate foreign brain lets hunters catch at most 1/1.10 of what they catch against the other-climate foreign brain, in all four targets.

The stage passes if either does. If neither does, climate isn't shown to shape brains in this measure, and the continents stay an experiment on bodies and gene flow.

Prediction: grazers pass and hunters don't. Winter changes what a grazer needs to do far more than it changes chasing one. I hold this loosely. Reported, not bars: each brain against its own home brain, which shows the cost of being foreign, and the same comparisons on sets E and F (no size).

## Result (2026-11-01): FAIL on both bars, with a one-way effect in hunters

Catches per hunter-step. Ratios compare the same-climate foreign brain with the other-climate foreign brain, so above 1 favours its own climate.

| target | hunters: home | same-climate | other-climate | hunter ratio | grazer ratio |
|---|---|---|---|---|---|
| meadow I2 | 0.0290 | 0.0288 | 0.0275 | 1.05x | 0.90x |
| meadow J2 | 0.0285 | 0.0283 | 0.0291 | 0.97x | 0.88x |
| tundra I2 | 0.0520 | 0.0529 | 0.0434 | **1.22x** | 1.03x |
| tundra J2 | 0.0530 | 0.0529 | 0.0456 | **1.16x** | 0.99x |
| meadow E, no size | 0.0511 | 0.0529 | 0.0383 | 1.38x | 1.09x |
| meadow F, no size | 0.0430 | 0.0425 | 0.0431 | 0.98x | 1.07x |
| tundra E, no size | 0.0509 | 0.0501 | 0.0459 | 1.09x | 1.00x |
| tundra F, no size | 0.0468 | 0.0462 | 0.0417 | 1.11x | 0.99x |

Both bars miss. My prediction was backwards: I said grazers would pass and hunters wouldn't.

**Training is repeatable.** Hunters from another run of the same climate catch within 1.7% of the home brain on six of eight targets, and 3.5% at worst. So being foreign costs a hunter almost nothing, and the control I added turned out not to matter much for them. Those differences also give a rough noise floor for this measure, of a few percent.

**Hunters show a one-way effect.** In tundra, brains trained in tundra beat brains trained in meadow on all four targets, by 9% to 22%. In meadow there's no steady difference: 1.05x, 0.97x, 1.38x, 0.98x. So tundra's brains work about as well in meadow as meadow's own, and meadow's lose out in tundra. That isn't a trade-off between climates. It looks more as if tundra's training teaches something general that meadow's doesn't, though I noticed this after seeing the results and it isn't tested. Meadow E's 1.38x doesn't repeat on F.

**Grazers show no home advantage.** In the size sets, hunters catch more of meadow's own grazers than of tundra's visiting ones, 0.90x and 0.88x, and those visitors live longer, about 100 steps against 80. Meadow-trained grazers also outlive tundra-trained ones in tundra, on all four targets. Lifetime isn't fitness, since a grazer that lives long may breed less, so I wouldn't read much into that without births.

**What this says about the continents.** Climate shapes bodies far more clearly than brains. I can't say brains adapt to their own climate. What I can say is that one climate trains hunters in a way that travels better than the other.

Limits: 32 worlds a cell, target bodies always, two climates only, and one measure of skill. It doesn't test whether a tundra brain would still be better after retraining alongside these partners.

