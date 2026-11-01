# Stage 11b: do climate's differences survive real migration?

Date: 2026-10-31. Written before any of these runs.

## Why

Stage 11a found grazer sight and hunter speed in the same order across climates on two sets of seeds. But migrants left at most 4.8% of any continent, so those worlds were close to isolated, and two flaws in the design made that worse: about 60 to 80% of grazer migrants were lost at full worlds, and the faster machine finished in half the time, so crossing stopped early.

## World

As stage 11a, with the flaws fixed:

- A migrant reaching a full world replaces a random resident (`--transit-displace`).
- Every trainer takes at least 4.2 seconds per update (`--min-update-seconds`), so all three stay in step and finish together. The slowest measured was about 4.0.

The crossing rate stays 0.2 per update from the 2 cells at the east edge. Sets C (seeds 20, 21, 22) and D (seeds 23, 24, 25), 600 updates each, with `--live-every 3`, which publishes clips for the phone viewer.

## Success test (can fail)

Both on both sets:

1. **Migrants matter this time:** at the end of training at least 0.05 of living animals on meadow and on tundra descend from arrivals, grazers and hunters each (mean of the last 10% of updates).
2. **Climate still shows:** grazer sight comes out meadow over forest over tundra, with at least 0.15 between highest and lowest.

Bar 1 leaves out forest, since its hunters are few and die out without restarts, which stage 11a already showed; forest is reported. Both continents must also still last, as reported, and not as a bar.

Prediction: bar 1 passes with displacing, and migration narrows climate's differences, with bar 2 still passing at a smaller range. I hold this loosely.

Reported, not bars: all the stage 11a numbers, whether arrivals do better or worse than residents, the size of the gene gap against stage 11a's, and whether live clips slow training.

## Result (2026-10-31): FAIL by the rule on bar 1, whose measure was flawed; bar 2 met

| | set C | set D | bar |
|---|---|---|---|
| grazer sight, meadow / forest / tundra | 0.372 / 0.219 / 0.113 | 0.704 / 0.254 / 0.103 | same order, range at least 0.15: met |
| descended from arrivals, meadow grazers, hunters | 0.022, 0.012 | 0.019, 0.010 | at least 0.05: missed |
| descended from arrivals, tundra grazers, hunters | 0.004, 0.000 | 0.000, 0.000 | at least 0.05: missed |

Both meadow and tundra last (1.000 on both sets) and forest doesn't (0.000 on both). Fixes worked: nearly all migrants now land, and all three trainers stayed in step (46 to 57 minutes).

**Bar 1 could not measure what it claimed.** See the correction in `2026-10-31-continents-design.md`: restarts wiped the lineage tags. I told Tom from set C alone that arrivals "seem to do much worse than residents". That is withdrawn.

**Meadow's sight is unsteady.** On set C it read 0.78 at update 449 and 0.23 at update 599, so its low mean of 0.372 comes from a late swing, and I was wrong to suggest from it that migration was pulling meadow down. Set D ended at 0.704. Set B of stage 11a also hinted at two groups in this gene in meadow.

