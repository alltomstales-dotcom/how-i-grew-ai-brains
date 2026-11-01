# Stage 11c: how much do migrants really contribute?

Date: 2026-10-31. Written before any of these runs.

## Why

Stages 11a and 11b measured migrants' descendants with a tag that every world's restart wiped, so they could not say how much gene flow there was (see the correction in `2026-10-31-continents-design.md`). I told viewers and Tom these worlds were nearly isolated, and that's unsupported.

## World

Stage 11b's, with one change: a lineage tag now passes through the gene pool, so it survives a restart. Sets E (seeds 30, 31, 32) and F (seeds 33, 34, 35), 600 updates each, same crossing rate, displacing and pace, with live clips.

## Success test (can fail)

Scored at the same phase of every episode, to take out any remaining sawtooth: in `transit.jsonl`, the mean of the last 10% of training, using only updates in the last 100 steps of an episode. On both sets:

1. **The measure works:** the share no longer collapses at restarts: the mean in the first 100 steps of an episode is at least half the mean in the last 100, across the second half of training, on meadow and tundra grazers.
2. **Gene flow is real:** at least 0.20 of meadow's and tundra's grazers descend from arrivals. A rough sum says about 3% an episode could compound to well above this.
3. **Climate holds under it:** grazer sight comes out meadow over forest over tundra, by at least 0.15.

Reported, not bars: hunters' shares, whether tags saturate (every line eventually marked), which continent's lines spread where, and whether arrivals do better or worse than residents, the question I answered too soon.

## Result (2026-10-31): FAIL by the rule, on meadow's share, and the strongest result so far

| | set E | set F | bar |
|---|---|---|---|
| 1. measure works, grazers, first and last 100 steps of an episode: meadow | 0.108, 0.093 | 0.163, 0.153 | first at least half of last: met |
| 1. tundra | 0.668, 0.665 | 0.870, 0.864 | met |
| 2. grazers descended from arrivals: tundra | 0.722 | 0.964 | at least 0.20: met |
| 2. meadow | 0.101 | 0.097 | at least 0.20: **missed** |
| 3. grazer sight, meadow / forest / tundra | 0.729 / 0.239 / 0.106 | 0.741 / 0.226 / 0.107 | same order, range at least 0.15: met |

Hunters descended from arrivals: meadow 0.513 and 0.423, forest 1.000 and 1.000, tundra 0.068 and 0.157. Forest's grazers: 0.121 and 0.069. Both species at step 1,999 with no crossing: meadow and tundra 1.000, forest 0.000, on both sets.

**These worlds are far from isolated, and climate holds anyway.** On set F, 96% of tundra's grazers descend from arrivals, all reaching tundra from forest, whose grazers average about 0.23 in sight, and yet tundra's grazers sit at 0.107, the same as 0.097 to 0.118 on every other set. So the difference between climates is kept up against heavy immigration. The natural reading is selection pulling each arrival's line back toward what that climate favours, generation by generation. That is my inference; I haven't traced lineages, and a tag also marks any line that ever arrived, so a line born here that went round the ring counts too.

**Grazer sight has now kept this order on all six sets:** 0.759 / 0.325 / 0.118, 0.682 / 0.323 / 0.097, 0.372 / 0.219 / 0.113, 0.704 / 0.254 / 0.103, 0.729 / 0.239 / 0.106, 0.741 / 0.226 / 0.107.

**Flow is lopsided, and I can't say why yet.** Tundra's grazers and forest's hunters are mostly or entirely arrivals, while meadow's grazers mostly stay native. Forest's hunters look like a sink: they die out on their own (0.000 with no crossing), and in training every one of them descends from an arrival, so forest may be holding hunters only because they keep arriving. Whether tundra's grazers are replaced because of winter crashes isn't measured.

**Withdrawn:** my statements that migrants left few descendants, and from set C alone that arrivals seem to do much worse than residents. Nothing here is a claim about real animals or climates.

