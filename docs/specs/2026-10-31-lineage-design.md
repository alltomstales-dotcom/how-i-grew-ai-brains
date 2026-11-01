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
