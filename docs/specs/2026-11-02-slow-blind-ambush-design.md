# Stage 18b: does an ambush predator evolve poor sight and slow legs?

Date: 2026-11-02. Written before any of these runs. A viewer asked on Part 2 how long it would take an ambush predator to evolve with low vision and low move speed, letting the prey come to it.

## Why

Existing runs nearly answer it, but in the wrong worlds. With walking costs and thickets (`lasting_ambush`, `pp-la-*`), hunters' speed ends at about 0.09 against 0.16 without them (`lasting`, `pp-last-*`), and sight at about 0.09 in both. Those hunters barely survive, at about 8 per world, so low genes there may just be starvation. The question is more interesting where hunters can make a living, and that needs camouflage (stage 10f).

## World

`ambush_stealth`: `lasting_stealth`, plus stage 8b's walking cost of 0.5 and thickets on a quarter of the map. Seeds 0 and 1, 150 updates, 48 x 48. The control is `lasting_stealth` on the same seeds from stage 18a (`pp-mw-1x-*`), which has neither walking costs nor thickets. Founders' speed is 0.2 and sight 0.5.

## Success test (can fail)

Hunters' genes over the last 15 updates, on both seeds:

1. **Slow legs:** speed at least 0.05 below the control.
2. **Poor sight:** sight at least 0.10 below the control.
3. **It's a living:** both species alive at step 1,999 in at least 0.5 of worlds in `ambush_stealth`.

Bars 1 to 3 together answer the viewer: yes, it evolves, in a world where the hunter makes a living.

**How long:** the first update where each gene has fallen within 0.02 of its final value, reported, since that's what the viewer asked. Also reported: `scripts.ambush`'s share of time hunters spend standing still and in thickets, a hunter to follow, and catches against the same grazers if the numbers make that worthwhile.

Prediction, the same guess I posted in reply on Part 2: they keep slow legs and lose the eyes, so bars 1 and 2 pass. Bar 3 is less sure, since walking costs come out of the same thin budget.

## Result (2026-11-02): bars 1 and 2 pass, bar 3 fails

Hunters' genes over the last 15 updates, seeds 0 and 1:

| | speed | sight | hunters per world |
|---|---|---|---|
| `ambush_stealth` | .206, .213 | .321, .346 | 6.2, 6.2 |
| `lasting_stealth` control | .298, .314 | .437, .557 | 29.0, 25.9 |
| gap | .092, .101 (bar .05): pass | .116, .211 (bar .10): pass | |

**Bar 3 fails:** both species alive at step 1,999 in 0.000 and 0.000 of worlds (bar 0.5). So this isn't the answer the spec asked for, in a world where the hunter makes a living. The worry in "Why" holds here too: these may be starving hunters and not ambushers.

**The prediction half held, and the bars hide a reversal.** Both genes fell fast and then came back. Speed went .196 to .102 and .114 by update 16, then back to .229 and .233 by update 46, ending about at the founders' 0.2. Sight went .496 to .142 and .268 by update 31, then settled near .33. So the legs are only slow against the control, which got faster; they aren't slower than the founders'. Sight really did fall, by about a third. I don't know why the genes turned back; catches per step doubled over the same stretch (.076 and .104 at update 16, about .2 from update 46), which fits but doesn't show a cause.

**How long:** the planned measure misleads, since it reports speed at update 1 (it passes through the final value on the way down) and sight at updates 12 and 11 (it went on falling after that). Read from the curve instead: the fall took about 15 to 30 updates and things settled by about update 46.

**Ambush behaviour (`scripts.ambush`, founders' bodies):** in thickets 38% of the time against 26% for random (1.50x, 1.52x, bar 1.3x); standing still 1.11x, 1.38x (bar 1.5x); catches in thickets no better seen than unseen (5.5% less, 11.7% more, bar 10%); so it fails on the same terms as earlier ambush stages. Against trained prey the learned hunters still beat the chaser rule (chaser .69x, .72x).

**Next:** making waiting pay would take a cheaper walking cost or richer meals, and the viewer gets asked what they'd change. Not queued.
