# Stage 16: should parents teach their young, or should the young learn by watching?

Date: 2026-11-02. Written before any of these runs, and before the film's vote has opened, so the vote can't colour the prediction. The film promises both arms get run and reported whichever way viewers vote.

## Why

Stage 14 tried passing knowledge down by copying weights. Lamarckian hunters ended up worse than their founders (14d). But real animals don't inherit what their parents learned as wiring: most of what passes between generations in a lifetime goes by behaviour. So here no weights are inherited at all. A newborn gets its parent's body genes and a blank brain, and whatever it knows, it has to get from its parent while the parent is alive and nearby.

## World

`tiny_stealth` (4,702 weights each), founders from `pp-ts-0` and `pp-ts-1`, 64 worlds, 1,000 updates, about 350 generations, seeds 0 and 1. Both species live by the same rule in each arm. The first animals start with the founders' trained brain, so there's something to pass on.

## Arms

- **genes (control):** `darwin`. A child gets its parent's birth brain plus noise, and no learning. Fresh runs on current code.
- **blank (floor):** a child gets the blank brain and nothing else.
- **watch:** for its first K steps, while its parent is alive and within sight (`view` cells), a child takes one learning step per game step towards what its parent just did **in the parent's situation**. Free for the parent. Information comes from whatever the parent happens to be doing.
- **teach:** the same K steps and the same condition, but the child learns what its parent's brain would do **in the child's own situation**, and the parent pays `teach_cost` energy for each lesson. Better information, at a price.
- **teach_free:** teach with no cost, reported to separate the information from the price.

No learning from one's own experience in any arm, and no learning after childhood, so these arms test passing knowledge on and nothing else. The blank brain is one fixed random network, the same for every animal in every arm, with a little noise per child to break ties.

## Settings fixed by a pilot, before the bars are applied

K, the learning rate and `teach_cost` get set from short pilots on seed 2, so nothing blows up (stage 14a needed this too). The pilot looks only at whether worlds survive and whether lessons happen. Values go in an addendum here before the full runs start.

## Test

Hunters from each arm against the **genes arm's grazers, frozen, same seed** (lesson 15). 64 worlds, 500 steps, no restarts. Hunters' culture stays live in the test: children are born blank and take lessons, because that's what the arm claims. Catches per hunter-step, on both seeds. Random hunters are the floor.

## Success test (can fail)

1. **Culture carries something:** watch and teach each catch at least 1.5x what blank catches.
2. **The film's vote:** teach catches at least 1.10x what watch catches. If watch beats teach by 1.10x, that's reported just as prominently, as the opposite answer. Neither by 1.10x means it's a tie.
3. **Culture against genes:** the better culture arm catches at least 0.9x what the genes arm catches.
4. **It lasts:** both species alive at step 1,999 in at least 0.5 of worlds in each culture arm's own world, with culture live.

Each is its own question, so there's no single stage pass. Also reported: catches over training (does knowledge fade down the generations, like a game of telephone?), the share of childhood steps with a lesson, how often a child's choice matches its parent's during lessons, and teach_free against teach.

## Prediction

Bar 1 passes for teach and misses for watch. A blank network with one sample per step gets too few lessons, and watch's samples come from a situation the child isn't in. Bar 2 passes, so teaching wins the vote's question on information. teach_free beats teach by at least 1.10x, so the price matters. Bar 3 fails: both culture arms land under 0.9x genes, because copying weights is close to perfect and lessons aren't. Bar 4 is the one I'm least sure of. Knowledge fades over the first few dozen generations and then holds at a lower level.

Real animals do teach, and it's rarer than you'd think. Caro and Hauser's 1992 definition requires a cost to the teacher, which is why the teach arm has one. I'm citing that from memory and it needs opening before it goes in a video.

## Addendum, settings from the pilots (2026-11-02, before any full run)

Pilots on seed 2, founders `pp-ts-0`, 150 updates, 64 worlds. "Agree" is how often a child's top choice matched the lesson, over the last 15 updates; chance is about 0.2.

| setting | watch: agree, hunters late, catches late | teach (cost 0.05): agree, hunters late, catches late |
|---|---|---|
| lr 0.02, K 32 | .22, 16.8, .42 | .27, 14.6, .36 |
| lr 0.1, K 32 | .22, 16.3, .41 | .29, 14.2, .35 |
| lr 0.3, K 32 | .44, 27.1, .79 | .44, 12.9, .31 |
| lr 1.0, K 32 | .86, 23.5, .66 | .97, 23.1, .66 |
| blank | | 16.8 hunters, .42 catches |

At the stage 14 rate of 0.02 a lesson moves a brain almost nowhere, and children agree with their teacher at about chance, so that would only test whether lessons that don't work, don't work. **Fixed for the full runs: lr 1.0, K 32, `teach_cost` 0.05, sigma 0.003, clip 1.0.** It's the only rate where both arms' lessons transfer; the same rate for both arms keeps it fair. Doubling K to 64 changed little at the low rate. A cost of 0.25 cut teaching worlds to about 9 hunters, so 0.05 it is; teach_free shows what even that costs.

**Warning noticed in the pilot, not a bar:** teach's agreement of .97 comes with weights barely moved from the blank brain (0.015 root mean square). High agreement can mean children copy parents who have themselves drifted back to near-blank. Agreement measures fidelity, not knowledge, so the bars use catches against the same grazers, as written. Random hunters are added to the test as the floor (`--random`).

## Result (2026-11-02): FAIL on bars 1 and 3, a tie on bar 2, bar 4 passes for watch and teach

**Test, against the genes arm's grazers, culture live, 64 worlds of 500 steps.** Catches per hunter-step, seeds 0 and 1:

| hunters | catches | against blank | against genes |
|---|---|---|---|
| random (floor) | .0134, .0132 | | |
| blank | .0143, .0139 | | |
| watch | .0144, .0150 | 1.01x, 1.08x | 0.39x, 0.44x |
| teach | .0150, .0135 | 1.05x, 0.97x | 0.41x, 0.39x |
| teach_free | .0143, .0149 | 1.00x, 1.07x | |
| genes | .0365, .0344 | | |

1. **Culture carries something: fail** for both (bar 1.5x). Every culture arm hunts at about random level against grazers with inherited brains, and their hunters are all but gone by step 500 (0.0 to 0.2 per world).
2. **The vote's question: a tie.** Teach against watch is 1.04x and 0.90x; neither reaches 1.10x on both seeds.
3. **Culture against genes: fail.** The better culture arm is 0.41x and 0.44x (bar 0.9x).
4. **It lasts, in its own world with both species' culture live, 2,000 steps:** watch .828, .828 pass; teach .578, .516 pass; teach_free .312, .531 fail; blank .250, .297 (reported).

**What happened, from training:** every culture arm crashed at about update 20, the same way and as far as blank did (catches about .3, hunters about 10 per world). That's when the founders die out. Children agreed with their teacher's choice 56 to 81% of the time in those first 20 updates, and it carried nothing the blank arm lacked. After the crash, watchers recovered in their own world to look almost like the genes arm (catches .71 and .68 against .96 and .79, hunters 26 and 24). **But that recovery was hunters and grazers getting worse together.** Against grazers with inherited brains they're random. That's lesson 15 a third time: a population's number in its own world isn't evidence of skill.

Teaching agreed 97% by the end and its weights sat about 0.01 from the blank brain, so children copied parents who had themselves drifted back to near-blank. Faithful copying of nothing. Watching moved weights four times as far (0.042) and agreed 85 to 88%.

**My prediction:**
- Bar 1, wrong for teach. I expected it to pass.
- Bar 2, wrong. I expected teaching to win, and it tied.
- teach_free beating teach, mixed (0.95x, 1.10x).
- Bar 3, right.
- Bar 4, where I was least sure, passed for watch and teach.
- "Knowledge fades over the first few dozen generations": too optimistic. It went with the founders, in about 20 updates.

**Limits:** one setting from the pilots (lr 1.0, K 32, one sample per step), on a 4,702-weight brain. Animals learn over longer childhoods, with practice, and with instincts they're born with. This shows that lessons alone, at this size and rate, don't hold onto what the founders knew; it doesn't show culture can't work.

**Next, not queued:** genes plus lessons, a child born with its parent's inherited brain and then taught, which is how real animals do it.
