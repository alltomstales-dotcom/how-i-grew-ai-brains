# Results, stage by stage

Each stage adds one pressure to the world. Its pass bar is fixed in a spec before any run, and it stays
where it was even after a near miss. The spec holds the full numbers and the caveats.

**The score so far: 6 stages or sub-stages pass and 9 fail, with one still training.** Counting each new version of a
failed world as its own attempt, there are 16 failures. Some passes are weaker than others, as the notes say.

| Stage | The question | The bar, set first | Result | Spec |
|---|---|---|---|---|
| 1 | Can two species learn from nothing? | Predators catch 2x as often as random ones; prey live 1.5x as long | **PASS:** 3.36x and 1.61x. Then a stalemate: prey dodge nearly every attack and hunters starve | [spec](specs/2026-10-14-predprey-crucible-design.md) |
| 2 | Do bodies and plants evolve? | Evolved lineages out-compete founders, and a plant gene shifts | **PASS, weak alone:** a first version passed with untrained brains, and pp-evo-002's random-brain control passes too, so the finding is the comparison with it. A later run (pp-evo-003) failed this test | [spec](specs/2026-10-14-evolution-design.md) |
| 3a | Do prey learn to use burrows? | Trained prey hide near predators 1.5x as selectively as random prey | **FAIL:** ratio 1.05 against 1.20 for random prey | [spec](specs/2026-10-14-terrain-design.md) |
| 3b | Does training against past opponents stop the hunters stalling? | Predator Elo rises by 100 | **FAIL:** -11 to +77. Hunters stall against prey that never learn too, and a hand-written chase does 0.83-1.06x as well in the evolution world. It's a ceiling on the world, not a learning failure | [roadmap](roadmap.md#stage-3-portable-brains-and-a-league-in-progress) |
| 4 | Does memory help in a seasonal world? | 1.2x on prey lifetime or catches, at matched world-steps | **FAIL:** 0.99x and 1.13x. Memory hunters learn faster per step, but feed-forward catches up | [spec](specs/2026-10-14-memory-design.md) |
| 5a | Can one founder pool split into two species? | Separation D of at least 2, minority of at least 0.2, gap of at least 0.3, in the last 3 snapshots | **FAIL, six versions:** the last was a near miss (D 1.95 on one snapshot). Without mating two size groups hold apart; two-parent breeding prevents it | [spec](specs/2026-10-14-speciation-design.md) |
| 5b | If a meal is worth what the prey stored? (a viewer's idea) | Hunters pick fat grazers 1.15x as selectively as the control, and meal value falls 10% when they can't see fat, on two seeds | **FAIL, twice:** 1.10x and 0.97x, then 1.07x and 1.00x in a fair retest. Hunters don't choose fat prey, and we can't yet say why | [spec](specs/2026-10-30-fatprey-design.md) |
| 6a | If a catch takes two hunters, do they learn teamwork? | 1.5x a lone chaser | **FAIL:** 1.02-1.04x. Chasers meet at the nearest prey anyway | [spec](specs/2026-10-14-pack-hunting-design.md) |
| 6b | Do animals learn to signal? | Catches drop 10% with signals blanked | **FAIL:** under 1%. A later review found signals could barely reach beyond sight, which weakens this result | [spec](specs/2026-10-14-signals-design.md) |
| 6c | If hunters must surround prey? | 1.5x a lone chaser, on two seeds | **PASS:** 3.0-3.8x in all eight readings. The first stage where learning clearly beats a simple rule. Signals help evolved hunters; in founder bodies only on one seed | [spec](specs/2026-10-14-flanking-design.md) |
| 7a | Does evolution buy brain where thinking pays? | Hunter brain gene 0.1 higher in flank than in pack, on two seeds | **FAIL, three tries:** at a flat cost brain size follows food, not the value of thinking (0.19 flank, 0.50 pack) | [spec](specs/2026-10-14-brain-size-design.md) |
| 7b | Same question, with the brain taxed as a share of each meal? | Gap of +0.12 | **PASS:** 0.53 against 0.18 and 0.69 against 0.38 on seed 0; 0.47 against 0.14 on seed 1 | [spec](specs/2026-10-29-brain-tax-design.md) |
| 7c | Can an animal learn within one life, when food turns to poison at random? | Memory beats no memory by 0.10 at 10-40 steps after a flip, and beats itself at 0-10 by 0.05, on two seeds | **PASS:** +0.218 and +0.122 (seed 1 only narrowly); recovery +0.398 and +0.272. See the limits below | [spec](specs/2026-10-30-lifetime-design.md) |
| 7d | Do brains with no memory use the map as their memory? | Scattering the grazers drops their good bites by 0.15, on two seeds, and memory brains drop less than half as much | **FAIL:** drops of 0.035 and 0.076. Survivorship explains the plants around them, as worlds where the rare plant feeds die out, but only 9-52% of the rise in good bites. The rest is unexplained. This changes how 7c should be read | [spec](specs/2026-10-30-scatter-design.md) |
| 8a | Do hills matter? | 10% worse with the height channel zeroed | **FAIL:** predators 4.8% and -5.4%, prey 2%. Hunters do take the high ground (78% against 64% at random) in plain hills | [spec](specs/2026-10-29-hills-design.md) |
| 8b | Do hunters learn to ambush, if walking costs energy and thickets hide them? (a viewer's idea) | 1.3x the map's thicket share, 1.5x random's stillness, 10% fewer catches when thickets stop hiding them, on two seeds | **PASS:** 64-65% in thickets (map 25%), 90-93% still (random 20%), 68-72% fewer catches. See the limits below | [spec](specs/2026-10-30-ambush-design.md) |
| 9a | If hunters can eat plants, do they leave meat, and does a poor gut keep them hunting? | Diet 0.15 below a control where the gene does nothing, and 0.10 higher with a poor gut, on two seeds | **FAIL on the second:** diet fell to 0.10 (control 0.46) and they stopped hunting, but a poor gut went lower still, 0.05, as hunters fell from 32 to 5 | [spec](specs/2026-10-30-omnivore-design.md) |
| 9b | How bad must a gut be to keep a hunter hunting? | Diet at least 0.50 at a gut of 0.03, and at most 0.25 at 0.20, on two seeds | **PASS:** 0.90 and 0.91, then 0.07. A sharp switch between 0.08 and 0.20, a little above our rough sum of 0.07, and only 2-3 hunters per world | [spec](specs/2026-10-31-gut-ladder-design.md) |
| 9c | Do winters make real omnivores? | A diet of 0.30-0.70, plants 0.20-0.80 of hunter energy, one group, and a no-winter control outside that range, on two seeds | **FAIL:** diet 0.11 and 0.12, plants 0.93. Winter also cut grazers from 31 to 11 | [spec](specs/2026-10-31-winter-design.md) |
| 9d | Do hunters split into two kinds, and live together? | Two diet groups (D 4, smaller group 0.2, gap 0.6), and both kinds alive at step 400 in 0.25 of worlds with hunters, on two fresh seeds | **FAIL by the rule, with a real split:** D 5.9 and 6.7 at a gut of 0.11. No world had hunters at step 400, a step we chose badly. They take turns rather than live together | [spec](specs/2026-10-31-two-diets-design.md) |
| 9e | Is that split made by each world's rich start? | The split repeats at the usual start and is gone at a lean one, on two fresh seeds | **FAIL by the rule, narrowly:** it repeated in kind (D above 5) but one seed's smaller group dipped to 0.16 under the 0.2 bar. A lean start erased it on both. The meat group vanished, not the plant eaters as we guessed | [spec](specs/2026-10-31-bonanza-design.md) |
| 10a | Is there a world that lasts, for hand-written animals? | Both species alive at step 2,000 in half of worlds, on two seed sets | **PASS, with a caveat:** all 12 settings with plants spreading at 0.2, none of the 24 slower. Those worlds sit near the slot caps of 96 and 32, which may be what holds them up | [spec](specs/2026-10-31-persist-design.md) |
| 10b | Do trained brains last in that world? | Both species alive at step 2,000 in half of worlds, and hunters beating a fixed chaser, on two seeds | **FAIL on survival:** 0.000 of worlds. The world lasts and hunters don't | [spec](specs/2026-10-31-lasting-brains-design.md) |
| 10c | Does a big enough meal let trained hunters last? | At 250, both alive in half of worlds and hunters beating a chaser; at 100, not, on two seeds | **FAIL:** 0.066 and 0.098 at 250. A probe shows trained grazers are what kill hunters | [spec](specs/2026-10-31-big-meals-design.md) |
| 10d | Can a viewer's walking costs and thickets make a world last? | Both alive in half of worlds, grazers under the cap, hunters beating a chaser, on two seeds | **FAIL:** 0.000 of worlds at meals of 25 and 100 | [spec](specs/2026-10-31-costly-dodging-design.md) |
| 10e | Do hunters built for the chase last? | Same three bars as 10d | **FAIL:** they evolved speed, and both species last in 0.000 and 0.008 of worlds | [spec](specs/2026-10-31-sprint-design.md) |
| 10f | Can grazers dodge what they can't see coming? (camouflaged hunters) | Same three bars as 10d | **FAIL by the rule, on grazers at the cap, and the first trained brains in a world that lasts:** 0.957 and 0.961 of worlds at step 2,000, against a best of 0.098 | [spec](specs/2026-10-31-camouflage-design.md) |
| 10g | With three times the slots, what sets the numbers? | Both last, grazers and hunters each under 0.9 of their caps, hunters beat a chaser | **FAIL by the rule, on hunters at their cap:** both last (0.984, 0.996), and grazers settle at 113 of 288, held down by hunters | [spec](specs/2026-10-31-room-design.md) |
| 11a | Do different climates make different animals? (Tom's idea; three connected worlds on two machines) | All three last, 5% descended from migrants, a gene differs by 0.15 in the same order on two sets | **FAIL on two of three bars, and climate clearly shaped bodies:** grazer sight 0.76 / 0.33 / 0.12 and 0.68 / 0.32 / 0.10. Forest's hunters die out, and migrants left few descendants | [spec](specs/2026-10-31-continents-design.md) |

## Limits worth repeating

These are the caveats most easily lost when a result gets shortened.

**Ambush (8b): hiding only matches chasing per meal.** A one-line chaser catches 1.4-1.7x as much as the trained
hunter and burns about two thirds more energy per step, so per meal the two are about even. Learning found a second
way to make the same living, not a better one.

**Lifetime learning (7c): "memory is mostly speed", and why that is now in question.** Good bites by brains with no
memory rise later too: by 40+ steps they reach 0.75, against 0.83 and 0.74 with memory. So 7c concluded that memory's
gift is speed. Stage 7d then found part of that rise is survivorship, not recovery: their evolved plant pools are about
85% shrubs, worlds where the rare kind feeds die out, and about nine in ten living worlds at 80+ steps are the lucky
kind. We first wrote "most of it". A later check found that selection explains only 9-52% of the rise in good bites,
depending on seed and how it is split, so the rise inside each kind of world may be the larger part, and it remains
unexplained. The two brains also did not face the same plants,
so the no-memory comparison is weaker than it looked. Whether memory gives more than speed was not separately tested.
The result stands on the cleaner control below. Seed 1 is weaker, and shows no gain in prey lifetime (35.7 against 36.0).

**Lifetime learning (7c): this is memory inside a trained network, not changing weights.** A brain's weights are fixed
during an animal's life. What changes is the memory state, and the ability to taste and remember was learned by the
species across training. Plastic weights remain for later. The controls here were added after the result: wiping the
memory every step takes the same brain back down to the no-memory level (0.66 and 0.60 at 10-40 steps), and after a
bad bite those brains are at about a coin flip.

**Flanking (6c): "smart" means smart in this rule.** Two hunters must stand exactly opposite the prey. It shows that
coordination is learned when a simple rule fails, not that hunters are generally clever. Hills cut the edge to 2.3x.

**Brain tax (7b): a fair cost changes the answer.** The flat-cost result reversed under a tax. The result rests on the
idea that a share of income is the fair comparison, and that was chosen by us.

**Stored meals (5b): the first test was unfair.** We assumed prey held about 20 energy, and they held about 9.5, so a meal paid about 12 where the control's paid 25. The fair retest failed too, so the idea is closed here. Grazers do vary a lot (energy 9 with a spread of 5), though that is across a world, not what one hunter sees at one moment. This says nothing about real predators.

**Omnivores (9a): one bar passed easily and one failed backwards.** The poor gut was meant to keep hunters hunting, and that was my prediction. Even where the gene does nothing, plants give hunters 0.97 of their energy, so the brains stop hunting without any change in the gut. The next stage tests a rough sum that puts the switch at a gut of about 0.07.

**Two diets (9d): a real split, and not what "two species" sounds like.** The hunter pool holds two distinct kinds, steadily, on fresh seeds. But in any one world plant eaters boom and crash by step 100, meat eaters outlast them, and hunters die out by about step 300. Worlds restart from the mixed pool, and that may be what keeps both going, which stage 9e is testing. The second bar failed because we picked step 400 without checking how long hunters last. We left the bar alone. We also tightened the first bar before training, because drift nearly passed stage 5a's version on this gene.

**Every world so far has died (10a).** A test we were about to run showed grazers dying out as well as hunters, even hand-written ones with plants on the map, and plant cover falling from 0.50 to 0.09 whoever was eating. So every world in this project has been a boom and bust of about 150-300 steps, hidden by restarts from a saved pool. Earlier results measured what happens in those worlds, which limits what they can say about lasting populations. We withdrew that test before spending training on it. Plants spreading faster fixes this for simple animals; whether it does for trained brains is stage 10b.

**Trained grazers kill the hunters (10c).** Swapping in hand-written animals settled this on the runs we tried: against hand-written grazers trained hunters fill every world to step 2,000, and hand-written hunters die against trained grazers. It's the same problem Part 1 ended on. Stage 10d's walking costs and thickets, a viewer's idea that passed its own test in stage 8b, didn't solve it. Nothing so far has given trained brains a world that lasts.

**Three continents (11a): what it does and doesn't show.** Two sets of fresh seeds gave the same order of genes across climates, and that repeat is the evidence. Migrants left almost no descendants, so the continents were close to isolated, and this doesn't test climate against strong migration. Forest finished its updates in half the time on a faster machine, so crossing stopped early for part of each run, a flaw in the design. Tundra's seasons give a real cycle: grazers crash each winter, and hunters follow with a lag. Why each climate favours its genes is not measured.

**Stage 2: tests can pass for the wrong reason.** That discovery is why controls are part of every test after it.
