# Results, stage by stage

Each stage adds one pressure to the world. Its pass bar is fixed in a spec before any run, and it stays
where it was even after a near miss. The spec holds the full numbers and the caveats.

**The score so far: 6 stages or sub-stages pass and 8 fail.** Counting each new version of a failed world as
its own attempt, there are 15 failures. Some passes are weaker than others, as the notes say.

| Stage | The question | The bar, set first | Result | Spec |
|---|---|---|---|---|
| 1 | Can two species learn from nothing? | Predators catch 2x as often as random ones; prey live 1.5x as long | **PASS:** 3.36x and 1.61x. Then a stalemate: prey dodge nearly every attack and hunters starve | [spec](specs/2026-10-14-predprey-crucible-design.md) |
| 2 | Do bodies and plants evolve? | Evolved lineages out-compete founders, and a plant gene shifts | **PASS, weak alone:** a first version passed with untrained brains, and pp-evo-002's random-brain control passes too, so the finding is the comparison with it. A later run (pp-evo-003) failed this test | [spec](specs/2026-10-14-evolution-design.md) |
| 3a | Do prey learn to use burrows? | Trained prey hide near predators 1.5x as selectively as random prey | **FAIL:** ratio 1.05 against 1.20 for random prey | [spec](specs/2026-10-14-terrain-design.md) |
| 3b | Does training against past opponents stop the hunters stalling? | Predator Elo rises by 100 | **FAIL:** -11 to +77. Hunters stall against prey that never learn too, and a hand-written chase does 0.83-1.06x as well in the evolution world. It's a ceiling on the world, not a learning failure | [roadmap](roadmap.md#stage-3-portable-brains-and-a-league-in-progress) |
| 4 | Does memory help in a seasonal world? | 1.2x on prey lifetime or catches, at matched world-steps | **FAIL:** 0.99x and 1.13x. Memory hunters learn faster per step, but feed-forward catches up | [spec](specs/2026-10-14-memory-design.md) |
| 5a | Can one founder pool split into two species? | Separation D of at least 2, minority of at least 0.2, gap of at least 0.3, in the last 3 snapshots | **FAIL, six versions:** the last was a near miss (D 1.95 on one snapshot). Without mating two size groups hold apart; two-parent breeding prevents it | [spec](specs/2026-10-14-speciation-design.md) |
| 6a | If a catch takes two hunters, do they learn teamwork? | 1.5x a lone chaser | **FAIL:** 1.02-1.04x. Chasers meet at the nearest prey anyway | [spec](specs/2026-10-14-pack-hunting-design.md) |
| 6b | Do animals learn to signal? | Catches drop 10% with signals blanked | **FAIL:** under 1%. A later review found signals could barely reach beyond sight, which weakens this result | [spec](specs/2026-10-14-signals-design.md) |
| 6c | If hunters must surround prey? | 1.5x a lone chaser, on two seeds | **PASS:** 3.0-3.8x in all eight readings. The first stage where learning clearly beats a simple rule. Signals help evolved hunters; in founder bodies only on one seed | [spec](specs/2026-10-14-flanking-design.md) |
| 7a | Does evolution buy brain where thinking pays? | Hunter brain gene 0.1 higher in flank than in pack, on two seeds | **FAIL, three tries:** at a flat cost brain size follows food, not the value of thinking (0.19 flank, 0.50 pack) | [spec](specs/2026-10-14-brain-size-design.md) |
| 7b | Same question, with the brain taxed as a share of each meal? | Gap of +0.12 | **PASS:** 0.53 against 0.18 and 0.69 against 0.38 on seed 0; 0.47 against 0.14 on seed 1 | [spec](specs/2026-10-29-brain-tax-design.md) |
| 7c | Can an animal learn within one life, when food turns to poison at random? | Memory beats no memory by 0.10 at 10-40 steps after a flip, and beats itself at 0-10 by 0.05, on two seeds | **PASS:** +0.218 and +0.122 (seed 1 only narrowly); recovery +0.398 and +0.272. See the limits below | [spec](specs/2026-10-30-lifetime-design.md) |
| 8a | Do hills matter? | 10% worse with the height channel zeroed | **FAIL:** predators 4.8% and -5.4%, prey 2%. Hunters do take the high ground (78% against 64% at random) in plain hills | [spec](specs/2026-10-29-hills-design.md) |
| 8b | Do hunters learn to ambush, if walking costs energy and thickets hide them? (a viewer's idea) | 1.3x the map's thicket share, 1.5x random's stillness, 10% fewer catches when thickets stop hiding them, on two seeds | **PASS:** 64-65% in thickets (map 25%), 90-93% still (random 20%), 68-72% fewer catches. See the limits below | [spec](specs/2026-10-30-ambush-design.md) |

## Limits worth repeating

These are the caveats most easily lost when a result gets shortened.

**Ambush (8b): hiding only matches chasing per meal.** A one-line chaser catches 1.4-1.7x as much as the trained
hunter and burns about two thirds more energy per step, so per meal the two are about even. Learning found a second
way to make the same living, not a better one.

**Lifetime learning (7c): memory is mostly speed.** Brains with no memory also recover after a flip, later. By 40+
steps they reach 0.75, against 0.83 and 0.74 with memory. Why they recover at all was not measured. Seed 1 is weaker,
and shows no gain in prey lifetime (35.7 against 36.0).

**Lifetime learning (7c): this is memory inside a trained network, not changing weights.** A brain's weights are fixed
during an animal's life. What changes is the memory state, and the ability to taste and remember was learned by the
species across training. Plastic weights remain for later. The controls here were added after the result: wiping the
memory every step takes the same brain back down to the no-memory level (0.66 and 0.60 at 10-40 steps), and after a
bad bite those brains are at about a coin flip.

**Flanking (6c): "smart" means smart in this rule.** Two hunters must stand exactly opposite the prey. It shows that
coordination is learned when a simple rule fails, not that hunters are generally clever. Hills cut the edge to 2.3x.

**Brain tax (7b): a fair cost changes the answer.** The flat-cost result reversed under a tax. The result rests on the
idea that a share of income is the fair comparison, and that was chosen by us.

**Stage 2: tests can pass for the wrong reason.** That discovery is why controls are part of every test after it.
