# Review: genes, memory and learning, and the updated plan

Date: 2026-11-01. Tom asked what is the most realistic way to model the relationship between genes, memory and learning; whether there's clear evidence of memories passing between generations through a genetic change in a living adult; whether we model genes that express at different times under environmental pressure; and whether we can add environmental mutation to population genomes. This records the answers, what they mean for this project, and the plan that follows. The evidence below is what I checked on the date above, and this literature is still moving.

## Short answers

1. **Can a memory pass genetically from an adult to its offspring?** There's no clear evidence that it can, for a specific memory. The evidence weakens the closer it gets to humans.
2. **What's most realistic?** A compressed genome that grows a brain, inherited learning rules, lifetime memory that dies with the animal, and culture to pass knowledge on. Of the three forms of inheritance in stage 14a, Baldwinian is closest to nature, and Lamarckian isn't how animals work.
3. **Do we model genes that express at different times?** No. Every gene in the simulation is fully expressed from birth to death, and none responds to age or season.
4. **Can we add environmental mutation?** Yes, and it's a small change. Today every child mutates by one fixed amount, wherever and however its parent lived. It becomes stage 15c.

## Do memories cross generations genetically?

| Species | Claim | Status | Refs |
|---|---|---|---|
| Worms (C. elegans) | Learned avoidance of a pathogen lasts several generations, carried by a small RNA from the bacterium | Disputed. One lab finds effects limited to one generation. Another independent group finds it through the F2 generation, fading, and says the assay is sensitive to conditions | 1, 2, 3 |
| Mice | Fear of a smell affects offspring's sensitivity to it | Contested. A statistical critique argues the results are too consistent to be believable. A later group reports related effects on smell neurons, and how they're inherited isn't settled | 4, 5, 6 |
| Humans | Famine and trauma effects across generations | No solid evidence. Reviews say culture, environment and genes can't be separated from it | 7, 8, 9 |

In mammals, DNA methylation is erased twice, once in germ cells and again after fertilization, which normally blocks inheritance of those marks (10). Some other vertebrates differ: zebrafish keep paternal methylation in their germline (11).

My framing, which I didn't check against a source in this pass: a memory lives mostly in the connections of neurons that never contribute to sperm or eggs. Even the positive results above look like a nudge in sensitivity, not a specific memory being passed on.

## The most realistic model

| Layer | In nature | In this project today |
|---|---|---|
| Genome | A compressed recipe for growing a brain, not its wiring. Networks compressed this way can start nearly as good as trained ones (12) | Darwinian and Baldwinian children copy all 4,702 weights directly, which is unrealistic |
| Learning rules | Inherited, and they shape how the brain changes. Evolved rules can teach random networks to walk (13) | A fixed rule, plus stage 14b's learning-rate gene |
| Lifetime memory | Changes in the brain's connections, which die with the animal | Stage 14's individual brains, with a weak rule |
| Passing it on | Mostly culture: watching, teaching, imitation | Missing |

Learning can still shape evolution without inheriting anything learned, which is the Baldwin effect (14), though how much it speeds evolution is disputed (15).

Tom's idea of descendants picking up what predecessors learned has a realistic form: culture. A child learns from its parents and doesn't receive their weights. Knowledge can still build up down a family line, with no biological barrier in the way.

**Stage 14a's harm has precedent.** In neural networks evolving under changing environments, Darwinian populations adapted best (16). A 2026 study on soft robots found Lamarckian inheritance loses only when changes are both conflicting and unpredictable, and that a sensor for the change restores its advantage (17). A 2024 robot study found Lamarckian doing better in changing environments (18). So this depends on conditions, and it isn't settled. Our animals have no season input, as I read `observe` in `simulations/evo.py`, which makes a sensor directly testable here.

## Genes expressed at different times

A gene often encodes a response more than a fixed trait. Continuous responses are called reaction norms, and discontinuous ones polyphenisms (19).

- **Predators:** Daphnia grow protective helmets only when they sense a predator's chemical signal (20).
- **Crowding:** locusts switch between solitary and swarming forms, with modules of gene activity that track the change (21).
- **Seasons:** some butterflies have wet-season and dry-season forms, and cues include temperature and day length (19, 20).

This could matter directly for results here. Tundra's hunters never reached camouflage alone (stage 13e), possibly because it costs energy all year. Tundra's grazers can't keep a big body through winter (stage 12h), and small grazers make small meals (stage 12i). All of that assumes a trait is on all the time, because the simulation can't do anything else. That reading is untested.

**Update, stage 15a:** tested for camouflage in tundra, and it didn't hold. A gene that works only in winter evolves no better than a constant gene at the same average cost, and it carries the world far worse. Price was the barrier, and timing wasn't. That's one trait in one world. A trait that does harm out of season might show the opposite.

## Environmental mutation

**Today:** a child's genes get Gaussian noise of a fixed size (`gene_sigma` 0.05) at birth, and an adult's genes never change. The environment plays no part.

**In nature, the claim is contested.** Starving E. coli have long been studied as raising mutation under stress, through known stress responses and error-prone repair (22, 23). Some reviews argue this speeds adaptation at a lower overall cost than a permanently high rate (24). A 2017 study found that allowing for deaths lowers, and sometimes removes, the signal in antibiotic experiments (25). I haven't checked sources on mutagens such as radiation in natural populations, or on mutational meltdown, so neither is claimed here.

**Three forms it can take here:**

- **By place:** zones on the map where children born there mutate more.
- **By season or climate:** more mutation in winter, or on one continent.
- **By stress:** a hungry parent's child mutates more, which is the bacterial idea. Stage 15c leaves this out, since animals here only breed when well fed, so a parent at birth is never hungry.

**The control that makes this a fair test.** More mutation of any kind might help, so each arm is compared with a control that has the same average mutation, spread evenly. Only then is the difference due to where and when mutation happens, and not just how much there is.

**Why it may matter here.** Stage 13e's tundra never reached camouflage alone, and stage 13a's split suggests hunters can get stuck low on that gene. Extra mutation under winter's stress might get them unstuck, or it might just damage what's already working. That's untested.

**Update, stage 15c:** the pattern didn't change where the gene ended up, while doubling the amount took tundra from never to about 30 updates. Worlds with mutation zones also lasted far better than even mutation, which stage 15d confirmed on fresh seeds. Stage 15f then found the same mix at random does just as well (0.977 each), so what helps is that mutation is uneven, not where it happens.

**Not in this stage:** mutating the genes of living adults. Today a gene change only reaches the next generation through a birth, and changing an adult's genes mid-life is a different question, which I'd test separately if this one shows anything.

## Updated plan

After stage 14c, in this order. Each gets its bars fixed in a spec before it runs, and each can fail.

| Stage | Question | What changes | First test |
|---|---|---|---|
| 15a | Can a conditional gene do what a constant one can't? | A camouflage gene expressed only in winter, costing energy only then | Can tundra evolve it alone, which it didn't in 13e? |
| 15b | Does knowing the season change what's inherited? | A season input for animals | Does it reduce Lamarckian harm in `tiny_winter` (14a), as (17) would suggest? |
| 15c | Does it matter where and when mutation happens? | Mutation rising by place, season or a parent's hunger, against controls with the same average | Can tundra reach camouflage alone, and does it beat even mutation? |
| 16 | Can knowledge build up through culture? | Juveniles learn from a parent during early life, and no weights are inherited | Do family lines improve beyond their founders, past Darwinian controls? |
| 17 | Can a compressed genome grow a brain? | A small genome decoded into weights, with evolved learning rules | Does evolving how to learn beat evolving weights? |

**Stage 17 is the rung I'd call recursive,** in a limited sense: the rule that does the improving would itself be what evolves. It would still be bounded by the simulation. Nothing before it is recursive self-improvement, and videos and docs won't call it that.

**What this changes in how the work runs.** Continents are for questions that need connection, such as gene flow, rescue by arrivals and traits spreading between them. Causes get tested in small worlds first. Stage 13d found climate shapes bodies far more clearly than brains, and stage 13e found connection delivering a trait a continent didn't reach alone.

**Stages already recorded here:** the individual-brain thread is in `2026-11-01-individual-brains-design.md`, `2026-11-01-heritable-learning-design.md` and `2026-11-01-long-inheritance-design.md`, and stage 14c is still running as this is written.

## Limits of this review

It's a focused search on one day, not a systematic review. Some sources were read as search summaries and abstracts, not in full, including 7, 8, 9, 14, 15, 16, 18, 22, 23, 24 and 25. I found no formal reply from the original worm lab to the failed replication in (1), and that doesn't mean none exists. Reference 20 is an encyclopedia entry, used for a textbook example.

## References

1. Gainey, Shubin, Hunter. Reported transgenerational responses to Pseudomonas aeruginosa in C. elegans are not robust. eLife. https://elifesciences.org/articles/100254
2. Independent validation of transgenerational inheritance of learned pathogen avoidance in C. elegans. eLife. https://elifesciences.org/articles/107034
3. The original worm claims are described in (1) and (2): Moore et al., and Kaletsky et al. 2020.
4. Emory University on Dias and Ressler 2014. https://news.emory.edu/stories/2014/01/mice-can-inherit-learned-sensitivity-smell
5. Discover on Francis 2014, "Too much success for recent groundbreaking epigenetic experiments". https://discovermagazine.com/mind/inherited-memories-too-good-to-be-true
6. Fear conditioning biases olfactory sensory neuron frequencies across generations. eLife, with reviews. https://elifesciences.org/articles/92882/peer-reviews
7. Review on transgenerational inheritance in humans, Russian Journal of Genetics. https://link.springer.com/article/10.1134/S1022795416030145
8. Bird. Transgenerational epigenetic inheritance: a critical perspective. Frontiers in Epigenetics and Epigenomics, 2024. https://doaj.org/article/3381b4c42fcc4871993d09eb33da9ece
9. Morgan and Whitelaw. Mammalian Genome, 2008. https://chd.ucsd.edu/_files/winter2009/Morgan.08.pdf
10. Zeng and Chen. DNA methylation reprogramming during mammalian development. Genes, 2019. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6523607/
11. Retention of paternal DNA methylome in the developing zebrafish germline. https://repositori.upf.edu/items/6341e28b-5376-431a-812f-103a0067580a/full
12. Shuvaev, Lachi, Koulakov, Zador. Encoding innate ability through a genomic bottleneck. PNAS, 2024. https://pmc.ncbi.nlm.nih.gov/articles/PMC11420173/
13. Najarro and Risi. Meta-learning through Hebbian plasticity in random networks. NeurIPS, 2020. https://arxiv.org/abs/2007.02686v5
14. The revival of the Baldwin effect, on Hinton and Nowlan 1987. https://arxiv.org/pdf/1702.08411
15. Phenotypic plasticity, the Baldwin effect, and the speeding up of evolution: the computational roots of an illusion. https://arxiv.org/pdf/1411.6843
16. Sasaki and Tokoro. On adaptation in changing environments, Darwinian and Lamarckian populations. https://www.sonycsl.co.jp/publications/id21648/
17. de Bruin, Glette, Ellefsen. Lamarckian inheritance in dynamic environments: how key variables affect evolutionary dynamics. 2026. https://arxiv.org/abs/2605.15769
18. Lamarckian inheritance improves robot evolution in dynamic environments. 2024. https://ar5iv.labs.arxiv.org/html/2403.19545
19. Nijhout. Development and evolution of adaptive polyphenisms. 2003. https://nijhoutlab.biology.duke.edu/files/2020/08/Polyphenism2003.pdf
20. Polyphenism. Wikipedia. https://en.wikipedia.org/wiki/Polyphenism
21. Comparative analysis of phenotypic plasticity sheds light on the evolution and molecular underpinnings of locust phase polyphenism. Scientific Reports, 2021. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8184943/
22. In times of stress, mutate early and often. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC529323/
23. Foster's review of stress-induced mutagenesis. https://pmc.ncbi.nlm.nih.gov/articles/PMC2747772/
24. A review arguing stress-induced mutagenesis is adaptive. https://pmc.ncbi.nlm.nih.gov/articles/PMC6028306
25. The 2017 study accounting for deaths in antibiotic-induced mutagenesis. https://link.springer.com/10.1007/s00294-017-0801-9
