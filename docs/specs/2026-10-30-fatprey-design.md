# Stage 5b: meals worth what the prey stored (a viewer's suggestion)

Date: 2026-10-30. Written before any run.

## Why

A viewer wrote on the first Short: "The primary advantage of a predator is the energy density of the meal. Grazers do all the work to collect and store the calories." In the world, a catch was a fixed 25 energy, so a fat grazer and a starving one were worth the same.

## World

`fatprey` (EvoConfig plus): a catch gives the hunter `meal_share` (1.25) times the energy the grazer held. A new channel, `fat_sense`, shows how much energy the grazers in each visible cell hold, after the thicket channel and before the brain gene. `fatprey_fixed` is the control: the same channel, but every meal is still 25.

## Success test (can fail)

`pp-fat-{0,1}` and `pp-fatfix-{0,1}`, seeds 0 and 1, 150 updates each. `python -m scripts.meals --run-dir runs/NAME --control runs/CONTROL --evolved`, 256 fixed seeds. Selectivity is the energy held by the grazers eaten, over what they would hold at the living population's mean. Both must hold on both seeds:

1. **They hunt the fat:** selectivity is at least 1.15x the control's.
2. **Sight is why:** with the fat channel zeroed for hunters, energy per meal falls at least 10%.

Reported, as predictions and not bars: do grazers stay leaner than in the control (mean energy, breeding gene), and catches, hunter numbers and `chase.py --evolved`.

## Result (2026-10-30): FAIL on both seeds

| | seed 0 | seed 1 | bar |
|---|---|---|---|
| selectivity, stored meals | 1.016 | 0.942 | |
| selectivity, control | 0.925 | 0.969 | |
| ratio to control | 1.10x | 0.97x | at least 1.15x |
| meal value drop with the fat channel zeroed | 3.4% | -0.2% | at least 10% |

Hunters did not hunt the fat. Selectivity sits near 1 in every condition, and blinding them to fat barely changes what a meal is worth, so nothing here shows they use the channel. The seed 0 ratio of 1.10x comes as much from a low control as from a high treatment, and seed 1 goes the other way.

The prediction about grazers did come out as guessed on both seeds: mean energy 8.9 and 9.0 against 9.5 and 9.3, breeding gene 0.158 and 0.152 against 0.214 and 0.208. It can't be read as hunters thinning out the fat, because hunters aren't selecting. There is also a confound in this design: a catch held about 9 energy, so at 1.25x a meal here pays about 11 where the control's fixed meal pays 25. The two worlds differ in how well hunting pays, as well as in whether fat matters.

Against trained prey the fixed chaser managed 0.69x and 0.68x the learned hunters' catches per predator-step, so the learned hunters are still real hunters.

Inference, not measured: grazers' stored energy may vary too little, or catches be too rare and too close-range, for a choice between fat and lean to arise. Testing that would need the spread of energy among grazers within sight at the moment of a catch.

Also learned: all four trainers started first try at `--mem-gb 16`, which supports the pool-fragmentation reading of the 12 GiB crashes.

## v2: a fair retest (written 2026-10-30, before any run)

v1 was an unfair test of the viewer's idea, because my design assumed prey held about 20 energy. The prey eaten held about 9.5, so a meal paid about 12 where the control's paid 25. World `fatprey_rich` sets `meal_share` to 2.6, which brings the average meal back to about 25 at v1's prey energies.

`pp-fatr-{0,1}`, seeds 0 and 1, same settings, against the same controls `pp-fatfix-{0,1}`. The bars don't move: selectivity at least 1.15x the control's, and meal value falling at least 10% with the fat channel zeroed, both on both seeds.

Reported, not bars: the spread of energy among living grazers (`prey_energy_sd`, new in `scripts.meals`), which tests v1's guess that there may be too little difference to choose between; grazer mean energy and breeding gene; catches; `chase.py --evolved`.

If v2 fails too, hunters not choosing fat is the finding, and the idea is closed at this map size and sight.

## v2 result (2026-10-30): FAIL on both seeds, idea closed

| | seed 0 | seed 1 | bar |
|---|---|---|---|
| selectivity, meals at about 25 | 0.988 | 0.973 | |
| selectivity, control | 0.925 | 0.969 | |
| ratio to control | 1.07x | 1.00x | at least 1.15x |
| meal value drop with the fat channel zeroed | 2.9% | 1.9% | at least 10% |

With hunting paying as well as in the control, hunters still did not choose fat prey. Selectivity stays just under 1 in every condition, so the grazers eaten are, if anything, slightly leaner than average.

v1's guess that there was too little difference to choose between is wrong for the population as a whole: living grazers hold 9.0 and 9.3 energy with a standard deviation of 5.1 and 5.3. Fat and lean grazers exist in large numbers. This measures all grazers across a world, not the ones within one hunter's sight at one moment, so it doesn't rule out that hunters rarely see two at once. Why they don't choose is not measured.

v1's leaner grazers mostly came from the confound. With meals back at 25, the breeding gene moved from 0.158 and 0.152 to 0.184 and 0.199, against the control's 0.214 and 0.208, and mean energy from 8.9 and 9.0 to 9.0 and 9.3, against 9.5 and 9.3. On seed 1 the energy difference is gone.

Hunters catch once per roughly 140 predator-steps (0.0075 and 0.0074), about the control's rate (0.0073 and 0.0067). The fixed chaser manages 0.73x and 0.66x of the learned hunters.

The viewer's premise may still be true of real predators. This world doesn't show it: at this map size and sight, a meal's value doesn't change who gets eaten. Closed, as the v2 plan said it would be.

