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
