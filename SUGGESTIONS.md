# Suggestions

This project is built over time with the people who watch it. Each idea below shows who suggested
it, where it stands and what happened. Add yours with the
[Suggest an experiment](https://github.com/alltomstales-dotcom/how-i-grew-ai-brains/issues/new?template=suggest-an-experiment.yml)
form, or in a comment on [any of the videos](https://www.youtube.com/@NeatScienceStuff).

| Idea | Suggested by | Status | Result |
|---|---|---|---|
| Hunters that hide, so they spend less energy chasing | [@rhetkuehne8613](https://www.youtube.com/@rhetkuehne8613), comment on Part 1 | **Tested:** [Stage 8b](docs/specs/2026-10-30-ambush-design.md) | **PASS on two seeds.** Hunters spend 64-65% of their time in thickets (25% of the map), and catch 68-72% less when thickets stop hiding them. But hiding only matches chasing per meal: a one-line chaser catches 1.4-1.7x as much and burns about two thirds more energy per step. |
| A meal's value depends on what the prey stored | A viewer | **Queued** | Not run yet |
| Omnivores | Tom, in a reply to a viewer's comment | **Idea,** not started | Not run yet |

## What the statuses mean

- **Idea:** noted, with no decision yet.
- **Queued:** it will be tried, after a spec and a pass bar are written.
- **Building** or **Running:** the spec is public in [docs/specs](docs/specs), so the bar is on record before the result.
- **Tested:** it has a result, and the result here says PASS or FAIL. Most tests fail, and a failure still counts as a result.
- **Not taken:** with a sentence saying why, such as the world can't yet express it.

There is one GPU and one trainer at a time, so not every idea will run, and there's no promised date.
See [CONTRIBUTING.md](CONTRIBUTING.md) for what makes an idea easy to test.
