# Contributing

The most useful thing you can bring is an idea for what to try next. You don't need to write code.

## Suggest an experiment

Open a [Suggest an experiment](https://github.com/alltomstales-dotcom/how-i-grew-ai-brains/issues/new?template=suggest-an-experiment.yml)
issue, or leave a comment on [a video](https://www.youtube.com/@NeatScienceStuff). Ideas are tracked in
[SUGGESTIONS.md](SUGGESTIONS.md) with who suggested them, their status and their result.

The ideas easiest to test tend to share three things:

- **One new pressure.** Something the world makes the animals face: a cost, a danger, a food that changes.
  "Moving costs energy" is easier to test than "make them smarter".
- **Something the world can't reward today.** The hiding idea led to Stage 8b because moving cost the same as
  standing still, so hiding could never pay however smart the hunter was.
- **A way to be wrong.** What number would show that the idea did not work?

## Credit

If your idea is used, it gets credited by the handle you give. If you don't give one, or it comes from a comment
and we can't ask, it's listed as "a viewer". To change or remove a credit, open an issue.

## How an idea becomes a stage

1. A spec goes in [docs/specs](docs/specs): the world's rules, the test and the pass bar, **written before any run**.
2. Training runs, on at least two seeds if the first passes.
3. The result is written into the spec, the roadmap and [SUGGESTIONS.md](SUGGESTIONS.md), pass or fail. The bar does
   not move afterwards. If a control shows that a pass happened for the wrong reason, that gets written down too.

## Code

Pull requests are welcome, for bug fixes and especially for mistakes in a measurement. A bug that changes a result
is one of the best things you can find here: the Stage 4 memory test and Stage 6b signal test were both changed or
weakened by review findings like that.

```bash
export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/autolearn-public
uv sync
uv run pytest -q
```

Tests run on CPU and need no GPU. Please keep changes to one world from altering any earlier one: new behaviour goes
behind a config flag that's off by default, so old runs still mean what they said.

Be kind in issues and comments. The animals can't read them, but people can.
