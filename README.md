# Atlas Converter
Tool to convert BrainGlobe formatted atlases into Pinpoint V compatible atlases.

> [!WARNING]
> Under active and early development. Feel free to poke around and contribute, but this is not a finished product yet.

## Install for development

1. Install [UV](https://docs.astral.sh/uv/getting-started/installation/).
2. Run `uv sync` to install the environment.
3. Run `uv run lefthook install` to set up hooks.

## Serving custom atlases

Atlases are built to `~/pinpoint_atlases`. The easiest way to make them available is running:

```bash
npx servitsy --cors ~/pinpoint_atlases/
```

There are many other tools that can also create a simple local HTTP server with CORS configured to serve these files.
