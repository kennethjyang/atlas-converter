# Atlas Converter

[![Test](https://github.com/kennethjyang/atlas-converter/actions/workflows/test.yml/badge.svg)](https://github.com/kennethjyang/atlas-converter/actions/workflows/test.yml)
[![Code Quality](https://github.com/kennethjyang/atlas-converter/actions/workflows/code-quality.yml/badge.svg)](https://github.com/kennethjyang/atlas-converter/actions/workflows/code-quality.yml)

Tool to convert BrainGlobe-formatted atlases into Pinpoint V-compatible atlases.

> [!WARNING]
> Under active and early development. Feel free to poke around and contribute, but this is not a finished product yet.

## Install for development

1. Install [UV](https://docs.astral.sh/uv/getting-started/installation/).
2. Run `uv sync` to install the environment.
3. Run `uv run lefthook install` to set up hooks.

## Usage

It's a CLI tool. Run `uv run main.py --help` to find commands and options.

The most common use would be `uv run main.py brainglobe` which will download the entire BrainGlobe atlas library (512 GB) and convert it into a Pinpoint Atlas library (10 GB) at `~/pinpoint_atlases`.

## Serving custom atlases

Atlases are built to `~/pinpoint_atlases`. The easiest way to make them available is to run:

```bash
npx serve -C ~/pinpoint_atlases/
```

Many other tools can also create a simple local HTTP server with CORS configured to serve these files.
