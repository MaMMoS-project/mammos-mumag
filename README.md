# Info
`mmag` is a finite-element micromagnetic simulation tool capable of simulating hysteresis loops of magnetic materials with multiple grains, developed and maintained by Thomas Schrefl at Zentrum für Modellierung und Simulation, Universität für Weiterbildung Krems.

This package allows users to use `mmag` in Python and includes some useful scripts for the use and development of `mmag`. On way to install the package would be to execute `pip install .`, but we recommend using [pixi](https://prefix.dev).


# Usage with Pixi (recommended)
Run `pixi shell` to activate a container where `mmag` is installed.
This package comes with several pixi tasks (in alphabetical order):
- `clean`
- `docs`
- `docs-clean`
- `format`
- `lint`
- `test`

To run a task, execute `pixi run <task_name>`.


## Style tasks
These tasks (`clean`, `format`, and `lint`) use [Ruff](https://docs.astral.sh/ruff/) to lint and format the code with the rules specified in [`pyproject.toml`](pyproject.toml)


## Test tasks
The task (`test`) executes tests found in [`test`](test/).


## Docs tasks
The tasks (`docs`, `docs-clean`) manage the documentation. In particular, `docs` builds the html docs, while `docs-clean` cleans the current build.
