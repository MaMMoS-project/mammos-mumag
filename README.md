# Info
`mmag` is a finite-element micromagnetic simulation tool capable of simulating hysteresis loops of magnetic materials with multiple grains, developed and maintained by Thomas Schrefl at Zentrum für Modellierung und Simulation, Universität für Weiterbildung Krems.

This package allows users to use `mmag` in Python and includes some useful scripts for the use and development of `mmag`. On way to install the package would be to execute `pip install .`, but we recommend using [pixi](https://prefix.dev).


# Usage with Pixi (recommended)
Run `pixi shell` to activate a container where `mmag` is installed.
This package comes with several pixi tasks (in alphabetical order):
- `build`
- `clean`
- `docs`
- `docs-clean`
- `format`
- `lint`
- `test`

To run a task, execute `pixi run <task_name>`.


## Building task
The `build` task helps building a container for `escript`.
The default configuration of
```terminal
pixi run build
```
builds `escript` inside and [Apptainer](https://apptainer.org/) container using 16 threads and storing the container in `~/.cache/mammos/mmag/mmag.sif`.
These options are stored as environment variables, and can be changes for example as follows:
```terminal
CONTAINER=podman THREADS=4 pixi run build
```


## Style tasks
These tasks (`clean`, `format`, and `lint`) use [Ruff](https://docs.astral.sh/ruff/) to lint and format the code with the rules specified in [`pyproject.toml`](pyproject.toml)


## Test tasks
The task (`test`) executes tests found in [`test`](test/).


## Docs tasks
The tasks (`docs`, `docs-clean`) manage the documentation. In particular, `docs` builds the html docs, while `docs-clean` cleans the current build.
