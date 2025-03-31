# Info
`mammos-mmag` is a finite-element micromagnetic simulation tool capable of simulating hysteresis loops of magnetic materials with multiple grains, developed and maintained by Thomas Schrefl at Zentrum für Modellierung und Simulation, Universität für Weiterbildung Krems.

This package allows users to use the library `mammos_mmag` in Python and includes some useful scripts for the use and development of `mammos-mmag`. On way to install the package would be to execute `pip install .`, but we recommend using [pixi](https://prefix.dev).


# Installation with conda-pip (discouraged)
The package `esys-escript` must be installed from `conda-forge` (see [here](https://github.com/LutzGross/esys-escript.github.io/)) with
```console
 conda install esys-escript -c conda-forge
 ```

`cuda` must be installed from the `nvidia` channeel with
```console
conda install cuda -c nvidia
```

Then, in the same environment where the two previous packages have been installed, we can install `mammos_mmag` with pip by running
```console
pip install .
```

> To install optional dependencies, run e.g. `pip install .[test]` or `pip install .'[test]'` (for example on zsh).


# Installation & usage with Pixi (recommended)
Run `pixi shell` to activate a container where `mmag` is installed.
This package comes with several pixi tasks (in alphabetical order):
- `clean`
- `docs`
- `docs-clean`
- `format`
- `lint`
- `test`

To run a task, execute `pixi run <task_name>` or `pixi r <task_name>`.


## Style tasks
These tasks (`clean`, `format`, and `lint`) use [Ruff](https://docs.astral.sh/ruff/) to lint and format the code with the rules specified in [`pyproject.toml`](pyproject.toml)


## Test tasks
The task (`test`) executes tests found in the [`test`](test/) directory.


## Docs tasks
The tasks (`docs`, `docs-clean`) manage the documentation. In particular, `docs` builds the html docs, while `docs-clean` cleans the current build.


## Working examples
Please refer to the examples:
- [Materials i/o](docs/source/notebooks/materials_io.ipynb)
- [Parameters i/o](docs/source/notebooks/parameters_io.ipynb)
- [Using the pre-defined scripts](docs/source/notebooks/scripts.ipynb)
- [Converting `unv` mesh to `fly`](docs/source/notebooks/unvtofly.ipynb)
