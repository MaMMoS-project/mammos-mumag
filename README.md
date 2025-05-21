# Info
`mammos-mumag` is a finite-element micromagnetic simulation tool capable of simulating hysteresis loops of magnetic materials with multiple grains, developed and maintained by Thomas Schrefl at Zentrum für Modellierung und Simulation, Universität für Weiterbildung Krems.

This package allows users to use the library `mammos_mumag` in Python and includes some useful scripts for the use and development of `mammos-mumag`. On way to install the package would be to execute `pip install .`, but we recommend using [pixi](https://prefix.dev).


# Installation

Clone the repository and install with pip, i.e. run
```console
git clone https://github.com/MaMMoS-project/mammos-mumag
pip install mammos-mumag
```

<!-- # Installation with conda-pip (discouraged)
The package `esys-escript` must be installed from `conda-forge` (see [here](https://github.com/LutzGross/esys-escript.github.io/)) with
```console
 conda install esys-escript -c conda-forge
 ```

`cuda` must be installed from the `nvidia` channeel with
```console
conda install cuda -c nvidia
```

Then, in the same environment where the two previous packages have been installed, we can install `mammos_mumag` with pip by running
```console
pip install .
```

> To install optional dependencies, run e.g. `pip install .[test]` or `pip install .'[test]'` (for example on zsh).


# Installation & usage with Pixi (recommended)
Run `pixi shell` in any subdirectory to activate a container where this package is installed.
This package comes with several pixi tasks (in alphabetical order):
- `clean`
- `docs`
- `docs-clean`
- `format`
- `lint`
- `pre-commit`
- `test`

To run a task, execute `pixi run <task_name>` or `pixi r <task_name>`.


## Style tasks
These tasks (`clean`, `format`, and `lint`) use [Ruff](https://docs.astral.sh/ruff/) to lint and format the code with the rules specified in [`pyproject.toml`](pyproject.toml)


## Test tasks
The task (`test`) executes tests found in the [`test`](test/) directory.


## Docs tasks
The tasks (`docs`, `docs-clean`) manage the documentation. In particular, `docs` builds the html docs, while `docs-clean` cleans the current build.


## Pre-commit task
The task to execute pre-commit can be run by the user to check that committed changes adhere to the formatting and linting rules.
If the pre-commit hook is installed, the command `pre-commit` is also executed automatically every time `git commit` is called, but one needs to activate the right environment first.
This is done with
```console
pixi shell -e pre-commit
```

> *pre-commit* has to be installed after the first activation of the environment. To do this, run
> ```console
> pixi run pre-commit install
> ``` -->


## Working examples
Please refer to the examples:
- [Materials i/o](docs/source/notebooks/materials_io.ipynb)
- [Parameters i/o](docs/source/notebooks/parameters_io.ipynb)
- [Using the pre-defined scripts](docs/source/notebooks/scripts.ipynb)
- [Run a hysteresis loop with pre-defined meshes](docs/source/notebooks/hysteresis_loop.ipynb)
- [Converting `unv` mesh to `fly`](docs/source/notebooks/unvtofly.ipynb)
