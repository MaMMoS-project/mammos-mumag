# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This project uses [towncrier](https://towncrier.readthedocs.io/) and the changes for the upcoming release can be found in [changes](changes).

<!-- towncrier release notes start -->

## [mammos-mumag 0.10.1](https://github.com/MaMMoS-project/mammos-mumag/tree/0.10.1) – 2025-12-03

### Misc

- Fix dependencies: add `matplotlib`, `pandas`, and `urllib3`. ([#93](https://github.com/MaMMoS-project/mammos-mumag/pull/93))


## [mammos-mumag 0.10.0](https://github.com/MaMMoS-project/mammos-mumag/tree/0.10.0) – 2025-11-27

### Added

- Added `tesla=True` option in the `plot` method of `mammos_mumag.hysteresis.Result` to generate the hysteresis loop in Tesla units. ([#87](https://github.com/MaMMoS-project/mammos-mumag/pull/87))


## [mammos-mumag 0.9.1](https://github.com/MaMMoS-project/mammos-mumag/tree/0.9.1) – 2025-11-04

### Added

- Added `multigrain-simulation` examples to documentation ([#85](https://github.com/MaMMoS-project/mammos-mumag/pull/85))


## [mammos-mumag 0.9.0](https://github.com/MaMMoS-project/mammos-mumag/tree/0.9.0) – 2025-11-03

### Added

- Create cli command `unv2fly` to convert unv mesh to fly format. ([#61](https://github.com/MaMMoS-project/mammos-mumag/pull/61))
- Added notebook `using_tesla.ipynb` for information on how to set up a workflow in Tesla. ([#68](https://github.com/MaMMoS-project/mammos-mumag/pull/68))
- Added possibility to install GPU support (both CUDA and ROCm) with `pip` via the extra dependencies. ([#81](https://github.com/MaMMoS-project/mammos-mumag/pull/81))

### Changed

- Now :py:func:`mammos_mumag.hysteresis.run` can be used to execute simulations with multigrain materials. ([#46](https://github.com/MaMMoS-project/mammos-mumag/pull/46))
- Implement automatic retries to download meshes if the requests fail. The requests will try three times in total, with a backoff factor of 0.1. ([#70](https://github.com/MaMMoS-project/mammos-mumag/pull/70))
- Documentation is updated. Parameters have been formatted to snake case when possible. The names `h_start`, `h_final`, `h_step`,  `n_h_steps`, `m_step`, `m_final`, and `tol_h_mag_factor` take the place of `hstart`, `hfinal`, `hstep`, `nhsteps`, `mstep`, `mfinal`, and `tol_hmag_factor`. Whenever possible, reasonable entities have been defined. The unused variables `iter_max`, `tol_u`, and `verbose` have been removed. Warning: this PR causes failure in previously defined workflows if the variables  were defined by the user. ([#71](https://github.com/MaMMoS-project/mammos-mumag/pull/71))

### Fixed

- Fixed default `outdir` input in two functions in `mammos_mumag.simulation`. ([#69](https://github.com/MaMMoS-project/mammos-mumag/pull/69))

### Misc

- Added `examples/hysteresis.ipynb` to document full functionality of `mammos-mumag` when running a hysteresis loop simulation. Additionally, show the functionality of the package irrelevant to an average user in `examples/additional-functionality.ipynb`. ([#42](https://github.com/MaMMoS-project/mammos-mumag/pull/42))


## [mammos-mumag 0.8.1](https://github.com/MaMMoS-project/mammos-mumag/tree/0.8.1) – 2025-08-13

### Fixed

- Fixed a small bug that occurred when the inputs to `hysteresis.run` were zero. ([#64](https://github.com/MaMMoS-project/mammos-mumag/pull/64))


## [mammos-mumag 0.8.0](https://github.com/MaMMoS-project/mammos-mumag/tree/0.8.0) – 2025-08-12

### Added

- Add function `mammos_mumag.hysteresis.read_result` to read the result of a hysteresis loop from a folder (without running the hysteresis calculation again). ([#48](https://github.com/MaMMoS-project/mammos-mumag/pull/48))
- Implement `mammos_mumag.mesh.Mesh` class that can read and display information of local meshes, meshes on Zenodo and meshes given by the user. ([#53](https://github.com/MaMMoS-project/mammos-mumag/pull/53))

### Changed

- Changed the output of the hysteresis loop in compliance with `mammos_entity.io` v2. ([#54](https://github.com/MaMMoS-project/mammos-mumag/pull/54))

### Misc

- Use [towncrier](https://towncrier.readthedocs.io) to generate changelog from fragments. Each new PR must include a changelog fragment. ([#49](https://github.com/MaMMoS-project/mammos-mumag/pull/49))
