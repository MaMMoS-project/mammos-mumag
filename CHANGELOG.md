# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This project uses [towncrier](https://towncrier.readthedocs.io/) and the changes for the upcoming release can be found in [changes](changes).

<!-- towncrier release notes start -->

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
