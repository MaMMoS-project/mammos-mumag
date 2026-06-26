# Agent Instructions for mammos-mumag

This file is only for AI coding agents. General package guidance belongs in
`CONTRIBUTING.md`; shared MaMMoS standards belong in
`CONTRIBUTING-MaMMoS.md`.

Read `CONTRIBUTING.md`, `CONTRIBUTING-MaMMoS.md`, and `README.md` before editing.

`CONTRIBUTING-MaMMoS.md` is a read-only package copy. Make edits to shared
MaMMoS standards in the `mammos-devtools` repository instead.

This repository must work as a standalone checkout. Do not assume
`mammos-devtools` or sibling repositories are present. Use this repository's
`pyproject.toml`, pixi tasks, tests, examples, and docs as the local source of
truth unless explicitly instructed otherwise.

If this checkout is located at `mammos-devtools/packages/mammos-mumag`, also
read `../../AGENTS.md` for umbrella-repository guidance.

Keep generated code and documentation simple, explicit, and easy for a human
maintainer to understand.
