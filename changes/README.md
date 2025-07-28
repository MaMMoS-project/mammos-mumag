# News fragments

This directory contains news fragments for unreleased changes that will be
processed with [towncrier](https://towncrier.readthedocs.io/) to create a new
section in the the top-level [CHANGELOG file](../CHANGELOG.md).

## Creating a PR

Most PRs should be mentioned in the changelog. As part of the PR you need to
create a changelog fragment in the changes/ directory (where this README
resides). The file name must be `<PR number>.<type>.md`. CI will check that the
file exists and has an appropriate name. The content must be markdown. It should
NOT start with a bullet point and NOT reference the pull request as both will be
added by towncrier automatically.

Allowed types are:
- added
- changed
- deprecated
- removed
- fixed
- misc

misc can be used for all internal changes that are not directly relevant to
users, e.g. refactoring, work on the CI, other repository infrastructure, ...

If a PR should not appear in the changelog add the label
`no-changelog-entry-required` to the PR.

### Examples

Assume we have two PRs:

1. A new feature has been added in PR 1.

   Add a file with name `1.added.md` with content:

   ```markdown
   A new feature X has been added to support use case Y.
   ```

2. Changes to the CI have been made in PR 2.

   Add a file with name `2.misc.md` with content:

   ```markdown
   Refactoring of the CI to increase parallelization.
   ```

This will result in the following additions to the changelog (assuming we
release this as mammos-mumag version 0.1.0):

```markdown
## [mammos-mumag 0.1.0](<url to 0.1.0 tag>) – <release date>

### Added

- A new feature X has been added to support use case Y. ([#1](<url to PR 1>))

### Misc

- Refactoring of the CI to increase parallelization. ([#2](<url to PR 2>))
```

## Making a release

When releasing bump the package version and subsequently run `towncrier build`
in the root directory to convert all changelog fragments into a new section in
the toplevel CHANGELOG.md file. Commit changes in the changelog file and
automatically removed fragment files prior to tagging the release.

To preview the changes run `towncrier build --draft`.
