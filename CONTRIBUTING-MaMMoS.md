<!--
Canonical source: MaMMoS-project/mammos-devtools, CONTRIBUTING-MaMMoS.md.
Package copies are generated with `pixi run sync-contributing` from
the mammos-devtools repository. Do not edit package copies directly.
-->

# Shared MaMMoS Contribution Guide

This guide contains contribution standards that apply across MaMMoS package
repositories. Repository-specific setup and checks belong
in each repository's `CONTRIBUTING.md`.

## Code and repository conventions

- Use American spelling everywhere.
- Use the `src` layout for Python packages.
- Keep tests outside the deployed package.
- Follow Ruff for style decisions.
- Use semantic versioning for all packages.
- Update the metapackage version according to the largest version change of any
  dependent package or of the metapackage itself.

## Naming

- Capitalization of physical quantities and the objects or functions that create
  them follows the natural domain capitalization, even where this differs from
  normal Python naming conventions.
- Other code should follow PEP 8 naming conventions: `snake_case` for functions,
  methods, and properties; `CamelCase` for classes.
- Prefer clear names over short names. For example, prefer `energy_density` over
  `w`.

## Type hints

- All code must have complete type hints.
- Type hints should use the full MaMMoS package names instead of abbreviations.
- Avoid `from` imports used only for type hints.
- Use `from __future__ import annotations` where useful.
- Use `if TYPE_CHECKING:` guards for imports needed only by type checkers.

## Return values and inputs

- When a function or method returns one MaMMoS value the return object should
  preferably be an Entity.
- When a function or method returns more than one MaMMoS value, prefer a custom
  composite object based on `mammos_entity.EntityCollection`.
- Attributes of such composite objects should be `mammos_entity.Entity` objects
  where possible. If that is not possible, prefer `astropy.units.Quantity`, then
  ordinary Python values.
- Where relevant, accept and validate inputs in this order:
  `mammos_entity.Entity`, compatible quantity objects, then raw Python values.
- When accepting raw numbers or arrays, document and apply the entity base unit.

## Documentation and examples

- Public APIs must have docstrings, where practical with examples.
- Private APIs should also have docstrings in most cases. Exceptions can be made for small utility functions when their behavior is fully clear from the function name.
- Examples should use explicit units and should show the expected entity labels.
- Keep user-facing documentation in README files, package
  docs, or examples.
- Agent-only operational advice belongs in `AGENTS.md`.
- When new information is needed, put it in the document for its audience. Use
  README files, examples, or normal docs for user-facing information. Use this
  package's `CONTRIBUTING.md` for mammos-entity developer guidance and
  `CONTRIBUTING-MaMMoS.md` for shared MaMMoS developer guidance. Use `AGENTS.md`
  only for AI-specific operating instructions.

### Validating inputs to a function

```python
from __future__ import annotations

from typing import TYPE_CHECKING

import mammos_entity as me

if TYPE_CHECKING:
    # Import with full names to keep annotations explicit.
    import mammos_entity
    import mammos_units
    import numpy.typing


def compute_speed(
    length: mammos_entity.Entity | mammos_units.Quantity | numpy.typing.ArrayLike,
    time: mammos_entity.Entity | mammos_units.Quantity | numpy.typing.ArrayLike,
) -> mammos_entity.Entity:
    """Compute travelling speed.

    Args:
        length: The travelled distance as :entity:`Length`.
            If no unit is provided, values are interpreted as `m`.
        time: The travel :entity:`Time`.
            If no unit is provided, values are interpreted as `s`.

    Returns:
        The travelling speed as :entity:`Speed`.

    Examples:
        Passing entities:

        >>> import mammos_entity as me
        >>> length = me.Entity("Length", 10, "mm")
        >>> time = me.Entity("Time", 2, "s")
        >>> compute_speed(length, time)
        Entity(ontology_label='Speed', value=5., unit='mm / s')

        Passing an array of raw numbers:

        >>> compute_speed(10, 2)
        Entity(ontology_label='Speed', value=5., unit='m / s')
    """
    length = me._entity.from_compatible("Length", "m", length=length)
    time = me.Entity.from_compatible("Time", "s", time=time)
    speed = length.q / time.q
    return me.Entity("Speed", speed)
```

### Subclassing `EntityCollection`

```python
from __future__ import annotations

from typing import TYPE_CHECKING

import mammos_entity as me

if TYPE_CHECKING:
    import mammos_entity


class IntrinsicProperties(me.EntityCollection):
    """Intrinsic properties of a ferromagnet."""

    def __init__(
        self,
        Ms: mammos_entity.Entity,
        A: mammos_entity.Entity,
        Tc: mammos_entity.Entity,
        extra_object: str,
        description: str = "",
    ) -> None:
        """Create a new instance.

        Args:
            Ms: :entity:`SpontaneousMagnetization`.
            A: :entity:`ExchangeStiffnessConstant`.
            Tc: :entity:`CurieTemperature`.
            extra_object: A custom object specific to this class.
            description: Description of the collection.
        """
        me._entity.ensure_entity("SpontaneousMagnetization", Ms=Ms)
        me._entity.ensure_entity("ExchangeStiffnessConstant", A=A)
        me._entity.ensure_entity("CurieTemperature", Tc=Tc)
        super().__init__(description=description, Ms=Ms, A=A, Tc=Tc)
        self._extra_object = extra_object

    @property
    def extra_object(self) -> str:
        """Custom subclass-specific object."""
        return self._extra_object
```

## Changelog fragments

- Follow the package repository's `changes/README.md` instructions.
- Most user-visible pull requests should include a Towncrier fragment.
- Internal-only changes can use the `misc` fragment type when the package allows
  it or avoid the changelog fragment all together (PR label `no-changelog-entry-required`).

## AI-assisted contributions

Code generated with help from an AI assistant must be marked in the commit
message:

```text
Assisted-by: agent:model
```

For example:

```text
Assisted-by: OpenAI Codex:gpt-5.3-codex
```

The human submitter remains responsible for reviewing, testing, and maintaining
the contribution.