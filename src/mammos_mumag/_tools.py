"""Tool functions."""

from textwrap import dedent

import mammos_mumag


def check_esys_escript() -> None:
    """Check if esys_escript is found in PATH.

    Raises:
        SystemError: esys-escript is not found

    """
    if mammos_mumag._run_escript_bin is None:
        raise SystemError(
            dedent(
                """
                esys-escript is not found.
                Is it correctly installed?
                Consider installing esys-escript in your environment with
                $ conda install esys-escript -c conda-forge
                or, using pixi,
                $ pixi add esys-escript
                """
            )
        )
