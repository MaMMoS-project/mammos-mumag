"""Tool functions."""

import pathlib
from textwrap import dedent

import mammos_mumag


def check_path(fname: str | pathlib.Path) -> pathlib.Path:
    """Check that file exists.

    Args:
        fname: File path.

    Returns:
        File path.

    Raises:
        FileNotFoundError: File not found.

    """
    path = pathlib.Path(fname).resolve()
    if not path.is_file():
        raise FileNotFoundError("File not found.")
    return path


def check_dir(outdir: str | pathlib.Path) -> pathlib.Path:
    """Check that directory exists.

    Args:
        outdir: Directory path.

    Returns:
        Checked directory path.

    """
    outdir = pathlib.Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)
    return outdir


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
