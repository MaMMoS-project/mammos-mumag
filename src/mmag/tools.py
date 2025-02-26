"""Tool functions."""

import pathlib


def check_path(fname):
    """Check that file exists.

    :param fname: file path
    :type fname: str or pathlib.Path
    :raises FileNotFoundError: File not found.
    """
    path = pathlib.Path(fname).resolve()
    if not path.is_file():
        raise FileNotFoundError("File not found.")
