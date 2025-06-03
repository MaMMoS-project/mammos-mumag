"""Test configuration."""

import pathlib

import pytest


@pytest.fixture(scope="session")
def DATA():
    """Define DATA folder."""
    return pathlib.Path(__file__).resolve().parent / "data"
