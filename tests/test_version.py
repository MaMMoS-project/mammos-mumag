import mammos_mumag


def test_version():
    """Check that __version__ exists and is a string."""
    assert isinstance(mammos_mumag.__version__, str)
