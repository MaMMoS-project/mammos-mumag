"""Test parameters file i/o."""

import numpy as np
import pytest
from pydantic import ValidationError

from mammos_mumag.parameters import Parameters


def test_m_vect():
    """Test type-checking of parameter `m_vect`."""
    with pytest.raises(ValidationError):
        Parameters(m_vect=1)
    with pytest.raises(ValidationError):
        Parameters(m_vect=1.0)
    with pytest.raises(ValidationError):
        Parameters(m_vect=[1])
    with pytest.raises(ValidationError):
        Parameters(m_vect=[1, 2])
    with pytest.raises(ValidationError):
        Parameters(m_vect=[1, 2, 3, 4])
    par = Parameters(m_vect=np.array([1, 2, 3]))
    assert par.m == [1.0 / np.sqrt(14), 2.0 / np.sqrt(14), 3.0 / np.sqrt(14)]


def test_h_vect():
    """Test type-checking of parameter `h_vect`."""
    with pytest.raises(ValidationError):
        Parameters(h_vect=1)
    with pytest.raises(ValidationError):
        Parameters(h_vect=1.0)
    with pytest.raises(ValidationError):
        Parameters(h_vect=[1])
    with pytest.raises(ValidationError):
        Parameters(h_vect=[1, 2])
    with pytest.raises(ValidationError):
        Parameters(h_vect=[1, 2, 3, 4])
    par = Parameters(h_vect=np.array([1, 2, 3]))
    assert par.h == [1.0 / np.sqrt(14), 2.0 / np.sqrt(14), 3.0 / np.sqrt(14)]


def test_parameters_file(DATA, tmp_path):
    """Test parameters files i/o.

    This test defines a :py:class:`~mammos_mumag.parameters.Parameters` instance.
    Then it is written to `p2` and `yaml` formats.
    Then two new empty :py:class:`~mammos_mumag.parameters.Parameters`
    instances are created reading, respectively, the `p2` and `yaml files.
    The first parameter instance is tested with the other two, by checking if each
    parameter is exactly equal or sufficiently close to the original one.
    """
    par = Parameters(
        m=[1, 0, 0],
        hstart=8.0,
        hfinal=-1.5,
        hstep=-0.01,
        iter_max=5000,
    )

    par.write_p2(tmp_path / "par.p2")
    par.write_yaml(tmp_path / "par.yaml")

    par_1 = Parameters()
    par_1.read(tmp_path / "par.p2")
    assert are_parameters_equal(par, par_1)

    par_2 = Parameters()
    par_2.read(tmp_path / "par.yaml")
    assert are_parameters_equal(par, par_2)


def are_parameters_equal(d1, d2):
    """Compare parameters.

    :return: True if parameters are equal, False otherwise.
    :rtype: bool
    """
    dict_2 = d2.__dict__
    for k, val in d1.__dict__.items():
        if k not in dict_2:
            return False
        if isinstance(val, str | int) and dict_2[k] != val:
            return False
        if isinstance(val, float) and abs(dict_2[k] - val) > 1.0e-11:
            return False
        if (
            isinstance(val, list)
            and sum([abs(val[i] - dict_2[k][i]) for i in range(len(val))]) > 1
        ):
            return False
    return True
