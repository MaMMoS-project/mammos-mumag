"""Test parameters file i/o."""

import pathlib

from mammos_mmag.parameters import Parameters

DATA = pathlib.Path(__file__).resolve().parent / "data"


def test_parameters_file(tmp_path):
    """Test parameters files i/o.

    This test defines a :py:class:`~mammos_mmag.parameters.Parameters` instance.
    Then it is written to `p2` and `yaml` formats.
    Then two new empty :py:class:`~mammos_mmag.parameters.Parameters`
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
        if isinstance(val, (str, int)):
            if dict_2[k] != val:
                return False
        if isinstance(val, float):
            if abs(dict_2[k] - val) > 1.0e-11:
                return False
        if isinstance(val, list):
            if sum([abs(val[i] - dict_2[k][i]) for i in range(len(val))]) > 1:
                return False
    return True
