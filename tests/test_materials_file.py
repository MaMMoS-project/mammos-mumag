"""Test materials file i/o."""

import mammos_entity as me
import mammos_units as u
import numpy as np
import pytest
from pydantic import ValidationError

from mammos_mumag.materials import MaterialDomain, Materials


def test_materials_file(DATA, tmp_path):
    """Test materials files i/o.

    This test defines a :py:class:`~mammos_mumag.materials.Materials` instance
    with a certain :py:attr:`domains` attribute. Then the material is written
    to `krn` and `yaml` formats.
    Then it creates two new empty :py:class:`~mammos_mumag.materials.Materials`
    instances that read, respectively, the `krn` and `yaml files.
    The first materials is tested with the other two, by checking if each
    material in each domain is sufficiently close to the original one.
    """
    mat = Materials(
        domains=[
            {
                "theta": 0,
                "phi": 0,
                "K1": me.Ku(4.9e06, unit=u.J / u.m**3),
                "K2": me.Ku(0, unit=u.J / u.m**3),
                "Ms": me.Ms(1.61, unit=u.A / u.m),
                "A": me.A(8.0e-11, unit=u.J / u.m),
            },
            {
                "theta": 0,
                "phi": 0,
                "K1": me.Ku(0, unit=u.J / u.m**3),
                "K2": me.Ku(0, unit=u.J / u.m**3),
                "Ms": me.Ms(0, unit=u.A / u.m),
                "A": me.A(0, unit=u.J / u.m),
            },
            {
                "theta": 0,
                "phi": 0,
                "K1": me.Ku(0, unit=u.J / u.m**3),
                "K2": me.Ku(0, unit=u.J / u.m**3),
                "Ms": me.Ms(0, unit=u.A / u.m),
                "A": me.A(0, unit=u.J / u.m),
            },
        ]
    )

    mat.write_krn(tmp_path / "mat.krn")
    mat.write_yaml(tmp_path / "mat.yaml")

    mat_1 = Materials()
    mat_1.read(tmp_path / "mat.krn")
    assert are_domains_equal(mat.domains, mat_1.domains)

    mat_2 = Materials()
    mat_2.read(tmp_path / "mat.yaml")
    assert are_domains_equal(mat.domains, mat_2.domains)

    mat_3 = Materials()
    assert not are_domains_equal(mat.domains, mat_3.domains)

    mat_4 = Materials(
        domains=[
            {
                "theta": 0,
                "phi": 0,
                "K1": me.Ku(1, unit=u.J / u.m**3),
                "K2": me.Ku(0, unit=u.J / u.m**3),
                "Ms": me.Ms(2, unit=u.A / u.m),
                "A": me.A(3, unit=u.J / u.m),
            },
            {
                "theta": 0,
                "phi": 0,
                "K1": me.Ku(0, unit=u.J / u.m**3),
                "K2": me.Ku(0, unit=u.J / u.m**3),
                "Ms": me.Ms(0, unit=u.A / u.m),
                "A": me.A(0, unit=u.J / u.m),
            },
            {
                "theta": 0,
                "phi": 0,
                "K1": me.Ku(0, unit=u.J / u.m**3),
                "K2": me.Ku(0, unit=u.J / u.m**3),
                "Ms": me.Ms(0, unit=u.A / u.m),
                "A": me.A(0, unit=u.J / u.m),
            },
        ]
    )
    assert not are_domains_equal(mat.domains, mat_4.domains)


def are_domains_equal(d1, d2):
    """Compare domains.

    :return: True if domains are equal, False otherwise.
    :rtype: bool
    """
    if len(d1) != len(d2):
        return False
    for i, d1_i in enumerate(d1):
        d2_i = d2[i]
        if not (
            np.allclose(d1_i.theta, d2_i.theta)
            and np.allclose(d1_i.phi, d2_i.phi)
            and d1_i.K1 == d2_i.K1
            and d1_i.K2 == d2_i.K2
            and d1_i.Ms == d2_i.Ms
            and d1_i.A == d2_i.A
        ):
            return False
    return True


def test_materials_types():
    """Test MaterialDomain instances initialized with different types.

    The instances are initialized with the same value, so we expect
    them to be defined as equal.
    """
    dom_1 = MaterialDomain(
        theta=0,
        phi=0,
        K1=me.Ku(1, unit=u.J / u.m**3),
        K2=me.Ku(2, unit=u.J / u.m**3),
        Ms=me.Ms(3, unit=u.A / u.m),
        A=me.A(4, unit=u.J / u.m),
    )

    dom_2 = MaterialDomain(
        theta=0,
        phi=0,
        K1=1,
        K2=2,
        Ms=3,
        A=4,
    )

    dom_3 = MaterialDomain(
        theta=0,
        phi=0,
        K1=1 * u.J / u.m**3,
        K2=2 * u.J / u.m**3,
        Ms=3 * u.A / u.m,
        A=4 * u.J / u.m,
    )

    dom_4 = MaterialDomain(
        theta=0,
        phi=0,
        K1=me.Ku(1, unit=u.J / u.m**3).q,
        K2=me.Ku(2, unit=u.J / u.m**3).q,
        Ms=me.Ms(3, unit=u.A / u.m).q,
        A=me.A(4, unit=u.J / u.m).q,
    )

    assert are_domains_equal([dom_1], [dom_2])
    assert are_domains_equal([dom_1], [dom_3])
    assert are_domains_equal([dom_1], [dom_4])

    dom_5 = MaterialDomain()
    dom_6 = MaterialDomain(
        theta=0,
        phi=0,
        K1=0,
        K2=0,
        Ms=0,
        A=0,
    )
    assert are_domains_equal([dom_5], [dom_6])


def test_wrong_domains():
    """Use wrong types in definition.

    All tests are supposed to raise `ValidationError`.
    """
    with pytest.raises(ValidationError):
        MaterialDomain(K1="K1")
    with pytest.raises(ValidationError):
        MaterialDomain(theta=0 * u.rad)
