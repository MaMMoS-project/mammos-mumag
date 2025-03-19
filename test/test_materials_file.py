"""Test materials file i/o."""

import pathlib
import numpy as np

from mammos_mmag.materials import Materials

DATA = pathlib.Path(__file__).resolve().parent / "data"


def test_materials_file(tmp_path):
    """Test materials files i/o.

    This test defines a :py:class:`~mammos_mmag.materials.Materials` instance
    with a certain :py:attr:`domains` attribute. Then the material is written
    to `krn` and `yaml` formats.
    Then it creates two new empty :py:class:`~mammos_mmag.materials.Materials`
    instances that read, respectively, the `krn` and `yaml files.
    The first materials is tested with the other two, by checking if each
    material in each domain is sufficiently close to the original one.
    """
    mat = Materials(
        domains=[
            {
                "theta": 0.0,
                "phi": 0.0,
                "K1": 4e-7 * np.pi * 4.9e06,
                "K2": 0.0,
                "Js": 1.61,
                "A": 4e-7 * np.pi * 8.0e-11,
            },
            {
                "theta": 0.0,
                "phi": 0.0,
                "K1": 0.0,
                "K2": 0.0,
                "Js": 0.0,
                "A": 0.0,
            },
            {
                "theta": 0.0,
                "phi": 0.0,
                "K1": 0.0,
                "K2": 0.0,
                "Js": 0.0,
                "A": 0.0,
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
    print(f"{type(mat.domains[0])=}")
    assert are_domains_equal(mat.domains, mat_2.domains)


def are_domains_equal(d1, d2):
    """Compare domains.

    :return: True if domains are equal, False otherwise.
    :rtype: bool
    """
    if len(d1) != len(d2):
        return False
    diff = 0.0
    for i, d1_i in enumerate(d1):
        d2_i = d2[i]
        diff += np.linalg.norm(
            [
                d1_i.theta - d2_i.theta,
                d1_i.phi - d2_i.phi,
                d1_i.K1 - d2_i.K1,
                d1_i.K2 - d2_i.K2,
                d1_i.Js - d2_i.Js,
                d1_i.A - d2_i.A,
            ]
        )
    return diff < 1.0e-8
