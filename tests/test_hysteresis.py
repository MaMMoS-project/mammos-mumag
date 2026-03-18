"""Test hysteresis module."""

import mammos_entity as me
import mammos_units as u
import numpy as np
import pytest

from mammos_mumag.hysteresis import run


def test_hysteresis_run_inputs_python(DATA, tmp_path):
    """Test validity of Python base type inputs."""
    run(
        Ms=0,
        A=0,
        K1=0,
        theta=0,
        phi=0,
        mesh=DATA / "cube.fly",
        h_n_steps=1,
        outdir=tmp_path,
    )
    run(
        Ms=0.0,
        A=0.0,
        K1=0.0,
        theta=0.0,
        phi=0.0,
        mesh=DATA / "cube.fly",
        h_n_steps=1,
        outdir=tmp_path,
    )
    run(
        Ms=[0],
        A=[0],
        K1=[0],
        theta=[0],
        phi=[0],
        mesh=DATA / "cube.fly",
        h_n_steps=1,
        outdir=tmp_path,
    )


def test_hysteresis_run_inputs_numpy_array(DATA, tmp_path):
    """Test validity of numpy array inputs."""
    run(
        Ms=np.array(0),
        A=np.array(0),
        K1=np.array(0),
        theta=np.array(0),
        phi=np.array(0),
        mesh=DATA / "cube.fly",
        h_n_steps=1,
        outdir=tmp_path,
    )
    run(
        Ms=np.array([0]),
        A=np.array([0]),
        K1=np.array([0]),
        theta=np.array([0]),
        phi=np.array([0]),
        mesh=DATA / "cube.fly",
        h_n_steps=1,
        outdir=tmp_path,
    )
    run(
        Ms=np.array([[0]]),
        A=np.array([[0]]),
        K1=np.array([[0]]),
        theta=np.array([[0]]),
        phi=np.array([[0]]),
        mesh=DATA / "cube.fly",
        h_n_steps=1,
        outdir=tmp_path,
    )


def test_hysteresis_run_inputs_Quantity(DATA, tmp_path):
    """Test validity of u.Quantity inputs."""
    run(
        Ms=0 * u.A / u.m,
        A=0 * u.J / u.m,
        K1=0 * u.J / u.m**3,
        theta=0 * u.dimensionless_unscaled,
        phi=0 * u.dimensionless_unscaled,
        mesh=DATA / "cube.fly",
        h_n_steps=1,
        outdir=tmp_path,
    )
    run(
        Ms=[0 * u.A / u.m],
        A=[0 * u.J / u.m],
        K1=[0 * u.J / u.m**3],
        theta=[0 * u.dimensionless_unscaled],
        phi=[0 * u.dimensionless_unscaled],
        mesh=DATA / "cube.fly",
        h_n_steps=1,
        outdir=tmp_path,
    )


def test_hysteresis_run_inputs_Entity(DATA, tmp_path):
    """Test validity of me.Entity inputs."""
    run(
        Ms=me.Ms(),
        A=me.A(),
        K1=me.K1(),
        theta=me.Entity("Angle"),
        phi=me.Entity("Angle"),
        mesh=DATA / "cube.fly",
        h_n_steps=1,
        outdir=tmp_path,
    )
    run(
        Ms=me.operations.concat_flat([me.Ms()]),
        A=me.operations.concat_flat([me.A()]),
        K1=me.operations.concat_flat([me.K1()]),
        theta=me.operations.concat_flat([me.Entity("Angle")]),
        phi=me.operations.concat_flat([me.Entity("Angle")]),
        mesh=DATA / "cube.fly",
        h_n_steps=1,
        outdir=tmp_path,
    )


def test_hysteresis_run_radians(DATA, tmp_path):
    """Test use of radians for angles.

    The Entity "Angle" does not accept any units, but we allow implicit conversion
    to dimensionless in the code.
    """
    run(
        Ms=0 * u.A / u.m,
        A=0 * u.J / u.m,
        K1=0 * u.J / u.m**3,
        theta=30 * u.degree,
        phi=1 * u.rad,
        mesh=DATA / "cube.fly",
        h_n_steps=1,
        outdir=tmp_path,
    )


def test_inconsistent_dimensions(DATA, tmp_path):
    """Test failure when materials parameters have different dimensions."""
    with pytest.raises(ValueError):
        run(
            Ms=0,
            A=[0, 1],
            K1=0,
            theta=0,
            phi=0,
            mesh=DATA / "cube.fly",
            h_n_steps=1,
            outdir=tmp_path,
        )
