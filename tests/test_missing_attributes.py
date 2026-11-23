"""Test ``Simulation._check_attributes`` method."""

import pytest
from pydantic import ValidationError

from mammos_mumag.simulation import Simulation


def test_missing_mesh():
    """Test impossible Simulation instantation without mesh."""
    with pytest.raises(ValidationError):
        Simulation()


def test_missing_attributes_exani(DATA):
    """Test necessary attributes for script 'exani'."""
    sim = Simulation(mesh=DATA / "cube.fly")
    with pytest.raises(AttributeError, match="have not been defined yet: materials$"):
        sim.run_exani()


def test_missing_attributes_external(DATA):
    """Test necessary attributes for script 'external'."""
    sim = Simulation(mesh=DATA / "cube.fly")
    with pytest.raises(
        AttributeError, match="have not been defined yet: materials, parameters$"
    ):
        sim.run_external()


def test_missing_attributes_hmag(DATA):
    """Test necessary attributes for script 'hmag'."""
    sim = Simulation(mesh=DATA / "cube.fly")
    with pytest.raises(AttributeError, match="have not been defined yet: materials$"):
        sim.run_hmag()


def test_missing_attributes_loop(DATA):
    """Test necessary attributes for script 'loop'."""
    sim = Simulation(mesh=DATA / "cube.fly")
    with pytest.raises(
        AttributeError, match="have not been defined yet: materials, parameters$"
    ):
        sim.run_loop()


def test_missing_attributes_magnetization(DATA):
    """Test necessary attributes for script 'magnetization'."""
    sim = Simulation(mesh=DATA / "cube.fly")
    with pytest.raises(
        AttributeError, match="have not been defined yet: materials, parameters$"
    ):
        sim.run_magnetization()


def test_missing_attributes_mapping(DATA):
    """Test necessary attributes for script 'mapping'."""
    sim = Simulation(mesh=DATA / "cube.fly")
    with pytest.raises(
        AttributeError, match="have not been defined yet: materials, parameters$"
    ):
        sim.run_mapping()


def test_missing_attributes_materials(DATA):
    """Test necessary attributes for script 'materials'."""
    sim = Simulation(mesh=DATA / "cube.fly")
    with pytest.raises(AttributeError, match="have not been defined yet: materials$"):
        sim.run_materials()


def test_missing_attributes_store(DATA):
    """Test necessary attributes for script 'store'."""
    sim = Simulation(mesh=DATA / "cube.fly")
    with pytest.raises(
        AttributeError, match="have not been defined yet: materials, parameters$"
    ):
        sim.run_store()
