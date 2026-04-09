"""Check loop script."""

import mammos_entity as me
import numpy as np

from mammos_mumag.simulation import Simulation


def test_loop(DATA, tmp_path):
    """Test loop."""
    sim = Simulation(
        mesh="cube20_singlegrain_msize2",
        materials_filepath=DATA / "cube.krn",
        parameters_filepath=DATA / "cube.p2",
    )

    # run loop
    sim.run_loop(outdir=tmp_path, name="cube")

    # check hysteresis loop
    content_1 = me.from_csv(DATA / "loop" / "cube.csv")
    content_2 = me.from_csv(tmp_path / "cube.csv")
    assert np.all(content_1.configuration_type == content_2.configuration_type)
    assert np.allclose(content_1.B_ext.q, content_2.B_ext.q)
    assert np.allclose(content_1.J.q, content_2.J.q)
    assert np.allclose(content_1.Jx.q, content_2.Jx.q)
    assert np.allclose(content_1.Jy.q, content_2.Jy.q)
    assert np.allclose(content_1.Jz.q, content_2.Jz.q)
    assert np.allclose(content_1.energy_density.q, content_2.energy_density.q)
