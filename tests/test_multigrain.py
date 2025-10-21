"""Check multigrain run."""

import mammos_units as u

import mammos_mumag

u.add_enabled_equivalencies(u.magnetic_flux_field())


def test_mg_run_zeros(tmp_path):
    """Test multigrain run with zeros.

    This test only checks that the function produces the correct output type.
    """
    results_hysteresis = mammos_mumag.hysteresis.run(
        mesh="cube40_3plat_grains12_gsize20",
        Ms=[0] * 13,
        A=[0] * 13,
        K1=[0] * 13,
        theta=[0] * 13,
        phi=[0] * 13,
        h_start=(1 * u.T).to("A/m"),
        h_final=-(1 * u.T).to("A/m"),
        h_n_steps=3,
        outdir=tmp_path,
    )
    assert isinstance(results_hysteresis, mammos_mumag.hysteresis.Result)


def test_mg_run(tmp_path):
    """Test uniform multigrain run.

    This test checks that a multigrain simulation with uniform intrinsic properties
    gives the same result as a singlegrain simulation.
    """
    results_mg = mammos_mumag.hysteresis.run(
        mesh="cube40_3plat_grains12_gsize20",
        Ms=[1.28e6] * 13,
        A=[7.7e-12] * 13,
        K1=[4.3e6] * 13,
        theta=[0] * 13,
        phi=[0] * 13,
        h_start=(10 * u.T).to("A/m"),
        h_final=-(10 * u.T).to("A/m"),
        h_n_steps=20,
        outdir=tmp_path,
    )
    results_sg = mammos_mumag.hysteresis.run(
        mesh="cube40_singlegrain_msize2",
        Ms=1.28e6,
        A=7.7e-12,
        K1=4.3e6,
        theta=0,
        phi=0,
        h_start=(10 * u.T).to("A/m"),
        h_final=-(10 * u.T).to("A/m"),
        h_n_steps=20,
        outdir=tmp_path,
    )
    assert u.allclose(results_sg.M.q, results_mg.M.q, rtol=1e-3)
