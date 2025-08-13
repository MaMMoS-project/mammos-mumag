"""Check loop script."""

import mammos_units as u

import mammos_mumag

u.add_enabled_equivalencies(u.magnetic_flux_field())


def test_run():
    """Test run."""
    results_hysteresis = mammos_mumag.hysteresis.run(
        mesh="cube20_singlegrain_msize2",
        Ms=1e6,
        A=1e-11,
        K1=1e5,
        hstart=(1 * u.T).to("A/m"),
        hfinal=-(1 * u.T).to("A/m"),
        hnsteps=3,
    )

    assert isinstance(results_hysteresis, mammos_mumag.hysteresis.Result)


def test_run_zeros():
    """Test run with zeros."""
    results_hysteresis = mammos_mumag.hysteresis.run(
        mesh="cube20_singlegrain_msize2",
        Ms=0,
        A=0,
        K1=0,
        hstart=(1 * u.T).to("A/m"),
        hfinal=-(1 * u.T).to("A/m"),
        hnsteps=3,
    )

    assert isinstance(results_hysteresis, mammos_mumag.hysteresis.Result)
