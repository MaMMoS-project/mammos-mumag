"""Postprocess functions."""

import numpy as np

import mammos_entity as me
import mammos_units as u


def get_extrinsic_properties(hystloop):
    """Evaluate extrinsic properties from hysteresis loop."""
    h = hystloop["mu0_Hext"]  # .to_numpy() * u.T
    # h = h.to("A/m", equivalencies=u.magnetic_flux_field())
    m = hystloop["pol"]  # .to_numpy() * u.T
    # m = m.to("A/m", equivalencies=u.magnetic_flux_field())

    sign_changes_m = np.where(np.diff(np.sign(m)))[0]
    sign_changes_h = np.where(np.diff(np.sign(h)))[0]

    if len(sign_changes_m) == 0:
        raise ValueError("No Hc")

    if len(sign_changes_h) == 0:
        raise ValueError("No Mc")

    index_before = sign_changes_m[0]
    index_after = sign_changes_m[0] + 1
    Hc = -1 * np.interp(
        0,
        [m[index_before], m[index_after]],
        [h[index_before], h[index_after]],
    )

    index_before = sign_changes_h[0]
    index_after = sign_changes_h[0] + 1
    Mr = np.interp(
        0,
        [h[index_before], h[index_after]],
        [m[index_before], m[index_after]],
    )

    Hc = me.Hc((Hc * u.T).to("A/m", equivalencies=u.magnetic_flux_field()))
    Mr = me.Mr((Mr * u.T).to("A/m", equivalencies=u.magnetic_flux_field()))
    bh = (h.to_numpy() * u.T) * (m.to_numpy() * u.T).to(
        "A/m", equivalencies=u.magnetic_flux_field()
    )
    BHmax = me.BHmax(max(bh))
    return Hc, Mr, BHmax
