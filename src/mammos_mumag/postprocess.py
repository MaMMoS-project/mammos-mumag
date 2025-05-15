"""Postprocess functions."""

import numpy as np

import mammos_entity as me


def get_extrinsic_properties(hystloop):
    """Evaluate extrinsic properties from hysteresis loop."""
    h = hystloop["mu0_Hext"]
    m = hystloop["pol"]

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
    Hc = me.Hc(Hc)
    Mr = me.Mr(Mr)
    BHmax = me.BHmax()
    return Hc, Mr, BHmax
