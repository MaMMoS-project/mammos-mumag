"""Results functions."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
import pyvista as pv
from typing import NamedTuple
from dataclasses import dataclass


import mammos_entity as me
import mammos_units as u


@dataclass(frozen=True)
class ResultsLoop:
    """Class ResultsLoop."""

    dataframe: pd.DataFrame
    configurations: list[pathlib.Path] | None = None

    def plot(self, duplicate: bool = True) -> None:
        """Plot hysteresis loop."""
        plt.plot(self.dataframe["mu0_Hext"], self.dataframe["pol"])
        j = 0
        for i, r in self.dataframe.iterrows():
            if r["idx"] != j:
                plt.plot(r["mu0_Hext"], r["pol"], "rx")
                j = r["idx"]
        if duplicate:
            plt.plot(-self.dataframe["mu0_Hext"], -self.dataframe["pol"])

    def plot_configuration(self, idx: int) -> None:
        """Plot configuration with index `idx`."""
        conf = pv.read(self.configurations[idx])
        conf.plot()

    def get_extrinsic_properties(self) -> tuple:
        """Evaluate extrinsic properties."""
        h = self.dataframe["mu0_Hext"]
        m = self.dataframe["pol"]

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
        ExtrinsicProperties = NamedTuple(
            "ExtrinsicProperties",
            [("Hc", me.Entity), ("Mr", me.Entity), ("BHmax", me.Entity)],
        )
        extrprops = ExtrinsicProperties(
            me.Hc((Hc * u.T).to("A/m", equivalencies=u.magnetic_flux_field())),
            me.Mr((Mr * u.T).to("A/m", equivalencies=u.magnetic_flux_field())),
            me.BHmax(
                max(
                    (h.to_numpy() * u.T)
                    * (m.to_numpy() * u.T).to(
                        "A/m", equivalencies=u.magnetic_flux_field()
                    )
                )
            ),
        )
        return extrprops
