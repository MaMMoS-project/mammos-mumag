"""Results functions."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
import pyvista as pv
from scipy.optimize import minimize
from typing import NamedTuple
from dataclasses import dataclass


import mammos_entity as me
import mammos_units as u


@dataclass(frozen=True)
class LoopResults:
    """Class LoopResults."""

    dataframe: pd.DataFrame
    configurations: list[pathlib.Path] | None = None

    def plot(self, duplicate: bool = True, configuration_marks: bool = False) -> None:
        """Plot hysteresis loop."""
        plt.plot(self.dataframe["mu0_Hext"], self.dataframe["polarisation"])
        j = 0
        if configuration_marks:
            for i, r in self.dataframe.iterrows():
                if r["idx"] != j:
                    plt.plot(r["mu0_Hext"], r["polarisation"], "rx")
                    j = r["idx"]
        if duplicate:
            plt.plot(-self.dataframe["mu0_Hext"], -self.dataframe["polarisation"])

    def plot_configuration(self, idx: int) -> None:
        """Plot configuration with index `idx`."""
        conf = pv.read(self.configurations[idx])
        conf.plot()

    def get_extrinsic_properties(self) -> tuple:
        """Evaluate extrinsic properties."""
        h = self.dataframe["mu0_Hext"]
        m = self.dataframe["polarisation"]

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

    def get_linearized_segment(self):
        """Evaluate linearized segment."""
        h = 0.5  # threshold_training
        mar = 0.05  # margin_to_line
        m0 = 1.0  # m_guess
        i0 = 0  # index_adjustment
        try:
            upper_index = i0 + np.argmin(np.abs(self.dataframe["polarisation"] - h))
            hh_u = self.dataframe["mu0_Hext"].iloc[upper_index]
            df = self.dataframe[self.dataframe["mu0_Hext"] < hh_u]

            lower_index = i0 + np.argmin(np.abs(self.dataframe["polarisation"] >= 0))
            hh_l = self.dataframe["mu0_Hext"].iloc[lower_index]
            df = df[df["mu0_Hext"] >= hh_l]
        except Exception as e:
            print(f"[ERROR]: Exception: {e}")
            raise ValueError("Failed Extraction")

        if df.shape[0] < 10:
            print(
                f"[ERROR]: Less than 10 points in margin [0,{h}] for linear regression (only {df.shape[0]})"
            )
            return 0.0

        def line(x, m, b=0.0):
            return m * x + b

        def penalty_function(m, x, y, b=0.0):
            return np.sum((y - line(x=x, m=m, b=b)) ** 2)

        try:
            b = df["polarisation"].iloc[np.argmin(np.abs(df["mu0_Hext"]))]
            res = minimize(
                penalty_function,
                m0,
                args=(df["mu0_Hext"], df["polarisation"], b),
            )
            m_opt = res.x
            if not res.success:
                print(f"Optimization did not converge in general: {res.message}")
                raise ValueError("Failed Linearization")
            if m_opt > 1000 or m_opt < 0:
                print(f"[ERROR]: Slope is unreasonable: {res.x}")
                raise ValueError("Failed Linearization")
        except Exception as e:
            print(f"[ERROR]: Something did not work: {e}.")
            ValueError("Failed Linearization")

        try:
            margin = (
                np.abs(
                    self.dataframe["polarisation"]
                    - line(self.dataframe["mu0_Hext"], m_opt, b)
                )
                < mar
            )
            # npo = np.sum(margin)  # number_points_in_margin
            x_max_lin = np.max(self.dataframe["mu0_Hext"][margin])
        except Exception as e:
            print(f"[ERROR]: Failed x_max_lin extraction: {e}.")

        return x_max_lin
