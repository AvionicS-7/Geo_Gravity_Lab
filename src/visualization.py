"""
visualization.py
----------------

Visualization utilities for GeoGravityLab.

Author
------
AvionicS-7

Project
-------
GeoGravityLab
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


class GravityVisualizer:
    """
    Visualization class for geological models,
    gravity anomalies and inversion results.
    """

    @staticmethod
    def plot_density_model(
        density: np.ndarray,
        title: str = "Density Model",
        cmap: str = "viridis",
    ):
        plt.figure(figsize=(10, 5))

        plt.imshow(
            density,
            origin="upper",
            aspect="auto",
            cmap=cmap,
        )

        plt.colorbar(
            label="Density Contrast (kg/m³)"
        )

        plt.xlabel("X Cell")

        plt.ylabel("Depth Cell")

        plt.title(title)

        plt.tight_layout()

        plt.show()

    @staticmethod
    def plot_gravity(
        x: np.ndarray,
        gravity: np.ndarray,
        title: str = "Gravity Anomaly",
    ):

        plt.figure(figsize=(10,4))

        plt.plot(
            x,
            gravity,
            linewidth=2,
        )

        plt.grid(True)

        plt.xlabel("Distance (m)")

        plt.ylabel("Gravity (mGal)")

        plt.title(title)

        plt.tight_layout()

        plt.show()

    @staticmethod
    def compare_gravity(
        x,
        observed,
        predicted,
    ):

        plt.figure(figsize=(10,4))

        plt.plot(
            x,
            observed,
            label="Observed",
            linewidth=2,
        )

        plt.plot(
            x,
            predicted,
            "--",
            label="Predicted",
            linewidth=2,
        )

        plt.legend()

        plt.grid(True)

        plt.xlabel("Distance (m)")

        plt.ylabel("Gravity (mGal)")

        plt.title("Observed vs Predicted")

        plt.tight_layout()

        plt.show()

    @staticmethod
    def plot_residual(
        x,
        residual,
    ):

        plt.figure(figsize=(10,4))

        plt.plot(
            x,
            residual,
            color="red",
        )

        plt.grid(True)

        plt.xlabel("Distance (m)")

        plt.ylabel("Residual")

        plt.title("Residual")

        plt.tight_layout()

        plt.show()

    @staticmethod
    def plot_convergence(
        rmse_history,
    ):

        plt.figure(figsize=(8,4))

        plt.plot(
            rmse_history,
            marker="o",
            linewidth=2,
        )

        plt.grid(True)

        plt.xlabel("Iteration")

        plt.ylabel("RMSE")

        plt.title("IRLS Convergence")

        plt.tight_layout()

        plt.show()

    @staticmethod
    def compare_models(
        true_model,
        recovered_model,
    ):

        fig, ax = plt.subplots(
            1,
            2,
            figsize=(12,5),
        )

        im1 = ax[0].imshow(
            true_model,
            origin="upper",
            aspect="auto",
        )

        ax[0].set_title(
            "True Model"
        )

        plt.colorbar(
            im1,
            ax=ax[0],
        )

        im2 = ax[1].imshow(
            recovered_model,
            origin="upper",
            aspect="auto",
        )

        ax[1].set_title(
            "Recovered Model"
        )

        plt.colorbar(
            im2,
            ax=ax[1],
        )

        plt.tight_layout()

        plt.show()

    @staticmethod
    def compare_all_methods(
        true_model,
        tik,
        depth,
        irls,
    ):

        fig, ax = plt.subplots(
            2,
            2,
            figsize=(12,8),
        )

        titles = [
            "True",
            "Tikhonov",
            "Depth Weighted",
            "IRLS",
        ]

        models = [
            true_model,
            tik,
            depth,
            irls,
        ]

        for a, m, t in zip(
            ax.ravel(),
            models,
            titles,
        ):

            im = a.imshow(
                m,
                origin="upper",
                aspect="auto",
            )

            a.set_title(t)

            plt.colorbar(
                im,
                ax=a,
            )

        plt.tight_layout()

        plt.show()