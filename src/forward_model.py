"""
forward_model.py
----------------

Gravity forward modelling routines for GeoGravityLab.

Implements the linear relationship

    d = G m

where

    d : gravity observations
    G : sensitivity matrix
    m : density contrast model

Author
------
PHEONIX

Project
-------
GeoGravityLab
"""

from __future__ import annotations

import numpy as np

from src.constants import (
    GRAVITATIONAL_CONSTANT,
    SI_TO_MICROGAL,
)


class ForwardModel:
    """
    Linear gravity forward modelling.

    Notes
    -----
    Uses the infinite-strike 2D approximation.

    The sensitivity matrix is computed once and can
    be reused by all inversion algorithms.
    """

    def __init__(self):

        self.G = None

    def build_sensitivity_matrix(
        self,
        x_obs: np.ndarray,
        x_grid: np.ndarray,
        z_grid: np.ndarray,
        dx: float,
        dz: float,
        z_obs: float = 0.0,
        regularization: float = 1e-6,
    ) -> np.ndarray:
        """
        Build the gravity sensitivity matrix.

        Parameters
        ----------
        x_obs
            Observation positions.

        x_grid
            X coordinates of cell centers.

        z_grid
            Z coordinates of cell centers.

        dx
            Cell width.

        dz
            Cell height.

        z_obs
            Observation elevation.

        regularization
            Small value added to avoid singularities.

        Returns
        -------
        ndarray
            Sensitivity matrix.
        """

        n_obs = len(x_obs)

        n_cells = x_grid.size

        G = np.zeros((n_obs, n_cells))

        x_flat = x_grid.flatten()

        z_flat = z_grid.flatten()

        for i, xo in enumerate(x_obs):

            r2 = (
                (x_flat - xo) ** 2
                + (z_flat - z_obs) ** 2
                + regularization
            )

            G[i, :] = (
                (
                    2.0
                    * GRAVITATIONAL_CONSTANT
                    * dx
                    * dz
                    * z_flat
                )
                / r2
            ) * SI_TO_MICROGAL * 1000.0

        self.G = G

        return G

    def forward(
        self,
        density_model: np.ndarray,
    ) -> np.ndarray:
        """
        Compute gravity observations.

        Parameters
        ----------
        density_model
            Flattened density contrast model.

        Returns
        -------
        ndarray
            Gravity anomaly.
        """

        if self.G is None:

            raise RuntimeError(
                "Sensitivity matrix has not been built."
            )

        return self.G @ density_model

    def predict(
        self,
        density_model: np.ndarray,
    ) -> np.ndarray:
        """
        Alias for forward().
        """

        return self.forward(density_model)

    @property
    def n_observations(self):

        if self.G is None:
            return 0

        return self.G.shape[0]

    @property
    def n_cells(self):

        if self.G is None:
            return 0

        return self.G.shape[1]

    def summary(self):

        print("=" * 60)
        print("Forward Model")
        print("=" * 60)

        print(f"Observations : {self.n_observations}")
        print(f"Cells        : {self.n_cells}")