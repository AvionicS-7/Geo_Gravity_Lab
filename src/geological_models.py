"""
geological_models.py
--------------------

Utilities for creating geological density models.

Author
------
PHEONIX

Project
-------
GeoGravityLab
"""

from __future__ import annotations

import numpy as np

from src.models import Body
from src.models import GeologicalModel


class GeologicalModelBuilder:
    """
    Converts GeologicalModel objects into numerical
    density grids for forward modelling.
    """

    def __init__(
        self,
        model: GeologicalModel,
    ):

        self.model = model

    def generate_density_grid(
        self,
        X: np.ndarray,
        Z: np.ndarray,
    ) -> np.ndarray:
        """
        Generate density contrast grid.

        Parameters
        ----------
        X
            X meshgrid.

        Z
            Z meshgrid.

        Returns
        -------
        ndarray
            Density model.
        """

        density = np.zeros_like(X, dtype=float)

        for body in self.model:

            mask = (
                (X >= body.left)
                & (X <= body.right)
                & (Z >= body.top)
                & (Z <= body.bottom)
            )

            density[mask] = body.density

        return density

    def flatten(
        self,
        X: np.ndarray,
        Z: np.ndarray,
    ) -> np.ndarray:
        """
        Return flattened density model.
        """

        return self.generate_density_grid(
            X,
            Z,
        ).ravel()