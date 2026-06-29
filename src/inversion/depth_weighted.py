"""
depth_weighted.py
-----------------

Depth-weighted gravity inversion.

Author
------
AvionicS-7

Project
-------
GeoGravityLab
"""

from __future__ import annotations

import numpy as np

from .base import BaseInversion


class DepthWeightedInversion(BaseInversion):
    """
    Depth-weighted Tikhonov inversion.

    The weighting function is

        w(z) = (depth / z)^beta

    which compensates for the natural decay of
    gravity sensitivity with depth.
    """

    def __init__(
    self,
    beta: float = 0.5,
    lambda_factor: float = 0.05,
    ):

        super().__init__()

        self.beta = beta
        self.lambda_factor = lambda_factor

        self.weights = None
        self.lambda_ = None

    def compute_weights(
        self,
        z_centers: np.ndarray,
    ) -> np.ndarray:
        """
        Compute depth weighting.
        """

        reference_depth = np.max(z_centers)

        self.weights = (
            reference_depth / z_centers
        ) ** self.beta

        return self.weights

    def fit(
        self,
        G: np.ndarray,
        d: np.ndarray,
        z_centers: np.ndarray,
    ):
        """
        Solve

        (GᵀG + λW)m = Gᵀd
        """

        self.compute_weights(
            z_centers
        )

        GTG = G.T @ G

        GTd = G.T @ d

        self.lambda_ = (
            self.lambda_factor
            * np.max(
                np.diag(GTG)
            )
        )

        W = np.diag(
            self.weights
        )

        self.model = np.linalg.solve(
            GTG + self.lambda_ * W,
            GTd,
        )

        return self.model

    def summary(self):

        print("=" * 60)

        print("Depth Weighted Inversion")

        print("=" * 60)

        print(
            f"Beta : {self.beta}"
        )

        print(
            f"Lambda : {self.lambda_:.4e}"
        )

        print(
            f"Recovered Cells : {len(self.model)}"
        )