"""
tikhonov.py
-----------

Tikhonov Regularized Gravity Inversion.

Author
------
PHEONIX

Project
-------
GeoGravityLab
"""

from __future__ import annotations

import numpy as np

from .base import BaseInversion


class TikhonovInversion(BaseInversion):
    """
    Solve

        min ||Gm-d||² + λ²||m||²
    """

    def __init__(
        self,
        lam: float = 0.1,
    ):

        super().__init__()

        self.lam = lam

    def fit(
        self,
        G: np.ndarray,
        d: np.ndarray,
    ):
        """
        Perform Tikhonov inversion.
        """

        GTG = G.T @ G

        I = np.eye(
            GTG.shape[0]
        )

        lambda_ = self.lam * np.max(np.diag(GTG))     # scale λ to the system, matching the notebook

        self.model = np.linalg.solve(
            GTG + lambda_ * I,
            G.T @ d,
        )

        return self.model

    def objective(
        self,
        G,
        d,
    ):
        """
        Compute objective function.
        """

        residual = G @ self.model - d

        data_term = np.linalg.norm(
            residual
        ) ** 2

        model_term = (
            self.lam**2
            * np.linalg.norm(
                self.model
            ) ** 2
        )

        return data_term + model_term