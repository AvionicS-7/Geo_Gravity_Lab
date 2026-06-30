"""
irls.py
-------

Iteratively Reweighted Least Squares (IRLS)
for compact gravity inversion.

Migrated from the GeoGravityLab notebook.

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
from .tikhonov import TikhonovInversion


class IRLSInversion(BaseInversion):
    """
    Compact gravity inversion using
    Iteratively Reweighted Least Squares (IRLS).
    """

    def __init__(
        self,
        n_iter: int = 30,
        epsilon: float = 1e-2,
        lambda_factor: float = 0.003,
        tol: float = 1e-6,
        verbose: bool = True,
    ):

        super().__init__()

        self.n_iter = n_iter
        self.epsilon = epsilon
        self.lambda_factor = lambda_factor
        self.tol = tol
        self.verbose = verbose

        self.rmse_history = []
        self.lambda_ = None

    def fit(
        self,
        G: np.ndarray,
        d: np.ndarray,
        w_depth: np.ndarray,
    ):
        """
        Run IRLS compact inversion.
        """

        GTG = G.T @ G
        GTd = G.T @ d

        self.lambda_ = (
            self.lambda_factor
            * np.max(np.diag(GTG))
        )

        # -------------------------------------------------
        # Warm Start using Tikhonov
        # -------------------------------------------------

        lambda_warm = 0.05 * np.max(np.diag(GTG))
        m_cur = np.linalg.solve(
            GTG + lambda_warm * np.diag(w_depth),
            GTd,
        )

        

        rmse_prev = None

        self.rmse_history.clear()

        # -------------------------------------------------
        # Main IRLS Loop
        # -------------------------------------------------

        for it in range(self.n_iter):

            # Compactness weights (Last & Kubik style)

            w_compact = 1.0 / np.sqrt(
                m_cur**2 + self.epsilon**2
            )

            # Prevent excessively large weights
            w_compact = np.clip(
                w_compact,
                0.1,
                20.0,
            )

            print(
                "Compact weights:",
                np.min(w_compact),
                np.max(w_compact)
            )

            w_total = np.sqrt(
                w_depth * w_compact
            )

            A = (
                GTG
                + self.lambda_
                * np.diag(w_total)
            )

            m_new = np.linalg.solve(
                A,
                GTd,
            )

            d_pred = G @ m_new

            rmse = np.sqrt(
                np.mean(
                    (d - d_pred) ** 2
                )
            )

            self.rmse_history.append(
                rmse
            )

            if self.verbose:

                print(
                    f"Iteration "
                    f"{it+1:02d}/"
                    f"{self.n_iter} "
                    f"RMSE = "
                    f"{rmse:.6f}"
                )

            if (
                rmse_prev is not None
                and abs(rmse - rmse_prev)
                < self.tol
            ):

                if self.verbose:

                    print(
                        "\nConverged."
                    )

                break

            rmse_prev = rmse

            m_cur = m_new

        self.model = m_cur

        return self.model

    def convergence_history(
        self,
    ):

        return self.rmse_history

    def summary(
        self,
    ):

        print("=" * 60)

        print(
            "IRLS Compact Inversion"
        )

        print("=" * 60)

        print(
            "Iterations :",
            len(self.rmse_history),
        )

        print(
            "Lambda :",
            self.lambda_,
        )

        print(
            "Final RMSE :",
            self.rmse_history[-1],
        )