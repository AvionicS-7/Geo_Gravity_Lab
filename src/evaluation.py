"""
evaluation.py
-------------

Evaluation metrics for gravity inversion.

Author
------
AvionicS-7

Project
-------
GeoGravityLab
"""

from __future__ import annotations

import numpy as np


class Evaluation:
    """
    Compute quantitative metrics for inversion quality.
    """

    @staticmethod
    def rmse(
        observed: np.ndarray,
        predicted: np.ndarray,
    ) -> float:
        """
        Root Mean Square Error.
        """
        return np.sqrt(
            np.mean(
                (observed - predicted) ** 2
            )
        )

    @staticmethod
    def mae(
        observed: np.ndarray,
        predicted: np.ndarray,
    ) -> float:
        """
        Mean Absolute Error.
        """
        return np.mean(
            np.abs(
                observed - predicted
            )
        )

    @staticmethod
    def mse(
        observed: np.ndarray,
        predicted: np.ndarray,
    ) -> float:
        """
        Mean Squared Error.
        """
        return np.mean(
            (observed - predicted) ** 2
        )

    @staticmethod
    def relative_error(
        observed: np.ndarray,
        predicted: np.ndarray,
    ) -> float:
        """
        Relative L2 error.
        """
        return (
            np.linalg.norm(
                observed - predicted
            )
            /
            np.linalg.norm(
                observed
            )
        )

    @staticmethod
    def residual(
        observed: np.ndarray,
        predicted: np.ndarray,
    ) -> np.ndarray:
        """
        Residual vector.
        """
        return observed - predicted

    @staticmethod
    def correlation(
        observed: np.ndarray,
        predicted: np.ndarray,
    ) -> float:
        """
        Pearson correlation coefficient.
        """
        return np.corrcoef(
            observed,
            predicted,
        )[0, 1]

    @staticmethod
    def r2_score(
        observed: np.ndarray,
        predicted: np.ndarray,
    ) -> float:
        """
        Coefficient of determination.
        """
        ss_res = np.sum(
            (observed - predicted) ** 2
        )

        ss_tot = np.sum(
            (
                observed
                - np.mean(observed)
            ) ** 2
        )

        return 1 - ss_res / ss_tot

    @classmethod
    def summary(
        cls,
        observed: np.ndarray,
        predicted: np.ndarray,
    ) -> dict:
        """
        Return all evaluation metrics.
        """

        metrics = {
            "RMSE": cls.rmse(
                observed,
                predicted,
            ),
            "MAE": cls.mae(
                observed,
                predicted,
            ),
            "MSE": cls.mse(
                observed,
                predicted,
            ),
            "Relative Error": cls.relative_error(
                observed,
                predicted,
            ),
            "Correlation": cls.correlation(
                observed,
                predicted,
            ),
            "R2 Score": cls.r2_score(
                observed,
                predicted,
            ),
        }

        print("=" * 60)
        print("Evaluation Summary")
        print("=" * 60)

        for key, value in metrics.items():
            print(f"{key:20s}: {value:.6f}")

        return metrics