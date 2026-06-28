"""
base.py
-------

Base class for all inversion algorithms.

Every inversion method in GeoGravityLab inherits from this class.

Author
------
PHEONIX
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod

import numpy as np


class BaseInversion(ABC):
    """
    Abstract base class for inversion algorithms.
    """

    def __init__(self):

        self.model = None
        self.predicted = None
        self.residual = None

    @abstractmethod
    def fit(
        self,
        G: np.ndarray,
        d: np.ndarray,
    ):
        """
        Estimate density model.
        """
        ...

    def predict(self):
        """
        Return recovered model.
        """

        return self.model

    def compute_residual(
        self,
        G: np.ndarray,
        d: np.ndarray,
    ):
        """
        Compute residual vector.
        """

        self.predicted = G @ self.model

        self.residual = d - self.predicted

        return self.residual

    def rmse(self):

        return np.sqrt(
            np.mean(
                self.residual ** 2
            )
        )

    def summary(self):

        print("=" * 60)

        print(type(self).__name__)

        print("=" * 60)

        if self.model is None:

            print("Model not estimated.")

            return

        print("Recovered Parameters :", len(self.model))

        if self.residual is not None:

            print("RMSE :", self.rmse())