"""
noise.py
--------

Noise simulation utilities for GeoGravityLab.

Author
------
PHEONIX

Project
-------
GeoGravityLab
"""

from __future__ import annotations

import numpy as np


class NoiseSimulator:
    """
    Simulate noise in gravity observations.
    """

    def __init__(self, seed: int | None = None):

        self.rng = np.random.default_rng(seed)

    def gaussian(
        self,
        signal: np.ndarray,
        std: float,
    ) -> np.ndarray:
        """
        Add Gaussian noise.

        Parameters
        ----------
        signal
            Clean gravity anomaly.

        std
            Standard deviation.

        Returns
        -------
        ndarray
            Noisy signal.
        """

        noise = self.rng.normal(
            loc=0.0,
            scale=std,
            size=signal.shape,
        )

        return signal + noise

    def percentage(
        self,
        signal: np.ndarray,
        percentage: float,
    ) -> np.ndarray:
        """
        Add percentage Gaussian noise.

        Example
        -------
        percentage = 3

        => 3% Gaussian noise
        """

        std = percentage / 100 * np.max(np.abs(signal))

        return self.gaussian(
            signal,
            std,
        )

    def snr(
        self,
        clean: np.ndarray,
        noisy: np.ndarray,
    ) -> float:
        """
        Compute Signal-to-Noise Ratio (dB).
        """

        signal_power = np.mean(clean ** 2)

        noise_power = np.mean(
            (clean - noisy) ** 2
        )

        return 10 * np.log10(
            signal_power / noise_power
        )