"""
filters.py
----------

Frequency-domain filtering utilities.

Author
------
PHEONIX

Project
-------
GeoGravityLab
"""

from __future__ import annotations

import numpy as np


class FrequencyFilter:
    """
    Frequency-domain filters for gravity data.
    """

    @staticmethod
    def low_pass(
        fft_signal: np.ndarray,
        frequency: np.ndarray,
        cutoff: float,
    ) -> np.ndarray:
        """
        Low-pass filter.
        """

        filtered = fft_signal.copy()

        filtered[np.abs(frequency) > cutoff] = 0

        return filtered

    @staticmethod
    def high_pass(
        fft_signal: np.ndarray,
        frequency: np.ndarray,
        cutoff: float,
    ) -> np.ndarray:
        """
        High-pass filter.
        """

        filtered = fft_signal.copy()

        filtered[np.abs(frequency) < cutoff] = 0

        return filtered

    @staticmethod
    def band_pass(
        fft_signal: np.ndarray,
        frequency: np.ndarray,
        low_cut: float,
        high_cut: float,
    ) -> np.ndarray:
        """
        Band-pass filter.
        """

        filtered = fft_signal.copy()

        mask = (
            (np.abs(frequency) < low_cut)
            |
            (np.abs(frequency) > high_cut)
        )

        filtered[mask] = 0

        return filtered

    @staticmethod
    def gaussian(
        fft_signal: np.ndarray,
        frequency: np.ndarray,
        sigma: float,
    ) -> np.ndarray:
        """
        Gaussian filter.
        """

        gaussian = np.exp(
            -(frequency ** 2) / (2 * sigma ** 2)
        )

        return fft_signal * gaussian