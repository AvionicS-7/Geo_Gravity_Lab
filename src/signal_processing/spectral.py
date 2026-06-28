"""
spectral.py
-----------

Frequency-domain analysis utilities for gravity data.

Author
------
AvionicS-7

Project
-------
GeoGravityLab
"""

from __future__ import annotations

import numpy as np


class SpectralAnalyzer:
    """
    Frequency-domain analysis of gravity anomalies.
    """

    def __init__(self):

        self.signal = None
        self.frequency = None
        self.fft = None
        self.power = None

    def compute_fft(
        self,
        signal: np.ndarray,
        dx: float,
    ):
        """
        Compute FFT of gravity signal.

        Parameters
        ----------
        signal
            Gravity anomaly.

        dx
            Observation spacing.

        Returns
        -------
        frequency, fft
        """

        self.signal = signal

        self.fft = np.fft.fft(signal)

        self.frequency = np.fft.fftfreq(
            signal.size,
            d=dx,
        )

        return self.frequency, self.fft

    def power_spectrum(self):

        if self.fft is None:

            raise RuntimeError(
                "Run compute_fft() first."
            )

        self.power = np.abs(self.fft) ** 2

        return self.power

    def amplitude_spectrum(self):

        if self.fft is None:

            raise RuntimeError(
                "Run compute_fft() first."
            )

        return np.abs(self.fft)

    def phase_spectrum(self):

        if self.fft is None:

            raise RuntimeError(
                "Run compute_fft() first."
            )

        return np.angle(self.fft)

    def inverse_fft(
        self,
        spectrum: np.ndarray | None = None,
    ):

        if spectrum is None:

            spectrum = self.fft

        return np.real(
            np.fft.ifft(
                spectrum
            )
        )

    def dominant_frequency(self):

        if self.power is None:

            self.power_spectrum()

        index = np.argmax(self.power)

        return self.frequency[index]

    def summary(self):

        print("=" * 60)
        print("Spectral Analysis")
        print("=" * 60)

        print(
            f"Samples : {len(self.signal)}"
        )

        print(
            f"Dominant Frequency : {self.dominant_frequency():.5f}"
        )