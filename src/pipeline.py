"""
pipeline.py
-----------

Main workflow pipeline for GeoGravityLab.

Author
------
PHEONIX

Project
-------
GeoGravityLab
"""

from __future__ import annotations

import numpy as np

from src.forward_model import ForwardModel
from src.signal_processing import (
    NoiseSimulator,
    SpectralAnalyzer,
    FrequencyFilter,
)
from src.inversion import (
    TikhonovInversion,
    DepthWeightedInversion,
    IRLSInversion,
)
from src.evaluation import Evaluation


class GravityPipeline:
    """
    Complete gravity modelling workflow.

    Workflow
    --------
    Geological Model
            ↓
    Forward Modelling
            ↓
    Noise Simulation
            ↓
    FFT
            ↓
    Filtering
            ↓
    Inversion
            ↓
    Evaluation
    """

    def __init__(self):

        self.forward = ForwardModel()

        self.noise = NoiseSimulator()

        self.spectral = SpectralAnalyzer()

        self.results = {}

    # -------------------------------------------------------

    def build_forward_model(
        self,
        x_obs,
        X,
        Z,
        dx,
        dz,
    ):

        self.forward.build_sensitivity_matrix(
            x_obs,
            X,
            Z,
            dx,
            dz,
        )

        return self

    # -------------------------------------------------------

    def simulate_gravity(
        self,
        density_vector,
    ):

        gravity = self.forward.forward(
            density_vector
        )

        self.results["gravity"] = gravity

        return gravity

    # -------------------------------------------------------

    def add_noise(
        self,
        percentage=5,
    ):

        noisy = self.noise.percentage(
            self.results["gravity"],
            percentage,
        )

        self.results["gravity_noisy"] = noisy

        return noisy

    # -------------------------------------------------------

    def fft(
        self,
        dx,
    ):

        freq, fft_signal = self.spectral.compute_fft(
            self.results["gravity_noisy"],
            dx,
        )

        self.results["frequency"] = freq

        self.results["fft"] = fft_signal

        return freq, fft_signal

    # -------------------------------------------------------

    def lowpass(
        self,
        cutoff,
    ):

        filtered = FrequencyFilter.low_pass(
            self.results["fft"],
            self.results["frequency"],
            cutoff,
        )

        gravity = self.spectral.inverse_fft(
            filtered,
        )

        self.results["filtered"] = gravity

        return gravity

    # -------------------------------------------------------

    def invert_tikhonov(
        self,
        lam=0.05,
    ):

        solver = TikhonovInversion(
            lam=lam,
        )

        model = solver.fit(
            self.forward.G,
            self.results["filtered"],
        )

        self.results["tikhonov"] = model

        return model

    # -------------------------------------------------------

    def invert_depth(
    self,
    depth,
    beta=0.5,
    lambda_factor=0.05,
    ):

        solver = DepthWeightedInversion(
        beta=beta,
        lambda_factor=lambda_factor,
        )

        model = solver.fit(
        self.forward.G,
        self.results["filtered"],
        depth,
    )

        self.results["depth"] = model

        return model

    # -------------------------------------------------------

    
    def invert_irls(
    self,
    depth,
    beta=0.5,
    ):

        reference_depth = np.max(depth)

        w_depth = (reference_depth / depth) ** beta

        solver = IRLSInversion()

        model = solver.fit(
            self.forward.G,
            self.results["filtered"],
            w_depth,
        )

        self.results["irls"] = model

        return model

    # -------------------------------------------------------

    def evaluate(
        self,
        observed,
        predicted,
    ):

        return Evaluation.summary(
            observed,
            predicted,
        )

    # -------------------------------------------------------

    def summary(
        self,
    ):

        print("=" * 60)

        print("GeoGravityLab Pipeline")

        print("=" * 60)

        for key in self.results:

            print("✔", key)