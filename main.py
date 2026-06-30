"""
main.py
-------

Interactive entry point for GeoGravityLab.
"""

from __future__ import annotations

import numpy as np

from examples.interactive_model import build_geological_model

from src.domain import create_domain
from src.geological_models import GeologicalModelBuilder
from src.pipeline import GravityPipeline


def main():

    print("=" * 70)
    print("GeoGravityLab")
    print("Interactive Gravity Forward Modelling & Inversion")
    print("=" * 70)

    # --------------------------------------------------
    # Build Geological Model
    # --------------------------------------------------

    model = build_geological_model()

    print("\nModel Created Successfully.\n")

    model.summary()

    # --------------------------------------------------
    # Create Computational Domain
    # --------------------------------------------------

    x, z, X, Z, x_obs, dx, dz = create_domain()

    # --------------------------------------------------
    # Density Grid
    # --------------------------------------------------

    builder = GeologicalModelBuilder(model)

    density_grid = builder.generate_density_grid(
        X,
        Z,
    )

    density_vector = density_grid.ravel()

    print("\nDensity Grid Generated.")
    print("Grid Shape :", density_grid.shape)

    # --------------------------------------------------
    # Gravity Pipeline
    # --------------------------------------------------

    pipeline = GravityPipeline()

    print("\nBuilding Forward Model...")

    pipeline.build_forward_model(
        x_obs,
        X,
        Z,
        dx,
        dz,
    )

    print("Forward Model Ready.")

    # --------------------------------------------------
    # Forward Gravity
    # --------------------------------------------------

    gravity = pipeline.simulate_gravity(
        density_vector
    )

    print("Synthetic Gravity Generated.")

    # --------------------------------------------------
    # Noise
    # --------------------------------------------------

    gravity_noisy = pipeline.add_noise(
        percentage=5,
    )

    print("Noise Added.")

    # --------------------------------------------------
    # FFT
    # --------------------------------------------------

    dx_obs = x_obs[1] - x_obs[0]

    frequency, fft_signal = pipeline.fft(
        dx_obs,
    )

    print("FFT Complete.")

    # --------------------------------------------------
    # Filtering
    # --------------------------------------------------

    filtered = pipeline.lowpass(
        cutoff=0.05,
    )

    print("Filtering Complete.")

    # --------------------------------------------------
    # Depth Vector
    # --------------------------------------------------

    depth = Z.ravel()

    # --------------------------------------------------
    # Inversions
    # --------------------------------------------------

    print("\nRunning Tikhonov Inversion...")

    tik = pipeline.invert_tikhonov()

    print("Done.")

    print("\nRunning Depth Weighted Inversion...")

    depth_model = pipeline.invert_depth(
        depth,
    )

    print("Done.")

    print("\nRunning IRLS Inversion...")

    irls = pipeline.invert_irls(
        depth,
    )

    print("Done.")

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    pipeline.summary()

    print("\n" + "=" * 70)
    print("GeoGravityLab Finished Successfully")
    print("=" * 70)


if __name__ == "__main__":
    main()