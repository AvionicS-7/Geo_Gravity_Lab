"""
generate_figures.py
-------------------

Generate publication-quality figures for GeoGravityLab.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt

from examples.interactive_model import build_geological_model

from src.domain import create_domain
from src.geological_models import GeologicalModelBuilder
from src.pipeline import GravityPipeline
from src.visualization import GravityVisualizer


FIGURE_DIR = "reports/figures"


def ensure_directory():

    os.makedirs(FIGURE_DIR, exist_ok=True)


def save(name):

    plt.savefig(
        os.path.join(FIGURE_DIR, name),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def main():

    ensure_directory()

    print("=" * 60)
    print("Generating GeoGravityLab Figures")
    print("=" * 60)

    model = build_geological_model()

    x, z, X, Z, x_obs, dx, dz = create_domain()

    builder = GeologicalModelBuilder(model)

    density = builder.generate_density_grid(
        X,
        Z,
    )

    density_vector = density.ravel()

    pipeline = GravityPipeline()

    pipeline.build_forward_model(
        x_obs,
        X,
        Z,
        dx,
        dz,
    )

    gravity = pipeline.simulate_gravity(
        density_vector
    )

    gravity_noisy = pipeline.add_noise(
        percentage=5,
    )

    frequency, spectrum = pipeline.fft(
        dx,
    )

    filtered = pipeline.lowpass(
        cutoff=0.05,
    )

    depth = Z.ravel()

    tik = pipeline.invert_tikhonov()

    depth_model = pipeline.invert_depth(
        depth,
    )

    irls = pipeline.invert_irls(
        depth,
    )

    # --------------------------------------------
    # Density Model
    # --------------------------------------------

    GravityVisualizer.plot_density_model(
        density,
        title="Density Model",
    )

    save("01_density_model.png")

    # --------------------------------------------
    # Gravity
    # --------------------------------------------

    GravityVisualizer.plot_gravity(
        x_obs,
        gravity,
        title="Gravity Anomaly",
    )

    save("02_gravity.png")

    # --------------------------------------------
    # FFT
    # --------------------------------------------

    plt.figure(figsize=(10,4))

    plt.plot(
        frequency,
        np.abs(spectrum),
    )

    plt.grid(True)

    plt.title("FFT Spectrum")

    plt.xlabel("Frequency")

    plt.ylabel("Amplitude")

    save("03_fft.png")

    # --------------------------------------------
    # Filtered Signal
    # --------------------------------------------

    GravityVisualizer.compare_gravity(
        x_obs,
        gravity_noisy,
        filtered,
    )

    save("04_filtered.png")

    # --------------------------------------------
    # Inversion Comparison
    # --------------------------------------------

    GravityVisualizer.compare_all_methods(
        density,
        tik.reshape(density.shape),
        depth_model.reshape(density.shape),
        irls.reshape(density.shape),
    )

    save("05_inversion_comparison.png")

    # --------------------------------------------
    # Convergence (Dummy if unavailable)
    # --------------------------------------------

    if "irls_rmse" in pipeline.results:

        GravityVisualizer.plot_convergence(
            pipeline.results["irls_rmse"]
        )

        save("06_convergence.png")

    print()

    print("=" * 60)
    print("Figures saved successfully.")
    print(FIGURE_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()