"""
utils.py
--------

Utility functions used throughout GeoGravityLab.

These functions are shared across forward modelling,
inversion, spectral analysis and visualization.

Author
------
PHEONIX

Project
-------
GeoGravityLab
"""

from __future__ import annotations

import numpy as np


def create_grid(
    nx: int,
    nz: int,
    dx: float,
    dz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a regular computational grid.

    Parameters
    ----------
    nx : int
        Number of cells along x.

    nz : int
        Number of cells along z.

    dx : float
        Cell width.

    dz : float
        Cell height.

    Returns
    -------
    X, Z : ndarray
        Meshgrid of cell centres.
    """

    x = np.arange(nx) * dx + dx / 2
    z = np.arange(nz) * dz + dz / 2

    return np.meshgrid(x, z)


def create_observation_points(
    x_min: float,
    x_max: float,
    n_points: int,
) -> np.ndarray:
    """
    Generate equally spaced observation locations.
    """

    return np.linspace(x_min, x_max, n_points)


def flatten_model(
    model: np.ndarray,
) -> np.ndarray:
    """
    Flatten a 2D density model into a vector.
    """

    return model.ravel()


def reshape_model(
    vector: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    """
    Convert a vector back to a 2D model.
    """

    return vector.reshape(shape)


def rmse(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """
    Root Mean Square Error.
    """

    return np.sqrt(np.mean((observed - predicted) ** 2))


def relative_error(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """
    Relative L2 error.
    """

    return np.linalg.norm(observed - predicted) / np.linalg.norm(observed)


def normalize(
    data: np.ndarray,
) -> np.ndarray:
    """
    Normalize array into [0,1].
    """

    minimum = np.min(data)
    maximum = np.max(data)

    return (data - minimum) / (maximum - minimum)


def add_noise(
    signal: np.ndarray,
    std: float,
    seed: int | None = None,
) -> np.ndarray:
    """
    Add Gaussian noise.
    """

    rng = np.random.default_rng(seed)

    noise = rng.normal(
        0.0,
        std,
        signal.shape,
    )

    return signal + noise