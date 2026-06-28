"""
config.py
---------

Global configuration for GeoGravityLab.

This module centralizes all configurable project parameters such as
grid dimensions, observation geometry, inversion settings and paths.

Changing values here automatically updates the entire project.
"""

from pathlib import Path

# =============================================================================
# Project Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"

REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
TABLE_DIR = REPORT_DIR / "tables"

RESULT_DIR = PROJECT_ROOT / "results"

# =============================================================================
# Grid Parameters
# =============================================================================

NX = 80
NZ = 40

DX = 100.0  # meters
DZ = 100.0  # meters

# =============================================================================
# Observation Parameters
# =============================================================================

OBSERVATION_HEIGHT = 0.0

N_OBSERVATIONS = NX

# =============================================================================
# Default Noise
# =============================================================================

DEFAULT_NOISE_STD = 0.03

RANDOM_SEED = 42

# =============================================================================
# Inversion Defaults
# =============================================================================

DEFAULT_LAMBDA = 0.1

DEFAULT_IRLS_ITERATIONS = 15

DEFAULT_TOLERANCE = 1e-6

# =============================================================================
# Visualization
# =============================================================================

FIGURE_DPI = 300

DEFAULT_CMAP = "viridis"

SAVE_FIGURES = True