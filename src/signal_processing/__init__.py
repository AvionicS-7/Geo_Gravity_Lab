"""
Signal processing package.

Contains all frequency-domain and noise processing utilities.
"""

from .noise import NoiseSimulator
from .spectral import SpectralAnalyzer
from .filters import FrequencyFilter

__all__ = [
    "NoiseSimulator",
    "SpectralAnalyzer",
    "FrequencyFilter",
]