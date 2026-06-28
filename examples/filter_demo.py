import matplotlib.pyplot as plt
import numpy as np

from src.signal_processing import (
    FrequencyFilter,
    SpectralAnalyzer,
)

x = np.linspace(0, 100, 512)

signal = (
    np.sin(2*np.pi*x/25)
    +
    0.3*np.sin(2*np.pi*x/5)
)

spectral = SpectralAnalyzer()

frequency, fft_signal = spectral.compute_fft(
    signal,
    dx=x[1]-x[0],
)

filtered_fft = FrequencyFilter.low_pass(
    fft_signal,
    frequency,
    cutoff=0.08,
)

filtered_signal = spectral.inverse_fft(
    filtered_fft
)

plt.figure(figsize=(12,5))

plt.plot(x, signal, label="Original")

plt.plot(
    x,
    filtered_signal,
    label="Filtered",
    linewidth=3,
)

plt.legend()

plt.grid(True)

plt.title("Low-pass Filtering")

plt.show()