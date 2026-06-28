import numpy as np
import matplotlib.pyplot as plt

from src.signal_processing import SpectralAnalyzer
x = np.linspace(
    0,
    100,
    256,
)

signal = (
    np.sin(2*np.pi*x/20)
    +
    0.5*np.sin(2*np.pi*x/5)
)

spectral = SpectralAnalyzer()

frequency, fft = spectral.compute_fft(
    signal,
    dx=x[1]-x[0],
)

power = spectral.power_spectrum()

spectral.summary()

plt.figure(figsize=(12,4))

plt.plot(
    frequency,
    power,
)

plt.xlabel("Frequency")

plt.ylabel("Power")

plt.title("Power Spectrum")

plt.grid(True)

plt.show()