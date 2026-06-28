import matplotlib.pyplot as plt
import numpy as np

from src.signal_processing import NoiseSimulator

x = np.linspace(
    0,
    100,
    300,
)

signal = np.sin(
    x / 8
)

noise = NoiseSimulator(seed=42)

noisy = noise.percentage(
    signal,
    percentage=5,
)

print(
    "SNR:",
    noise.snr(
        signal,
        noisy,
    ),
)

plt.figure(figsize=(12,5))

plt.plot(
    x,
    signal,
    label="Original",
)

plt.plot(
    x,
    noisy,
    label="Noisy",
)

plt.legend()

plt.grid(True)

plt.title("Gaussian Noise Simulation")

plt.show()