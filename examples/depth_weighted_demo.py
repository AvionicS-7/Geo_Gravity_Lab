import numpy as np

from src.inversion import DepthWeightedInversion

np.random.seed(42)

# Sensitivity Matrix
G = np.random.randn(50, 20)

# True Model
true_model = np.random.randn(20)

# Synthetic Data
d = G @ true_model

# Add Noise
d += 0.05 * np.random.randn(50)

# Depth of each cell
depth = np.linspace(100, 2000, 20)

solver = DepthWeightedInversion()

model = solver.fit(
    G,
    d,
    depth,
)

solver.compute_residual(
    G,
    d,
)

solver.summary()

print("\nRecovered Model Shape :", model.shape)