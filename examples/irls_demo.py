import numpy as np

from src.inversion import IRLSInversion

np.random.seed(42)

# Sensitivity Matrix
G = np.random.randn(60, 25)

# Sparse True Model
true_model = np.zeros(25)

true_model[8:13] = 2.5

# Synthetic Data
d = G @ true_model

# Add Noise
d += 0.05 * np.random.randn(60)

# Depth Weight
depth = np.linspace(100, 2500, 25)

solver = IRLSInversion(
    n_iter=20,
    verbose=True,
)

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