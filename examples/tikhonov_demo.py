import numpy as np

from src.inversion import TikhonovInversion

np.random.seed(42)

# Sensitivity Matrix
G = np.random.randn(50, 20)

# True Model
true_model = np.random.randn(20)

# Synthetic Gravity Data
d = G @ true_model

# Add Noise
d += 0.05 * np.random.randn(50)

solver = TikhonovInversion(lam=0.1)

model = solver.fit(G, d)

solver.compute_residual(G, d)

solver.summary()

print("\nRecovered Model Shape :", model.shape)
print("First 5 Parameters :", model[:5])