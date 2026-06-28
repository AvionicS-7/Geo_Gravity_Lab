import numpy as np

from src.forward_model import ForwardModel

solver = ForwardModel()

NX = 20
NZ = 10

dx = 100
dz = 100

x = np.linspace(0, 2000, NX)

z = np.linspace(100, 1000, NZ)

X, Z = np.meshgrid(x, z)

x_obs = np.linspace(0, 2000, 50)

G = solver.build_sensitivity_matrix(
    x_obs,
    X,
    Z,
    dx,
    dz,
)

density = np.ones(X.size)

gravity = solver.forward(density)

solver.summary()

print()

print("Gravity vector shape :", gravity.shape)