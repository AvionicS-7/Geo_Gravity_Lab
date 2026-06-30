import numpy as np
def create_domain():
    x = np.linspace(0, 6000, 20)
    z = np.linspace(100, 3000, 10)

    X, Z = np.meshgrid(x, z)

    x_obs = np.linspace(0, 6000, 120)

    dx = x[1] - x[0]
    dz = z[1] - z[0]

    return x, z, X, Z, x_obs, dx, dz