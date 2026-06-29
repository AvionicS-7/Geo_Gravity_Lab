import numpy as np

from src.pipeline import GravityPipeline

pipeline = GravityPipeline()

NX = 40
NZ = 20

dx = 100
dz = 100

x = np.linspace(
    0,
    4000,
    NX,
)

z = np.linspace(
    100,
    2000,
    NZ,
)

X, Z = np.meshgrid(
    x,
    z,
)

obs = np.linspace(
    0,
    4000,
    80,
)

pipeline.build_forward_model(
    obs,
    X,
    Z,
    dx,
    dz,
)

density = np.random.randn(
    X.size,
)

pipeline.simulate_gravity(
    density,
)

pipeline.add_noise(
    percentage=3,
)

pipeline.fft(
    dx=obs[1]-obs[0],
)

pipeline.lowpass(
    cutoff=0.03,
)

depth = np.linspace(
    1,
    2,
    X.size,
)

pipeline.invert_tikhonov()

pipeline.invert_depth(
    depth,
)

pipeline.invert_irls(
    depth,
)

pipeline.summary()