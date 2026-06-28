import matplotlib.pyplot as plt

from src.models import Body
from src.models import GeologicalModel

from src.geological_models import GeologicalModelBuilder

from src.utils import create_grid

model = GeologicalModel("Three Body Model")

model.add_body(
    Body(
        x_center=2500,
        z_center=700,
        width=900,
        height=500,
        density=400,
    )
)

model.add_body(
    Body(
        x_center=5200,
        z_center=1200,
        width=1000,
        height=700,
        density=-250,
    )
)

model.add_body(
    Body(
        x_center=7200,
        z_center=1800,
        width=700,
        height=600,
        density=550,
    )
)

X, Z = create_grid(
    nx=100,
    nz=40,
    dx=100,
    dz=100,
)

builder = GeologicalModelBuilder(model)

density = builder.generate_density_grid(
    X,
    Z,
)

plt.figure(figsize=(10,4))

plt.imshow(
    density,
    origin="upper",
    aspect="auto",
)

plt.colorbar(label="Density Contrast")

plt.title("Generated Geological Model")

plt.show()