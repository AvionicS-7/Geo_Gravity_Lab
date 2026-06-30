"""
interactive_model.py
--------------------

Interactive Geological Model Builder for GeoGravityLab.
"""

from __future__ import annotations

from src.models import Body
from src.models import GeologicalModel


def build_geological_model() -> GeologicalModel:
    """
    Build a GeologicalModel interactively.
    """

    print("=" * 60)
    print("GeoGravityLab Interactive Geological Model Builder")
    print("=" * 60)

    model_name = input(
        "\nModel Name : "
    )

    model = GeologicalModel(model_name)

    n_bodies = int(
        input(
            "\nNumber of Geological Bodies : "
        )
    )

    print()

    for i in range(n_bodies):

        print("=" * 40)
        print(f"BODY {i + 1}")
        print("=" * 40)

        name = input("Body Name : ")

        x_center = float(
            input("X Center (m) : ")
        )

        z_center = float(
            input("Depth (m) : ")
        )

        width = float(
            input("Width (m) : ")
        )

        height = float(
            input("Height (m) : ")
        )

        density = float(
            input(
                "Density Contrast (kg/m³) : "
            )
        )

        model.add_body(

            Body(
                x_center=x_center,
                z_center=z_center,
                width=width,
                height=height,
                density=density,
                name=name,
            )

        )

        print()

    return model