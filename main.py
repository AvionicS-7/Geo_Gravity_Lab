"""
main.py

Entry point for GeoGravityLab.
"""

from src.models import Body
from src.models import GeologicalModel


def main():

    model = GeologicalModel("Demo Model")

    model.add_body(
        Body(
            x_center=2500,
            z_center=700,
            width=800,
            height=500,
            density=450,
            name="Body 1",
        )
    )

    model.add_body(
        Body(
            x_center=5200,
            z_center=1200,
            width=1200,
            height=600,
            density=-300,
            name="Body 2",
        )
    )

    print(model)
    model.summary()


if __name__ == "__main__":
    main()