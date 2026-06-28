"""
body.py
-------

Defines the Body class used to represent a rectangular subsurface
geological body for gravity forward modelling.

Author: PHEONIX
Project: GeoGravityLab
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(slots=True)
class Body:
    """
    Represents a single rectangular geological body.

    Parameters
    ----------
    x_center : float
        Horizontal center (m)

    z_center : float
        Vertical center/depth (m)

    width : float
        Width of body (m)

    height : float
        Height of body (m)

    density : float
        Density contrast (kg/m³)

    name : str
        Optional body name.
    """

    x_center: float
    z_center: float
    width: float
    height: float
    density: float
    name: str = "Body"

    def __post_init__(self) -> None:

        if self.width <= 0:
            raise ValueError("Width must be positive.")

        if self.height <= 0:
            raise ValueError("Height must be positive.")

    @property
    def left(self) -> float:
        return self.x_center - self.width / 2

    @property
    def right(self) -> float:
        return self.x_center + self.width / 2

    @property
    def top(self) -> float:
        return self.z_center - self.height / 2

    @property
    def bottom(self) -> float:
        return self.z_center + self.height / 2

    @property
    def area(self) -> float:
        return self.width * self.height

    def contains_point(self, x: float, z: float) -> bool:
        """Return True if (x,z) lies inside the body."""
        return (
            self.left <= x <= self.right
            and self.top <= z <= self.bottom
        )

    def bounds(self) -> tuple[float, float, float, float]:
        """Return (left, right, top, bottom)."""
        return self.left, self.right, self.top, self.bottom

    def to_dict(self) -> Dict:
        """Convert Body to dictionary."""
        return {
            "name": self.name,
            "x_center": self.x_center,
            "z_center": self.z_center,
            "width": self.width,
            "height": self.height,
            "density": self.density,
        }

    def __repr__(self) -> str:

        return (
            f"Body(name='{self.name}', "
            f"density={self.density}, "
            f"center=({self.x_center}, {self.z_center}))"
        )