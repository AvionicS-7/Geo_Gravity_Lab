"""
geological_model.py

Container for multiple geological bodies.

Supports
---------
• Single body
• Two body
• Three body
• N-body
"""

from __future__ import annotations

from typing import Iterator

from .body import Body


class GeologicalModel:
    """Collection of geological bodies."""

    def __init__(self, name: str = "Geological Model"):

        self.name = name
        self.bodies: list[Body] = []

    def add_body(self, body: Body) -> None:
        self.bodies.append(body)

    def remove_body(self, index: int) -> None:
        del self.bodies[index]

    def clear(self) -> None:
        self.bodies.clear()
    def total_density(self):

        return sum(
            body.density
            for body in self.bodies
        )

    @property
    def n_bodies(self) -> int:
        return len(self.bodies)

    def __len__(self) -> int:
        return len(self.bodies)

    def __iter__(self) -> Iterator[Body]:
        return iter(self.bodies)

    def summary(self) -> None:

        print("=" * 60)
        print(self.name)
        print("=" * 60)

        for i, body in enumerate(self.bodies, start=1):

            print(f"\nBody {i}")
            print(body)

    def to_dict(self):

        return {
            "name": self.name,
            "n_bodies": self.n_bodies,
            "bodies": [b.to_dict() for b in self.bodies],
        }

    def __repr__(self):

        return (
            f"GeologicalModel("
            f"name='{self.name}', "
            f"n_bodies={self.n_bodies})"
        )