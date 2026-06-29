from .base import BaseInversion
from .tikhonov import TikhonovInversion
from .depth_weighted import DepthWeightedInversion
from .irls import IRLSInversion

__all__ = [
    "BaseInversion",
    "TikhonovInversion",
    "DepthWeightedInversion",
    "IRLSInversion",
]