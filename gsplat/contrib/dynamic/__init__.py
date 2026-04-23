"""Deformable / 4D Gaussian Splatting (experimental).

Ported from G-SHARP v0.2's ``training/scene`` package. See the proposal at
``docs/source/proposals/gsharp_v0_2_port.rst`` for motivation and the tracked
branch ``vnath_gsharp`` for progress.
"""

from .deformation import DeformationTable, DeformNetwork
from .hexplane import HexPlaneField
from .regulation import plane_smoothness, time_l1, time_smoothness
from .strategy import DynamicStrategy

__all__ = [
    "DeformationTable",
    "DeformNetwork",
    "DynamicStrategy",
    "HexPlaneField",
    "plane_smoothness",
    "time_l1",
    "time_smoothness",
]
