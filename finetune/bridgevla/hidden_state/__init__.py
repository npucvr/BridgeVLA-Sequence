"""Modules implementing the filter-based hidden-state route.

``F_phi`` predicts the next belief state after an executed waypoint action.
``FilterCorrection`` uses the current visual tokens as a measurement and
produces the structured token residual consumed by the original BridgeVLA
action path.
"""

from .filter_correction import FilterCorrection, MeasurementProjection
from .transition import F_phi

__all__ = [
    "F_phi",
    "FilterCorrection",
    "MeasurementProjection",
]
