"""Modules implementing the probabilistic hidden-state route.

The public names follow the model description:

* ``A_psi`` corrects the current observation tokens using the current hidden
  state ``y``.
* ``F_phi`` predicts the next state after an executed waypoint action.
* ``U_omega`` applies the current PaliGemma tokens as an observation update.
"""

from .observation_update import U_omega
from .token_correction import A_psi
from .transition import F_phi

__all__ = ["A_psi", "F_phi", "U_omega"]
