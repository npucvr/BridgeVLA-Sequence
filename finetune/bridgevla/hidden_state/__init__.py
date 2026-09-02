"""Modules implementing the probabilistic hidden-state route.

The public names follow the model description:

* ``A_psi`` corrects the current observation tokens using the current hidden
  state ``y``.
* ``F_phi`` predicts the next state after an executed waypoint action.
* ``U_omega`` applies the current PaliGemma tokens as an observation update.
* ``ObservationDecoder`` predicts a detached current-observation feature from
  the pre-observation state ``y_t^-``.
"""

from .observation_decoder import ObservationDecoder, pool_visual_tokens
from .observation_update import U_omega
from .token_correction import A_psi
from .transition import F_phi

__all__ = [
    "A_psi",
    "F_phi",
    "ObservationDecoder",
    "U_omega",
    "pool_visual_tokens",
]
