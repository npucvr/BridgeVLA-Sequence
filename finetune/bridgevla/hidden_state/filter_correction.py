"""Filter-style token correction for the frozen BridgeVLA token route.

This module implements the design in
``docs/bridgevla_filter_token_correction_design.md``: the token residual is
constrained to be a measurement-space Bayes/Kalman update,

    Delta z_t = C_theta K_t e_t,      e_t = z_t - C_theta mu_t^-,

so the correction direction comes from the innovation and the magnitude comes
from a gain derived from learned uncertainty.  Only the filter model
(``C_theta``, ``A_theta``, ``Q_theta``, ``R_theta``) is trainable; the
correction function itself is not learned.

Two modules are provided:

* :class:`MeasurementProjection` -- the frozen measurement map ``P`` together
  with its exact adjoint ``P^T``.  ``P`` sums each camera view onto a coarse
  spatial grid and applies a fixed semi-orthogonal projection, so the
  measurement keeps a spatial layout and the gain has somewhere to act.
* :class:`FilterCorrection` -- the trainable filter model plus the Kalman
  update and the token-space correction.

The update is evaluated in information form.  Because the measurement noise is
diagonal, the posterior has a closed form that never materialises the ``m x m``
innovation covariance ``S``:

    Sigma^+ = (Sigma^-^{-1} + C^T R^{-1} C)^{-1}
    mu^+    = mu^- + Sigma^+ C^T R^{-1} e
    log det S = sum(log r) + log det(Sigma^-) + log det(Sigma^+^{-1})

Everything is ``O(m d + d^3)``.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

# Guards the ``d x d`` inversions against a singular prior covariance.
_COVARIANCE_JITTER = 1e-6


def _symmetrize(matrix):
    """Return the symmetric part of a batch of square matrices."""
    return 0.5 * (matrix + matrix.transpose(-1, -2))


class MeasurementProjection(nn.Module):
    """Frozen measurement map ``P`` and its adjoint.

    Args:
        token_dim: Width ``D`` of each PaliGemma token.
        num_views: Number of rendered camera views ``V``.
        patch_grid: Side length of the per-view patch grid; the token layout is
            ``patch_grid x patch_grid`` per view.
        grid: Side length ``G`` of the coarse pooling grid.  Must divide
            ``patch_grid``.
        measure_dim: Width ``D_z`` of the per-cell measurement subspace.
        seed: Seed for the frozen random projection.

    The measurement has size ``m = V * G * G * D_z``.  ``P`` is random but
    frozen: leaving it trainable together with ``C_theta`` would create a scale
    degeneracy (``C -> CS``, ``P -> S^{-1} P``).
    """

    def __init__(
        self,
        token_dim=2048,
        num_views=2,
        patch_grid=16,
        grid=4,
        measure_dim=64,
        seed=0,
    ):
        super().__init__()
        if int(token_dim) < 1:
            raise ValueError(f"token_dim must be positive, got {token_dim}")
        if int(num_views) < 1:
            raise ValueError(f"num_views must be positive, got {num_views}")
        if int(patch_grid) < 1:
            raise ValueError(f"patch_grid must be positive, got {patch_grid}")
        if int(grid) < 1:
            raise ValueError(f"grid must be positive, got {grid}")
        if int(measure_dim) < 1:
            raise ValueError(f"measure_dim must be positive, got {measure_dim}")
        if int(patch_grid) % int(grid) != 0:
            raise ValueError(
                "patch_grid must be divisible by grid: "
                f"patch_grid={patch_grid}, grid={grid}"
            )

        self.token_dim = int(token_dim)
        self.num_views = int(num_views)
        self.patch_grid = int(patch_grid)
        self.grid = int(grid)
        self.measure_dim = int(measure_dim)

        self.tokens_per_view = self.patch_grid * self.patch_grid
        self.cell_tokens = (self.patch_grid // self.grid) ** 2
        self.num_cells = self.num_views * self.grid * self.grid
        self.measure_size = self.num_cells * self.measure_dim

        generator = torch.Generator().manual_seed(int(seed))
        if self.measure_dim > self.token_dim:
            raise ValueError(
                "measure_dim must not exceed token_dim for a semi-orthogonal "
                f"projection: measure_dim={self.measure_dim}, "
                f"token_dim={self.token_dim}"
            )
        random_basis = torch.randn(
            self.token_dim, self.measure_dim, generator=generator
        )
        # A Gaussian matrix with std=1/sqrt(token_dim) has the right expected
        # entry scale but still introduces a random singular-value scale.  QR
        # makes the retained measurement subspace explicitly orthonormal:
        # projection.T @ projection = I.  This removes an avoidable gain
        # attenuation while keeping P frozen and deterministic.
        projection = torch.linalg.qr(random_basis, mode="reduced").Q
        # A buffer, never a parameter: frozen by construction.
        self.register_buffer("projection", projection)

    def _cell_sums(self, tokens):
        """Return per-cell summed tokens with shape ``[B, V, G, G, D]``."""
        batch_size = tokens.shape[0]
        cells = tokens.reshape(
            batch_size,
            self.num_views,
            self.grid,
            self.patch_grid // self.grid,
            self.grid,
            self.patch_grid // self.grid,
            self.token_dim,
        )
        # The previous mean pooling and the corresponding 1/cell_tokens lift
        # made the token correction smaller by another 1/cell_tokens factor.
        # Sum pooling keeps the same spatial layout but gives the fixed map a
        # calibrated scale; ``lift`` below is its exact adjoint.
        return cells.sum(dim=(3, 5))

    def forward(self, tokens):
        """Map ``[B, V*S, D]`` visual tokens to the measurement ``[B, m]``."""
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
            raise ValueError(
                "tokens must be a tensor with shape [B, V*S, D], got "
                f"{tuple(tokens.shape) if isinstance(tokens, torch.Tensor) else type(tokens).__name__}"
            )
        expected = self.num_views * self.tokens_per_view
        if tokens.shape[1] != expected:
            raise ValueError(
                f"expected {expected} visual tokens, got {tokens.shape[1]}"
            )
        if tokens.shape[-1] != self.token_dim:
            raise ValueError(
                f"expected token dim {self.token_dim}, got {tokens.shape[-1]}"
            )

        pooled = self._cell_sums(tokens.to(self.projection.dtype)).reshape(
            -1, self.token_dim
        )
        projected = pooled @ self.projection
        return projected.reshape(tokens.shape[0], self.measure_size)

    def lift(self, delta_measurement):
        """Apply the adjoint ``P^T`` to a measurement-space correction.

        The forward map uses a cell sum, so its adjoint applies the projected
        correction equally to every token in that cell.  There is deliberately
        no division by ``cell_tokens``: that division would be correct for a
        mean-pooling map, but would reintroduce the scale collapse this route
        is designed to avoid.
        """
        if not isinstance(delta_measurement, torch.Tensor):
            raise ValueError("delta_measurement must be a tensor")
        if delta_measurement.ndim != 2:
            raise ValueError(
                "delta_measurement must have shape [B, m], got "
                f"{tuple(delta_measurement.shape)}"
            )
        if delta_measurement.shape[-1] != self.measure_size:
            raise ValueError(
                f"expected measurement size {self.measure_size}, "
                f"got {delta_measurement.shape[-1]}"
            )

        batch_size = delta_measurement.shape[0]
        cells = delta_measurement.reshape(
            batch_size, self.num_views, self.grid, self.grid, self.measure_dim
        )
        lifted = cells @ self.projection.to(cells.dtype).transpose(0, 1)
        lifted = lifted.reshape(
            batch_size, self.num_views, self.grid, 1, self.grid, 1, self.token_dim
        )
        expanded = lifted.expand(
            batch_size,
            self.num_views,
            self.grid,
            self.patch_grid // self.grid,
            self.grid,
            self.patch_grid // self.grid,
            self.token_dim,
        )
        return expanded.reshape(
            batch_size, self.num_views * self.tokens_per_view, self.token_dim
        )


class FilterCorrection(nn.Module):
    """Trainable Kalman filter model and the token residual it produces.

    Args:
        token_dim: Width ``D`` of each PaliGemma token.
        num_views: Number of rendered camera views ``V``.
        hidden_state_dim: Width ``d`` of the latent belief state.
        measure_dim: Width ``D_z`` of the per-cell measurement subspace.
        grid: Side length ``G`` of the coarse pooling grid.
        patch_grid: Side length of the per-view patch grid.
        full_covariance: Keep the full ``d x d`` posterior covariance; when
            ``False`` only its diagonal is retained.
        init_log_process_noise: Initial log diagonal process noise.
        init_log_measure_noise: Initial log diagonal measurement noise.
        init_state_log_scale: Initial log prior state standard deviation.
        seed: Seed for the frozen measurement projection.

    ``alpha`` is zero-initialised, so the corrected tokens equal the input
    tokens at the start of fine-tuning and the released BridgeVLA behaviour is
    preserved exactly.
    """

    def __init__(
        self,
        token_dim=2048,
        num_views=2,
        hidden_state_dim=64,
        measure_dim=64,
        grid=4,
        patch_grid=16,
        full_covariance=True,
        init_log_process_noise=-4.6,
        # This is the inverse-softplus parameter.  At this scale
        # softplus(48) ~= 48, which keeps the first posterior update from
        # collapsing when m/d is large (the default route has m=3072,d=64).
        init_log_measure_noise=48.0,
        init_state_log_scale=0.0,
        seed=0,
    ):
        super().__init__()
        if int(hidden_state_dim) < 1:
            raise ValueError(
                f"hidden_state_dim must be positive, got {hidden_state_dim}"
            )

        self.token_dim = int(token_dim)
        self.num_views = int(num_views)
        self.hidden_state_dim = int(hidden_state_dim)
        self.full_covariance = bool(full_covariance)

        self.measurement = MeasurementProjection(
            token_dim=self.token_dim,
            num_views=self.num_views,
            patch_grid=int(patch_grid),
            grid=int(grid),
            measure_dim=int(measure_dim),
            seed=seed,
        )
        self.measure_size = self.measurement.measure_size

        # Linear measurement model: z_hat = C_theta mu^-.
        self.measurement_matrix = nn.Linear(
            self.hidden_state_dim, self.measure_size, bias=False
        )
        nn.init.normal_(
            self.measurement_matrix.weight,
            std=1.0 / math.sqrt(self.hidden_state_dim),
        )

        # Covariance prediction.  Identity keeps the prior covariance stable at
        # initialisation instead of contracting or exploding it.
        self.covariance_transition = nn.Parameter(torch.eye(self.hidden_state_dim))
        self.log_process_noise = nn.Parameter(
            torch.full((self.hidden_state_dim,), float(init_log_process_noise))
        )
        self.log_measure_noise = nn.Parameter(
            torch.full((self.measure_size,), float(init_log_measure_noise))
        )
        self.log_state_scale = nn.Parameter(
            torch.full((self.hidden_state_dim,), float(init_state_log_scale))
        )

        # Zero-initialised residual gate: a strict no-op at the start.
        self.alpha = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------------
    # filter model
    # ------------------------------------------------------------------
    @property
    def measure_size_(self):
        return self.measure_size

    def process_noise(self):
        """Return the diagonal process noise ``diag(Q_theta)``, shape ``[d]``."""
        return F.softplus(self.log_process_noise) + 1e-6

    def measure_noise(self):
        """Return the diagonal measurement noise ``diag(R_theta)``, ``[m]``."""
        return F.softplus(self.log_measure_noise) + 1e-6

    def initial_state(self, batch_size, device=None, dtype=None):
        """Return the zero-mean prior belief at the start of an episode."""
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        parameter = next(self.parameters())
        device = parameter.device if device is None else device
        dtype = parameter.dtype if dtype is None else dtype
        scale = torch.exp(self.log_state_scale).to(device=device, dtype=dtype)
        mean = torch.zeros(
            batch_size, self.hidden_state_dim, device=device, dtype=dtype
        )
        variance = scale.square()
        if self.full_covariance:
            covariance = torch.diag(variance).unsqueeze(0).expand(
                batch_size, self.hidden_state_dim, self.hidden_state_dim
            ).contiguous()
        else:
            covariance = variance.unsqueeze(0).expand(
                batch_size, self.hidden_state_dim
            ).contiguous()
        return mean, covariance

    @property
    def state_size(self):
        """Flat width of the packed belief state.

        The sequence trainer carries one opaque ``[B, state_size]`` tensor per
        episode, so packing the covariance into that tensor keeps the existing
        state plumbing (row indexing, ``index_copy``, ``torch.where``,
        ``detach``) unchanged.
        """
        if self.full_covariance:
            return self.hidden_state_dim + self.hidden_state_dim ** 2
        return 2 * self.hidden_state_dim

    def pack_state(self, mean, covariance):
        """Flatten ``(mean, covariance)`` into ``[B, state_size]``."""
        if mean.ndim != 2 or mean.shape[-1] != self.hidden_state_dim:
            raise ValueError(
                "mean must have shape [B, hidden_state_dim], got "
                f"{tuple(mean.shape)}"
            )
        if self.full_covariance:
            flat_covariance = covariance.reshape(mean.shape[0], -1)
        else:
            flat_covariance = covariance
        return torch.cat([mean, flat_covariance], dim=-1)

    def unpack_state(self, packed):
        """Split ``[B, state_size]`` back into ``(mean, covariance)``."""
        if not isinstance(packed, torch.Tensor) or packed.ndim != 2:
            raise ValueError("packed state must be a tensor with shape [B, S]")
        if packed.shape[-1] != self.state_size:
            raise ValueError(
                f"packed state has the wrong width: expected {self.state_size}, "
                f"got {packed.shape[-1]}"
            )
        hidden_state_dim = self.hidden_state_dim
        mean = packed[:, :hidden_state_dim]
        rest = packed[:, hidden_state_dim:]
        if self.full_covariance:
            covariance = rest.reshape(
                packed.shape[0], hidden_state_dim, hidden_state_dim
            )
        else:
            covariance = rest
        return mean, covariance

    def predict_covariance(self, posterior_covariance):
        """Propagate the covariance through one control step.

        ``Sigma^- = A_theta Sigma^+ A_theta^T + Q_theta``.  The mean is
        propagated by ``F_phi`` outside this module, so only the covariance is
        handled here.
        """
        transition = self.covariance_transition
        noise = self.process_noise().to(transition.dtype)
        if self.full_covariance:
            propagated = transition @ posterior_covariance @ transition.transpose(0, 1)
            return _symmetrize(propagated) + torch.diag(noise)
        diagonal = transition.diagonal().square()
        return diagonal * posterior_covariance + noise

    # ------------------------------------------------------------------
    # Kalman update in information form
    # ------------------------------------------------------------------
    def update(self, prior_mean, prior_covariance, measurement):
        """Run one measurement update.

        Args:
            prior_mean: ``[B, d]`` action-predicted prior mean ``mu_t^-``.
            prior_covariance: ``[B, d, d]`` prior covariance, or ``[B, d]``
                diagonal when ``full_covariance=False``.
            measurement: ``[B, m]`` current measurement ``z_t``.

        Returns the posterior belief, the innovation ``e_t``, the
        measurement-space correction ``Delta z_t``, and the two scalar terms of
        the innovation likelihood, so the caller can form the negative log
        likelihood without materialising ``S``.
        """
        if not isinstance(prior_mean, torch.Tensor) or prior_mean.ndim != 2:
            raise ValueError("prior_mean must be a tensor with shape [B, d]")
        if prior_mean.shape[-1] != self.hidden_state_dim:
            raise ValueError(
                "prior_mean has the wrong width: "
                f"expected {self.hidden_state_dim}, got {prior_mean.shape[-1]}"
            )
        if not isinstance(measurement, torch.Tensor) or measurement.ndim != 2:
            raise ValueError("measurement must be a tensor with shape [B, m]")
        if measurement.shape[-1] != self.measure_size:
            raise ValueError(
                f"expected measurement size {self.measure_size}, "
                f"got {measurement.shape[-1]}"
            )
        if measurement.shape[0] != prior_mean.shape[0]:
            raise ValueError(
                "batch size mismatch between prior mean and measurement: "
                f"{prior_mean.shape[0]} vs {measurement.shape[0]}"
            )

        dtype = self.measurement_matrix.weight.dtype
        device = self.measurement_matrix.weight.device
        prior_mean = prior_mean.to(dtype=dtype)
        measurement = measurement.to(dtype=dtype)
        weight = self.measurement_matrix.weight  # C_theta, shape [m, d]

        # Predicted measurement and innovation e_t = z_t - C mu^-.
        innovation = measurement - self.measurement_matrix(prior_mean)

        measure_noise = self.measure_noise().to(dtype)
        inverse_measure_noise = 1.0 / measure_noise

        # Information matrix contribution C^T R^{-1} C, shape [d, d].
        information = weight.transpose(0, 1) @ (
            weight * inverse_measure_noise.unsqueeze(-1)
        )

        jitter = _COVARIANCE_JITTER * torch.eye(
            self.hidden_state_dim, device=device, dtype=dtype
        )
        if self.full_covariance:
            prior_covariance = _symmetrize(prior_covariance.to(dtype)) + jitter
            log_prior_covariance = torch.linalg.slogdet(prior_covariance)[1]
            posterior_information = _symmetrize(
                torch.linalg.inv(prior_covariance) + information
            )
            posterior_covariance = torch.linalg.inv(posterior_information)
            log_posterior_information = torch.linalg.slogdet(
                posterior_information
            )[1]
        else:
            prior_covariance = prior_covariance.to(dtype).clamp_min(
                _COVARIANCE_JITTER
            )
            log_prior_covariance = torch.log(prior_covariance).sum(-1)
            posterior_information = _symmetrize(
                torch.diag_embed(1.0 / prior_covariance) + information
            )
            posterior_diagonal = posterior_information.diagonal(
                dim1=-2, dim2=-1
            ).clamp_min(_COVARIANCE_JITTER)
            posterior_covariance = torch.diag_embed(1.0 / posterior_diagonal)
            # Diagonal assumed-density approximation: the determinant is taken
            # over the retained diagonal so the likelihood stays consistent
            # with the covariance actually used.
            log_posterior_information = torch.log(posterior_diagonal).sum(-1)

        # Mean correction: mu^+ - mu^- = Sigma^+ C^T R^{-1} e.
        weighted_innovation = inverse_measure_noise * innovation  # R^{-1} e
        projected = weighted_innovation @ weight  # C^T R^{-1} e
        mean_correction = torch.einsum(
            "bij,bj->bi", posterior_covariance, projected
        )
        posterior_mean = prior_mean + mean_correction

        # Measurement-space correction Delta z_t = C (mu^+ - mu^-).
        delta_measurement = self.measurement_matrix(mean_correction)

        # Innovation likelihood without forming S.
        quadratic = (inverse_measure_noise * innovation.square()).sum(-1)
        quadratic = quadratic - (projected * mean_correction).sum(-1)
        log_determinant = (
            torch.log(measure_noise).sum()
            + log_prior_covariance
            + log_posterior_information
        )

        return {
            "posterior_mean": posterior_mean,
            "posterior_covariance": posterior_covariance,
            "innovation": innovation,
            "delta_measurement": delta_measurement,
            "innovation_quadratic": quadratic,
            "innovation_log_determinant": log_determinant,
        }

    def innovation_nll(self, update_output):
        """Return the per-sample innovation negative log likelihood."""
        quadratic = update_output["innovation_quadratic"]
        log_determinant = update_output["innovation_log_determinant"]
        if torch.is_tensor(log_determinant) and log_determinant.ndim == 0:
            log_determinant = log_determinant.expand_as(quadratic)
        return 0.5 * (quadratic + log_determinant)

    # ------------------------------------------------------------------
    # token correction
    # ------------------------------------------------------------------
    def correct_tokens(self, tokens, delta_measurement):
        """Add ``alpha * B Delta z_t`` to the tokens.

        ``B`` is the frozen adjoint ``P^T``.  With ``alpha`` zero-initialised
        this returns the tokens unchanged.  The arithmetic is done in the
        trainable path's dtype (float32) so a small correction is not rounded
        away by bfloat16 tokens.
        """
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
            raise ValueError("tokens must be a tensor with shape [B, V*S, D]")
        delta_tokens = self.measurement.lift(delta_measurement)
        tokens = tokens.to(dtype=delta_tokens.dtype)
        return tokens + self.alpha.to(tokens.dtype) * delta_tokens

    def forward(self, tokens, prior_mean, prior_covariance):
        """Measure, filter, and correct the current visual tokens.

        Always performs the full measurement update and correction.  Whether the
        posterior is carried forward is the caller's decision: BridgeVLA's
        second (stage-two) pass refines the same observation in another view
        space, so it must correct its tokens from the carried belief without
        advancing the belief a second time.
        """
        measurement = self.measurement(tokens)
        filtered = self.update(prior_mean, prior_covariance, measurement)
        corrected = self.correct_tokens(tokens, filtered["delta_measurement"])
        filtered["measurement"] = measurement
        filtered["innovation_nll"] = self.innovation_nll(filtered)
        filtered.update(self.diagnostics(tokens, prior_mean, filtered))
        return corrected, filtered

    @torch.no_grad()
    def diagnostics(self, tokens, prior_mean, filtered):
        """Return the detached scalars required by the design document §10.3.

        These monitor the three ways the filter can silently stop working:
        a vanishing residual (the module is a no-op), a collapsing posterior
        trace (the gain becomes a constant), and a runaway measurement noise
        (the filter stops trusting any observation).
        """
        delta_tokens = self.measurement.lift(filtered["delta_measurement"])
        applied = self.alpha.detach().abs() * delta_tokens.norm()
        covariance = filtered["posterior_covariance"]
        if self.full_covariance:
            posterior_trace = covariance.diagonal(dim1=-2, dim2=-1).sum(-1)
        else:
            posterior_trace = covariance.sum(-1)
        return {
            "residual_ratio": applied / tokens.norm().clamp_min(1e-12),
            "posterior_trace": posterior_trace.mean(),
            "measure_noise_trace": self.measure_noise().sum(),
            "process_noise_trace": self.process_noise().sum(),
            "mean_correction_norm": (
                filtered["posterior_mean"] - prior_mean
            ).norm(dim=-1).mean(),
            "alpha": self.alpha.detach().abs().mean(),
        }
