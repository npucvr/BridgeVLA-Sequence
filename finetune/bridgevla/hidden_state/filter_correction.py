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
        use_measure_adapter=False,
        measure_adapter_hidden=64,
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

        # Optional residual on z = P(pool); zero-init so it starts as a no-op.
        if use_measure_adapter:
            self.measure_adapter = MeasurementAdapter(
                token_dim=self.token_dim,
                measure_dim=int(measure_dim),
                hidden_dim=int(measure_adapter_hidden),
            )
        else:
            self.measure_adapter = None

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

        pooled = self._cell_sums(tokens.to(self.projection.dtype))
        projected = pooled.reshape(-1, self.token_dim) @ self.projection
        measurement = projected.reshape(tokens.shape[0], self.measure_size)
        if self.measure_adapter is not None:
            pooled_cells = pooled.reshape(tokens.shape[0], self.num_cells, self.token_dim)
            measurement = measurement + self.measure_adapter(pooled_cells)
        return measurement

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


class MeasurementAdapter(nn.Module):
    """Zero-init residual on the pooled measurement.

    Adds a small learned term to ``z = P(H)`` without making ``P`` itself
    trainable (which is degenerate with a free ``C_theta``). Each pooled cell
    ``[D]`` is mapped to a residual in its own measurement slice
    ``[measure_dim]``, so the stacked residual matches ``z``'s ``[B, m]``
    layout. The last layer is zero-initialised so the measurement stays
    exactly ``P(H)`` at the start.
    """

    def __init__(self, token_dim, measure_dim, hidden_dim=64):
        super().__init__()
        self.measure_dim = int(measure_dim)
        self.net = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.measure_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, pooled_tokens):
        """Map ``[B, num_cells, D]`` pooled tokens to a residual ``[B, m]``."""
        if pooled_tokens.ndim != 3:
            raise ValueError(
                "pooled_tokens must have shape [B, num_cells, D], got "
                f"{tuple(pooled_tokens.shape)}"
            )
        batch_size, num_cells, width = pooled_tokens.shape
        residual = self.net(pooled_tokens.reshape(batch_size * num_cells, width))
        return residual.reshape(batch_size, num_cells * self.measure_dim)


class NonlinearMeasurementResidual(nn.Module):
    """Low-rank residual that turns linear ``C`` into ``C_theta(mu) = W mu + V sigma(U mu)``.

    ``V`` is zero-initialised so ``C_theta`` starts as the linear map exactly.
    The Jacobian is formed analytically as ``J = W + V diag(sigma'(U mu)) U``
    and never via autograd over a full ``[B, m, d]`` map.
    """

    def __init__(self, hidden_dim, measure_size, rank=16):
        super().__init__()
        if int(rank) < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
        self.hidden_dim = int(hidden_dim)
        self.measure_size = int(measure_size)
        self.rank = int(rank)
        self.up = nn.Parameter(
            torch.randn(self.rank, self.hidden_dim) / math.sqrt(self.hidden_dim)
        )
        # Zero-init: residual is an exact no-op at the start of fine-tuning.
        self.down = nn.Parameter(torch.zeros(self.measure_size, self.rank))

    def forward(self, mean):
        """Return ``V relu(U mu)`` with shape ``[B, m]``."""
        pre = mean @ self.up.transpose(0, 1)  # [B, r]
        return F.relu(pre) @ self.down.transpose(0, 1)

    def jacobian(self, mean):
        """Return ``V diag(relu'(U mu)) U`` with shape ``[B, m, d]``.

        Built from the analytic low-rank factors (one batched einsum), not by
        differentiating a full measurement map.
        """
        pre = mean @ self.up.transpose(0, 1)  # [B, r]
        gate = (pre > 0).to(dtype=mean.dtype)  # relu'(pre)
        return torch.einsum(
            "mr,br,rd->bmd", self.down, gate, self.up
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
        per_cell_alpha: If ``True``, use one zero-init gate per spatial cell
            instead of a single global scalar.
        use_measure_adapter: If ``True``, add a zero-init residual adapter to
            the measurement so ``z`` can specialise beyond the frozen ``P``.
        nonlinear_measure: If ``True``, use ``C_theta(mu) = W mu + V relu(U mu)``
            with a zero-init low-rank residual ``V`` and its analytic Jacobian.
        measure_rank: Rank ``r`` of the nonlinear measurement residual.

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
        per_cell_alpha=True,
        use_measure_adapter=True,
        nonlinear_measure=True,
        measure_rank=16,
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
        self.per_cell_alpha = bool(per_cell_alpha)
        self.use_measure_adapter = bool(use_measure_adapter)
        self.nonlinear_measure = bool(nonlinear_measure)

        self.measurement = MeasurementProjection(
            token_dim=self.token_dim,
            num_views=self.num_views,
            patch_grid=int(patch_grid),
            grid=int(grid),
            measure_dim=int(measure_dim),
            seed=seed,
            use_measure_adapter=bool(use_measure_adapter),
            measure_adapter_hidden=max(64, int(hidden_state_dim)),
        )
        self.measure_size = self.measurement.measure_size

        # The residual adapter lives on MeasurementProjection so z = P(pool)
        # + adapter(pool) stays in one place; expose it here for checkpoints.

        # Linear measurement model: z_hat = C_theta mu^-.
        self.measurement_matrix = nn.Linear(
            self.hidden_state_dim, self.measure_size, bias=False
        )
        nn.init.normal_(
            self.measurement_matrix.weight,
            std=1.0 / math.sqrt(self.hidden_state_dim),
        )
        if self.nonlinear_measure:
            self.measure_residual = NonlinearMeasurementResidual(
                hidden_dim=self.hidden_state_dim,
                measure_size=self.measure_size,
                rank=int(measure_rank),
            )
        else:
            self.measure_residual = None

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
        # Per-cell gates let different spatial regions inject different
        # correction magnitudes while staying inside the filter residual form.
        if self.per_cell_alpha:
            self.alpha = nn.Parameter(torch.zeros(self.measurement.num_cells))
        else:
            self.alpha = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------------
    # filter model
    # ------------------------------------------------------------------
    @property
    def measure_size_(self):
        return self.measure_size

    @property
    def measure_adapter(self):
        """Residual measurement adapter (owned by ``measurement``)."""
        return self.measurement.measure_adapter

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
    # measurement model C_theta
    # ------------------------------------------------------------------
    def measure_model(self, mean):
        """Return ``C_theta(mu)`` with shape ``[B, m]``.

        Linear: ``W mu``.  Nonlinear: ``W mu + V relu(U mu)`` (zero-init ``V``).
        """
        predicted = self.measurement_matrix(mean)
        if self.measure_residual is not None:
            predicted = predicted + self.measure_residual(mean)
        return predicted

    def measure_jacobian(self, mean):
        """Return ``J = d C_theta / d mu`` at ``mean``, shape ``[B, m, d]``.

        Analytic form only: ``J = W + V diag(relu'(U mu)) U`` when the residual
        is enabled, else the constant ``W``.  Never built with autograd.
        """
        batch_size = mean.shape[0]
        weight = self.measurement_matrix.weight  # [m, d]
        if self.measure_residual is None:
            return weight.unsqueeze(0).expand(batch_size, -1, -1)
        return weight.unsqueeze(0) + self.measure_residual.jacobian(mean)

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

        # EKF linearisation at mu^-: predicted measurement C_theta(mu^-) and
        # Jacobian J = dC_theta/dmu (analytic; never an autograd [B, m, d]).
        predicted_measurement = self.measure_model(prior_mean)
        jacobian = self.measure_jacobian(prior_mean)  # [B, m, d]
        innovation = measurement - predicted_measurement

        measure_noise = self.measure_noise().to(dtype)
        inverse_measure_noise = 1.0 / measure_noise

        # Information matrix contribution J^T R^{-1} J, shape [B, d, d]
        # (per-sample once the residual is active; broadcasts when linear).
        weighted_jacobian = jacobian * inverse_measure_noise.unsqueeze(-1)
        information = torch.matmul(
            jacobian.transpose(1, 2), weighted_jacobian
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

        # Mean correction: mu^+ - mu^- = Sigma^+ J^T R^{-1} e.
        weighted_innovation = inverse_measure_noise * innovation  # R^{-1} e
        projected = torch.matmul(
            jacobian.transpose(1, 2), weighted_innovation.unsqueeze(-1)
        ).squeeze(-1)  # J^T R^{-1} e
        mean_correction = torch.einsum(
            "bij,bj->bi", posterior_covariance, projected
        )
        posterior_mean = prior_mean + mean_correction

        # Measurement-space correction Delta z_t = J (mu^+ - mu^-).
        # Equals C (mu^+ - mu^-) when the residual is inactive.
        delta_measurement = torch.matmul(
            jacobian, mean_correction.unsqueeze(-1)
        ).squeeze(-1)

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
    def _gate(self, dtype, device):
        """Return injection gates broadcastable to ``[B, num_cells, D_z]``."""
        gate = self.alpha.to(dtype=dtype, device=device)
        if self.per_cell_alpha:
            return gate.view(1, self.measurement.num_cells, 1)
        return gate.view(1, 1, 1)

    def correct_tokens(self, tokens, delta_measurement):
        """Add ``alpha * B Delta z_t`` to the tokens.

        ``B`` is the frozen adjoint ``P^T``.  With ``alpha`` zero-initialised
        this returns the tokens unchanged.  The arithmetic is done in the
        trainable path's dtype (float32) so a small correction is not rounded
        away by bfloat16 tokens.
        """
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
            raise ValueError("tokens must be a tensor with shape [B, V*S, D]")
        delta = delta_measurement
        if self.per_cell_alpha:
            delta = delta.reshape(
                delta.shape[0], self.measurement.num_cells, self.measurement.measure_dim
            )
            delta = delta * self._gate(delta.dtype, delta.device)
            delta = delta.reshape(delta.shape[0], self.measure_size)
        else:
            delta = delta * self.alpha.to(dtype=delta.dtype, device=delta.device)
        delta_tokens = self.measurement.lift(delta)
        tokens = tokens.to(dtype=delta_tokens.dtype)
        return tokens + delta_tokens

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
        if self.per_cell_alpha:
            delta = filtered["delta_measurement"].reshape(
                filtered["delta_measurement"].shape[0],
                self.measurement.num_cells,
                self.measurement.measure_dim,
            )
            gated = delta * self._gate(delta.dtype, delta.device).reshape(
                1, self.measurement.num_cells, 1
            )
            gated = gated.reshape(filtered["delta_measurement"].shape[0], -1)
        else:
            gated = filtered["delta_measurement"] * self.alpha.to(
                filtered["delta_measurement"].dtype
            )
        delta_tokens = self.measurement.lift(gated)
        applied = delta_tokens.norm()
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
