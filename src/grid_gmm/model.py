"""Truncated Gaussian load model for a single appliance / household connection."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy.stats import truncnorm


@dataclass(frozen=True)
class GaussianComponent:
    """Active-power distribution of one connection while it is switched on.

    The power is drawn from ``N(mu, sigma)`` truncated to ``[0, p_max]``. ``mu`` is the
    *location* of the underlying (untruncated) normal distribution. Because of the
    truncation it is in general neither the mean nor the median of the drawn power;
    use :attr:`mean` for the expected active power.

    Instances are immutable so that the invariants checked in ``__post_init__`` always
    hold. Use :meth:`with_overrides` to derive a modified, re-validated copy.

    Attributes:
        p_max: Maximum (connection) power in kW, > 0.
        mu_ratio: Location ``mu`` as a ratio of ``p_max``, in [0, 1].
        sigma_ratio: Scale ``sigma`` as a ratio of ``p_max``, >= 0 (0 = deterministic load).
        name: Human-readable identifier.
    """

    p_max: float
    mu_ratio: float = 0.5
    sigma_ratio: float = 0.2
    name: str = "default_load"

    def __post_init__(self) -> None:
        if not math.isfinite(self.p_max) or self.p_max <= 0:
            raise ValueError(f"p_max must be positive, got {self.p_max}")
        if not math.isfinite(self.sigma_ratio) or self.sigma_ratio < 0:
            raise ValueError(f"sigma_ratio must be non-negative, got {self.sigma_ratio}")
        if not (0 <= self.mu_ratio <= 1):
            raise ValueError(f"mu_ratio must be between 0 and 1, got {self.mu_ratio}")

    def with_overrides(self, **changes: Any) -> GaussianComponent:
        """Return a validated copy with the given fields replaced; ``None`` values are ignored."""
        return replace(self, **{key: value for key, value in changes.items() if value is not None})

    @property
    def mu(self) -> float:
        """Location of the underlying normal distribution in kW."""
        return self.mu_ratio * self.p_max

    @property
    def sigma(self) -> float:
        """Scale of the underlying normal distribution in kW."""
        return self.sigma_ratio * self.p_max

    @property
    def mean(self) -> float:
        """Expected active power in kW, i.e. the mean of the truncated distribution."""
        if self.sigma == 0:
            return float(min(self.p_max, max(0.0, self.mu)))
        a, b = self._standardized_bounds()
        return float(truncnorm.mean(a, b, loc=self.mu, scale=self.sigma))

    def expected_load(self, p_on: float) -> float:
        """Expected load per connection (kW) for activation probability ``p_on`` (the N -> inf limit)."""
        return self.mean * p_on

    def _standardized_bounds(self) -> tuple[float, float]:
        return (0.0 - self.mu) / self.sigma, (self.p_max - self.mu) / self.sigma

    def sample_active_power(
        self,
        size: int,
        rng: np.random.Generator | int | None = None,
    ) -> np.ndarray:
        """Sample active power values (kW) from the truncated normal distribution on [0, p_max].

        Args:
            size: Number of samples to generate.
            rng: NumPy Generator or integer seed for reproducibility.
        """
        if size <= 0:
            return np.empty(0, dtype=float)

        if self.sigma == 0:
            return np.full(size, self.mean, dtype=float)

        a, b = self._standardized_bounds()
        samples = truncnorm.rvs(a=a, b=b, loc=self.mu, scale=self.sigma, size=size, random_state=rng)
        # Clip for safety against floating point inaccuracies at the bounds
        return np.clip(samples, 0.0, self.p_max)
