"""Model definitions for appliance/connection Gaussian load distributions."""

from dataclasses import dataclass
from typing import Optional, Union
import numpy as np
from scipy.stats import truncnorm


@dataclass
class GaussianComponent:
    """Represents a Gaussian load component for an appliance or household connection.

    Attributes:
        p_max: Maximum power in kW.
        mu_ratio: Median value when the connection is ON, expressed as a ratio of p_max (0 <= mu_ratio <= 1).
        sigma_ratio: Standard deviation expressed as a ratio of p_max (sigma_ratio > 0).
        name: Optional identifier for the component.
    """

    p_max: float
    mu_ratio: float = 0.5
    sigma_ratio: float = 0.2
    name: str = "default_load"

    def __post_init__(self):
        if self.p_max <= 0:
            raise ValueError(f"p_max must be positive, got {self.p_max}")
        if self.sigma_ratio < 0:
            raise ValueError(f"sigma_ratio must be non-negative, got {self.sigma_ratio}")
        if not (0 <= self.mu_ratio <= 1):
            raise ValueError(f"mu_ratio must be between 0 and 1, got {self.mu_ratio}")

    @property
    def mu(self) -> float:
        """Absolute median/mean power in kW when active."""
        return self.mu_ratio * self.p_max

    @property
    def sigma(self) -> float:
        """Absolute standard deviation in kW."""
        return self.sigma_ratio * self.p_max

    def sample_active_power(
        self,
        size: int,
        rng: Optional[Union[np.random.Generator, int]] = None,
    ) -> np.ndarray:
        """Sample active power values from the truncated normal distribution bounded in [0, p_max].

        Args:
            size: Number of samples to generate.
            rng: NumPy Generator or integer seed for reproducibility.

        Returns:
            1D numpy array of sampled power values in kW, all within [0, p_max].
        """
        if size <= 0:
            return np.empty(0, dtype=float)

        if self.sigma == 0:
            return np.full(size, min(self.p_max, max(0.0, self.mu)), dtype=float)

        # Standardized truncation bounds for scipy.stats.truncnorm
        a = (0.0 - self.mu) / self.sigma
        b = (self.p_max - self.mu) / self.sigma

        samples = truncnorm.rvs(
            a=a,
            b=b,
            loc=self.mu,
            scale=self.sigma,
            size=size,
            random_state=rng,
        )

        # Clip for safety against floating point inaccuracies
        return np.clip(samples, 0.0, self.p_max)
