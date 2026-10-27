"""Unit tests for GaussianComponent."""

import numpy as np
import pytest
from grid_gmm.model import GaussianComponent


def test_gaussian_component_validation():
    # Valid component
    comp = GaussianComponent(p_max=11.0, mu_ratio=0.4, sigma_ratio=0.1)
    assert comp.p_max == 11.0
    assert comp.mu == pytest.approx(4.4)
    assert comp.sigma == pytest.approx(1.1)

    # Invalid p_max
    with pytest.raises(ValueError, match="p_max must be positive"):
        GaussianComponent(p_max=-5.0, mu_ratio=0.4, sigma_ratio=0.1)

    # Invalid sigma_ratio
    with pytest.raises(ValueError, match="sigma_ratio must be non-negative"):
        GaussianComponent(p_max=10.0, mu_ratio=0.4, sigma_ratio=-0.1)

    # Valid deterministic component (sigma_ratio = 0)
    comp_zero = GaussianComponent(p_max=10.0, mu_ratio=1.0, sigma_ratio=0.0)
    assert comp_zero.sigma == 0.0
    samples_zero = comp_zero.sample_active_power(size=10)
    assert np.all(samples_zero == 10.0)

    # Invalid mu_ratio
    with pytest.raises(ValueError, match="mu_ratio must be between 0 and 1"):
        GaussianComponent(p_max=10.0, mu_ratio=1.5, sigma_ratio=0.1)


def test_sample_active_power_bounds():
    comp = GaussianComponent(p_max=10.0, mu_ratio=0.5, sigma_ratio=0.3)
    rng = np.random.default_rng(123)
    samples = comp.sample_active_power(size=1000, rng=rng)

    assert len(samples) == 1000
    assert np.all(samples >= 0.0)
    assert np.all(samples <= 10.0)


def test_sample_active_power_empty():
    comp = GaussianComponent(p_max=10.0, mu_ratio=0.5, sigma_ratio=0.1)
    samples = comp.sample_active_power(size=0)
    assert len(samples) == 0
