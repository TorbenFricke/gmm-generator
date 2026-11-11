"""Unit tests for GaussianComponent."""

import dataclasses

import numpy as np
import pytest
from scipy.stats import truncnorm

from grid_gmm.model import GaussianComponent


def test_gaussian_component_validation():
    comp = GaussianComponent(p_max=11.0, mu_ratio=0.4, sigma_ratio=0.1)
    assert comp.p_max == 11.0
    assert comp.mu == pytest.approx(4.4)
    assert comp.sigma == pytest.approx(1.1)

    with pytest.raises(ValueError, match="p_max must be positive"):
        GaussianComponent(p_max=-5.0, mu_ratio=0.4, sigma_ratio=0.1)
    with pytest.raises(ValueError, match="sigma_ratio must be non-negative"):
        GaussianComponent(p_max=10.0, mu_ratio=0.4, sigma_ratio=-0.1)
    with pytest.raises(ValueError, match="mu_ratio must be between 0 and 1"):
        GaussianComponent(p_max=10.0, mu_ratio=1.5, sigma_ratio=0.1)


def test_deterministic_component():
    comp = GaussianComponent(p_max=10.0, mu_ratio=1.0, sigma_ratio=0.0)
    assert comp.sigma == 0.0
    assert comp.mean == 10.0
    assert np.all(comp.sample_active_power(size=10) == 10.0)


def test_component_is_immutable_and_overrides_revalidate():
    comp = GaussianComponent(p_max=10.0, mu_ratio=0.5, sigma_ratio=0.1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        comp.mu_ratio = 1.5  # type: ignore[misc]

    changed = comp.with_overrides(p_max=20.0, mu_ratio=None)
    assert changed.p_max == 20.0
    assert changed.mu_ratio == 0.5  # None means "keep"
    assert comp.p_max == 10.0  # original untouched

    with pytest.raises(ValueError, match="mu_ratio"):
        comp.with_overrides(mu_ratio=1.5)


@pytest.mark.parametrize(("mu_ratio", "sigma_ratio"), [(0.35, 0.15), (0.9, 0.3), (1.0, 0.2)])
def test_mean_is_truncated_mean_not_mu(mu_ratio, sigma_ratio):
    comp = GaussianComponent(p_max=10.0, mu_ratio=mu_ratio, sigma_ratio=sigma_ratio)
    a, b = -comp.mu / comp.sigma, (comp.p_max - comp.mu) / comp.sigma
    assert comp.mean == pytest.approx(truncnorm.mean(a, b, loc=comp.mu, scale=comp.sigma))

    samples = comp.sample_active_power(size=200_000, rng=np.random.default_rng(0))
    assert samples.mean() == pytest.approx(comp.mean, rel=5e-3)
    assert comp.expected_load(0.5) == pytest.approx(0.5 * comp.mean)


def test_sample_active_power_bounds():
    comp = GaussianComponent(p_max=10.0, mu_ratio=0.5, sigma_ratio=0.3)
    samples = comp.sample_active_power(size=1000, rng=np.random.default_rng(123))
    assert len(samples) == 1000
    assert np.all((samples >= 0.0) & (samples <= 10.0))


def test_sample_active_power_reproducible_and_empty():
    comp = GaussianComponent(p_max=10.0, mu_ratio=0.5, sigma_ratio=0.1)
    assert len(comp.sample_active_power(size=0)) == 0
    np.testing.assert_array_equal(comp.sample_active_power(50, rng=7), comp.sample_active_power(50, rng=7))
