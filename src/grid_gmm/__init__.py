"""Grid GMM: Monte Carlo simulation of simultaneous household grid load.

Despite the historical package name ("GMM"), each connection is modelled by a *single*
truncated Gaussian component (:class:`grid_gmm.model.GaussianComponent`); Gaussian mixtures
are not implemented.
"""

__version__ = "0.2.0"
