"""gaussflux: the sampling-free Gaussian average f0 + Delta2 of SIMSOPT's SquaredFlux under a Gaussian coil-error law.

Kernel convention: SIMSOPT exp(-d^2/l^2) periodized on [0,1) (sum over j = -5..5), l in the curve parameter; sigma in metres per
Cartesian component per error level. The one-call constructor is GaussianSquaredFlux.from_error_model; see production/USAGE.md.
"""

__version__ = "0.1.0"

from .gaussian_flux import DEFINITIONS, GaussianSquaredFlux  # noqa: E402
from .kl_modes import LEVELS, ModeSet, check_error_model, fourier_modes, realized_modes  # noqa: E402
from .mc_reference import EighSampler, make_sampler  # noqa: E402

__all__ = ["GaussianSquaredFlux", "DEFINITIONS", "LEVELS", "ModeSet", "check_error_model", "fourier_modes", "realized_modes",
           "EighSampler", "make_sampler", "__version__"]
