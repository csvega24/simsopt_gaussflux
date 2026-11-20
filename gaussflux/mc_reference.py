"""Sample-mean reference for the stochastic squared-flux objective.

This module holds the sampling protocol of the two-level SIMSOPT error model
and the per-sample evaluation of the three ``SquaredFlux`` definitions. It
contains no expansion code; it is the oracle of the project (PLAN.md §3.1,
``mc_reference.py``).

Error model (read in code, see records/phase0_proposal_opus.md §2):
SIMSOPT ``GaussianSampler`` kernel sigma^2 exp(-d^2/l^2) on t in [0, 1),
periodized over -5..5; i.i.d. Cartesian components. Two levels with one
sampler: a systematic draw per base curve applied before symmetry (so each
image carries the rotated/reflected draw), then an independent statistical
draw per coil after symmetry. This is the order of
``examples/2_Intermediate/stage_two_optimization_stochastic.py`` and of
Gallegos et al.'s scripts.

Seed protocol of this project: sample i of a run with root seed r uses
``Generator(PCG64DXSM(SeedSequence(r).spawn(N)[i]))``; children of a
``SeedSequence`` depend only on i, so sample i is the same for every N.
"""

from __future__ import annotations

import hashlib
import time

import numpy as np
from numpy.random import Generator, PCG64DXSM, SeedSequence

from simsopt.field import BiotSavart, Coil, apply_symmetries_to_curves
from simsopt.geo import CurvePerturbed, GaussianSampler, PerturbationSample
from simsopt.objectives import SquaredFlux

FACTORS = ("simsopt_ldl", "eigh")
DEFINITIONS = ("quadratic flux", "normalized", "local")
QUANTILES = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)


# ---------------------------------------------------------------------------
# Coil-set structure
# ---------------------------------------------------------------------------

def base_curves_of(coils, nfp, stellsym, atol=1e-14):
    """Return the base curves of a symmetric coil list and verify the list.

    The list must equal ``apply_symmetries_to_curves(base, nfp, stellsym)``
    element by element in gamma and gammadash (max-abs difference below
    ``atol`` times the coil scale), which fixes the ordering the two-level
    draw relies on.
    """
    copies = int(nfp) * (2 if stellsym else 1)
    if len(coils) % copies:
        raise ValueError(f"{len(coils)} coils is not a multiple of {copies}")
    nbase = len(coils) // copies
    base = [c.curve for c in coils[:nbase]]
    images = apply_symmetries_to_curves(base, nfp, stellsym)
    for j, (coil, image) in enumerate(zip(coils, images)):
        for name in ("gamma", "gammadash"):
            a = getattr(coil.curve, name)()
            b = getattr(image, name)()
            scale = max(1.0, float(np.max(np.abs(b))))
            if a.shape != b.shape or np.max(np.abs(a - b)) > atol * scale:
                raise ValueError(
                    f"coil {j} {name} differs from the symmetry image of its base curve")
    qp = base[0].quadpoints
    if any(not np.array_equal(c.quadpoints, qp) for c in base):
        raise ValueError("base curves must share one quadrature grid")
    return base


# ---------------------------------------------------------------------------
# Samplers: SIMSOPT's unchanged factor, or a kernel-exact factor
# ---------------------------------------------------------------------------

def kernel_matrix(points, sigma, length_scale, n_derivs):
    """The covariance matrix SIMSOPT's GaussianSampler builds before factoring.

    Same construction as curveperturbed.py:55-70 (sympy kernel periodized over
    -5..5 and its derivatives).
    """
    from sympy import Symbol, exp, lambdify
    xs = np.asarray(points)
    n = len(xs)
    cov = np.zeros((n * (n_derivs + 1), n * (n_derivs + 1)))
    x, y = Symbol("x"), Symbol("y")
    f = sum((sigma ** 2) * exp(-(x - y + i) ** 2 / (length_scale ** 2)) for i in range(-5, 6))
    XX, YY = np.meshgrid(xs, xs, indexing="ij")
    for ii in range(n_derivs + 1):
        for jj in range(n_derivs + 1):
            g = f if ii + jj == 0 else f.diff(*(ii * [x] + jj * [y]))
            cov[ii * n:(ii + 1) * n, jj * n:(jj + 1) * n] = lambdify((x, y), g, "numpy")(XX, YY)
    return cov


class EighSampler:
    """Kernel-exact sampler: L = V sqrt(max(w, 0)) from eigh of the kernel matrix.

    Same interface and draw as SIMSOPT's GaussianSampler (z of shape
    (n(n_derivs+1), 3), sample = L @ z), so ``PerturbationSample`` accepts it.
    """

    def __init__(self, points, sigma, length_scale, n_derivs=1):
        self.points, self.sigma, self.length_scale, self.n_derivs = points, sigma, length_scale, n_derivs
        w, V = np.linalg.eigh(kernel_matrix(points, sigma, length_scale, n_derivs))
        self.L = V * np.sqrt(np.maximum(w, 0.0))

    def draw_sample(self, randomgen=None):
        n, nd = len(self.points), self.n_derivs
        if randomgen is None:
            randomgen = Generator(PCG64DXSM())
        z = randomgen.standard_normal(size=(n * (nd + 1), 3))
        out = self.L @ z
        return [out[i * n:(i + 1) * n, :] for i in range(nd + 1)]


def make_sampler(points, sigma, length_scale, n_derivs=1, factor="simsopt_ldl"):
    """``simsopt_ldl``: SIMSOPT's GaussianSampler unchanged; ``eigh``: kernel-exact."""
    if factor == "simsopt_ldl":
        return GaussianSampler(points, float(sigma), float(length_scale), n_derivs=n_derivs)
    if factor == "eigh":
        return EighSampler(points, float(sigma), float(length_scale), n_derivs=n_derivs)
    raise ValueError(f"factor must be one of {FACTORS}")


def covariance_report(sampler):
    """Realized covariance L L^T against the kernel matrix, overall and per block.

    Block (a, b) error = max|(L L^T - K)_ab| / sqrt(max diag K_aa * max diag K_bb);
    overall = max|L L^T - K| / max|K|.
    """
    n, nd = len(sampler.points), sampler.n_derivs
    K = kernel_matrix(sampler.points, sampler.sigma, sampler.length_scale, nd)
    E = sampler.L @ sampler.L.T - K
    diag = [np.max(np.diag(K)[a * n:(a + 1) * n]) for a in range(nd + 1)]
    blocks = {f"{a}{b}": float(np.max(np.abs(E[a * n:(a + 1) * n, b * n:(b + 1) * n]))
                         / np.sqrt(diag[a] * diag[b]))
              for a in range(nd + 1) for b in range(nd + 1)}
    return {"overall_rel_maxabs": float(np.max(np.abs(E)) / np.max(np.abs(K))),
            "blocks_rel": blocks, "n_points": n, "n_derivs": nd,
            "sigma": float(sampler.sigma), "length_scale": float(sampler.length_scale)}


# ---------------------------------------------------------------------------
# Draws
# ---------------------------------------------------------------------------

def sample_rng(root_seed, index, n_samples):
    """Generator of sample ``index`` under this project's seed protocol."""
    child = SeedSequence(int(root_seed)).spawn(int(n_samples))[int(index)]
    return Generator(PCG64DXSM(child))


def draw_two_level(sampler, nbase, ncoils, rng):
    """Draw systematic (per base curve) then statistical (per coil) samples."""
    systematic = [PerturbationSample(sampler, randomgen=rng) for _ in range(nbase)]
    statistical = [PerturbationSample(sampler, randomgen=rng) for _ in range(ncoils)]
    return systematic, statistical


def perturbed_coils(coils, base, nfp, stellsym, systematic, statistical):
    """Coils with the systematic draw on base curves and statistical draw per coil."""
    perturbed_base = [CurvePerturbed(c, s) for c, s in zip(base, systematic)]
    curves = apply_symmetries_to_curves(perturbed_base, nfp, stellsym)
    return [Coil(CurvePerturbed(curve, s), coil.current)
            for curve, s, coil in zip(curves, statistical, coils)]


def draw_levels(samplers, nbase, ncoils, rng):
    """Unit 4.3: draws of the levels present in ``samplers`` ({level: sampler}): systematic per base curve, then statistical
    per coil, from one generator. With both levels and one sampler these are the calls of ``draw_two_level`` in the same
    order, so the arrays are equal bit for bit. A level absent from ``samplers`` gives None."""
    systematic = ([PerturbationSample(samplers["systematic"], randomgen=rng) for _ in range(nbase)]
                  if "systematic" in samplers else None)
    statistical = ([PerturbationSample(samplers["statistical"], randomgen=rng) for _ in range(ncoils)]
                   if "statistical" in samplers else None)
    return systematic, statistical


def perturbed_coils_levels(coils, base, nfp, stellsym, systematic, statistical):
    """Unit 4.3: ``perturbed_coils`` for any subset of levels (None = level absent); equal to it with both levels."""
    curves = apply_symmetries_to_curves([CurvePerturbed(c, s) for c, s in zip(base, systematic)] if systematic is not None
                                        else list(base), nfp, stellsym)
    if statistical is not None:
        curves = [CurvePerturbed(curve, s) for curve, s in zip(curves, statistical)]
    return [Coil(curve, coil.current) for curve, coil in zip(curves, coils)]


def sample_arrays(samples):
    """Stack ``PerturbationSample`` arrays: shape (len, n_derivs+1, nqp, 3)."""
    return np.stack([np.stack([np.asarray(s[k], dtype=float)
                               for k in range(len(s._sample))]) for s in samples])


def draw_hash(systematic, statistical):
    """SHA-256 of the drawn arrays (systematic then statistical, C order)."""
    h = hashlib.sha256()
    for arr in (sample_arrays(systematic), sample_arrays(statistical)):
        h.update(np.ascontiguousarray(arr, dtype="<f8").tobytes())
    return h.hexdigest()


def array_hash(systematic_arr, statistical_arr):
    """SHA-256 of stored arrays, identical to ``draw_hash`` of the same draw."""
    h = hashlib.sha256()
    for arr in (systematic_arr, statistical_arr):
        h.update(np.ascontiguousarray(arr, dtype="<f8").tobytes())
    return h.hexdigest()


def samples_from_arrays(sampler, arrays, scale=1.0):
    """Rebuild ``PerturbationSample`` objects from stored arrays times ``scale``."""
    return [PerturbationSample(sampler, sample=[scale * a[k] for k in range(a.shape[0])])
            for a in arrays]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def flux_values(surface, coils, target=None, definitions=DEFINITIONS):
    """All requested ``SquaredFlux`` definitions from one Biot-Savart field."""
    field = BiotSavart(coils)
    field.set_points(surface.gamma().reshape((-1, 3)))
    return {d: float(SquaredFlux(surface, field, target=target, definition=d).J())
            for d in definitions}


def summarize(values):
    """Mean, sample standard deviation, s.e. and quantiles of one array."""
    v = np.asarray(values, dtype=float)
    n = v.size
    std = float(np.std(v, ddof=1)) if n > 1 else float("nan")
    return {
        "n": int(n),
        "mean": float(np.mean(v)),
        "std": std,
        "se": std / np.sqrt(n) if n > 1 else float("nan"),
        "quantiles": {f"{q:g}": float(np.quantile(v, q)) for q in QUANTILES},
    }


def run_baseline(surface, coils, nfp, stellsym, sigmas, length_scale, root_seed,
                 n_samples, target=None, definitions=DEFINITIONS, n_derivs=1,
                 progress=None, factor="simsopt_ldl", keep_samplers=False):
    """Sample means at several sigma with the same draws at every sigma.

    For each sample i the generator is rebuilt from the same child seed for
    every sigma, so the standard-normal vectors are shared and the four
    estimates are paired. Returns per-sample values, hashes (per sigma) and
    timings.
    """
    base = base_curves_of(coils, nfp, stellsym)
    nbase, ncoils = len(base), len(coils)
    qp = base[0].quadpoints
    out = {"nominal": flux_values(surface, coils, target, definitions), "sigmas": {}}
    for sigma in sigmas:
        t0 = time.perf_counter()
        sampler = make_sampler(qp, sigma, length_scale, n_derivs, factor)
        t_sampler = time.perf_counter() - t0
        values = {d: np.empty(n_samples) for d in definitions}
        hashes, times = [], np.empty(n_samples)
        for i in range(n_samples):
            t1 = time.perf_counter()
            rng = sample_rng(root_seed, i, n_samples)
            sys_s, stat_s = draw_two_level(sampler, nbase, ncoils, rng)
            pcoils = perturbed_coils(coils, base, nfp, stellsym, sys_s, stat_s)
            vals = flux_values(surface, pcoils, target, definitions)
            for d in definitions:
                values[d][i] = vals[d]
            hashes.append(draw_hash(sys_s, stat_s))
            times[i] = time.perf_counter() - t1
            if progress is not None and (i + 1) % progress == 0:
                print(f"[baseline] sigma={sigma:g} sample {i + 1}/{n_samples} "
                      f"{times[:i + 1].sum():.1f} s", flush=True)
        out["sigmas"][float(sigma)] = {
            "values": values, "hashes": hashes, "times": times,
            "sampler_build_s": t_sampler, "factor": factor,
            "summary": {d: summarize(values[d]) for d in definitions},
        }
        if keep_samplers:
            out["sigmas"][float(sigma)]["sampler"] = sampler
    return out


def draw_arrays(base_quadpoints, nbase, ncoils, sigma, length_scale, root_seed,
                n_samples, n_derivs=1, progress=None, factor="simsopt_ldl", sampler=None):
    """Held-out arrays at ``sigma`` under this project's protocol.

    Returns (systematic, statistical, hashes) with arrays of shape
    (n_samples, nbase or ncoils, n_derivs+1, nqp, 3). With the unchanged
    SIMSOPT factor the arrays are exact only at sigma times powers of two
    (records/phase0_review_fable.md); the kernel-exact factor is drawn at unit
    sigma and scaled.
    """
    if sampler is None:
        sampler = make_sampler(base_quadpoints, sigma, length_scale, n_derivs, factor)
    nqp = len(base_quadpoints)
    sys_arr = np.empty((n_samples, nbase, n_derivs + 1, nqp, 3))
    stat_arr = np.empty((n_samples, ncoils, n_derivs + 1, nqp, 3))
    hashes = []
    for i in range(n_samples):
        rng = sample_rng(root_seed, i, n_samples)
        sys_s, stat_s = draw_two_level(sampler, nbase, ncoils, rng)
        sys_arr[i] = sample_arrays(sys_s)
        stat_arr[i] = sample_arrays(stat_s)
        hashes.append(draw_hash(sys_s, stat_s))
        if progress is not None and (i + 1) % progress == 0:
            print(f"[heldout] {i + 1}/{n_samples}", flush=True)
    return sys_arr, stat_arr, hashes


# ---------------------------------------------------------------------------
# The AL driver's cloud (case 5): extract its draws without changing it
# ---------------------------------------------------------------------------

def cloud_draw_arrays(cloud, nbase):
    """Systematic and statistical arrays of every sample held by a cloud.

    ``StochasticFluxCloud`` (two-level) stores, per sample, a list of
    ``CurvePerturbed(image, statistical_sample)`` where ``image`` is the
    symmetry image of ``CurvePerturbed(base, systematic_sample)``; the first
    ``nbase`` images are those perturbed base curves themselves.
    """
    sys_list, stat_list, hashes = [], [], []
    for curves in cloud.curves:
        stat = [c.sample for c in curves]
        sys_ = [curves[j].curve.sample for j in range(nbase)]
        sys_list.append(sample_arrays(sys_))
        stat_list.append(sample_arrays(stat))
        hashes.append(draw_hash(sys_, stat))
    return np.stack(sys_list), np.stack(stat_list), hashes
