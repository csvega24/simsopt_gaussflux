r"""Stage-two coil optimization with the Gaussian average of the squared flux in place of a sample mean.

The problem is SIMSOPT's stochastic stage-two example (examples/2_Intermediate/stage_two_optimization_stochastic.py): the QA
configuration of arXiv:2108.03711 (input.LandremanPaul2021_QA), 4 base coils of order 24, nfp 2, stellarator symmetric, a zero
target, and the example's penalty terms and weights. There, the flux term is the mean of SquaredFlux over N_SAMPLES perturbed coil
sets; here it is f0 + Delta2, the expectation of SquaredFlux under the same two-level Gaussian error law to second order in sigma:

    J = GaussianSquaredFlux (f0 + Delta2)
        + LENGTH_WEIGHT * sum CurveLength + DISTANCE_WEIGHT * CurveCurveDistance + CURVATURE_WEIGHT * LpCurveCurvature
        + MSC_WEIGHT * MeanSquaredCurvature penalty + ARCLENGTH_WEIGHT * ArclengthVariation

Error model (as the example): sigma = 1 mm per Cartesian component per level, correlation length l = 0.5 in the curve parameter,
kernel exp(-d^2/l^2) periodized on [0,1); systematic errors (one per base coil, carried to its images) plus statistical errors (one per
coil). The Gaussian objective uses the exact kernel; a sample mean draws from its sampler's realized law, which can differ from
the kernel (see docs/USAGE.md, section 5).

  python stage_two_gaussian_example.py --check      build, print the summary, Taylor test of the gradient, accuracy check; no optimization
      "check passed" means: the Taylor test (minimum relative error <= 1e-6), the identity of the retained law, and the law residual of
      the retained mode set against the stated kernel <= 1e-6 (retained_is_stated: agreement of covariance on the
      stated scales, not equality of laws and not a bound on Delta2). It is not a verdict on the size of the remainder R; read R
      and its standard error from the printed accuracy check (relative figures print as undefined when Delta2 = 0).
  python stage_two_gaussian_example.py --maxiter 20 optimize with L-BFGS-B, write outputs to --out
Needs gaussflux importable (pip install --no-build-isolation --no-deps -e . from the repository root) and SIMSOPT's test file
input.LandremanPaul2021_QA (found beside an editable SIMSOPT install, in SIMSOPT's tests/test_files/, or given with --surface).
Written 2026-11-19 and 2026-11-20 by Claude Opus 5.5 (developer).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import simsopt
from simsopt.field import Current, coils_via_symmetries
from simsopt.geo import (ArclengthVariation, CurveCurveDistance, CurveLength, LpCurveCurvature, MeanSquaredCurvature,
                         SurfaceRZFourier, create_equally_spaced_curves)
from simsopt.objectives import QuadraticPenalty

from gaussflux import GaussianSquaredFlux

NCOILS, R0, R1, ORDER = 4, 1.0, 0.5, 24
LENGTH_WEIGHT, DISTANCE_THRESHOLD, DISTANCE_WEIGHT = 1e-6, 0.1, 10.0
CURVATURE_THRESHOLD, CURVATURE_WEIGHT, MSC_THRESHOLD, MSC_WEIGHT, ARCLENGTH_WEIGHT = 5.0, 1e-6, 5.0, 1e-6, 1e-2
SIGMA, L = 1e-3, 0.5


def default_surface_file():
    p = Path(simsopt.__file__).resolve().parents[2] / "tests" / "test_files" / "input.LandremanPaul2021_QA"
    return p if p.exists() else None


def build(surface_file, nphi, ntheta, sigma, l, levels):
    s = SurfaceRZFourier.from_vmec_input(str(surface_file), range="full torus", nphi=nphi, ntheta=ntheta)
    base_curves = create_equally_spaced_curves(NCOILS, s.nfp, stellsym=True, R0=R0, R1=R1, order=ORDER)
    base_currents = [Current(1e5) for _ in range(NCOILS)]
    base_currents[0].fix_all()
    coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
    curves = [c.curve for c in coils]
    Jf = GaussianSquaredFlux.from_error_model(s, coils, s.nfp, True, sigma, l, levels=levels)
    Jls = [CurveLength(c) for c in base_curves]
    JF = (Jf + LENGTH_WEIGHT * sum(Jls) + DISTANCE_WEIGHT * CurveCurveDistance(curves, DISTANCE_THRESHOLD, num_basecurves=NCOILS)
          + CURVATURE_WEIGHT * sum(LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves)
          + MSC_WEIGHT * sum(QuadraticPenalty(MeanSquaredCurvature(c), MSC_THRESHOLD, "max") for c in base_curves)
          + ARCLENGTH_WEIGHT * sum(ArclengthVariation(c) for c in base_curves))
    return s, coils, Jf, JF


def taylor_test(JF, seed=1):
    """Central differences of JF along one random direction, h = 1e-2 ... 1e-7 (relative to the dofs' scale)."""
    x0 = JF.x.copy()
    v = np.random.default_rng(seed).standard_normal(x0.size) * np.maximum(np.abs(x0), 1e-2)
    g = float(JF.dJ() @ v)
    rows = []
    for h in 10.0 ** -np.arange(2.0, 7.01, 1.0):
        JF.x = x0 + h * v; jp = JF.J()
        JF.x = x0 - h * v; jm = JF.J()
        rows.append({"h": float(h), "fd": (jp - jm) / (2 * h), "rel_err": abs((jp - jm) / (2 * h) - g) / abs(g)})
    JF.x = x0
    return {"grad_dot_v": g, "rows": rows, "min_rel_err": min(r["rel_err"] for r in rows)}


def accuracy_lines(acc):
    """The printed lines of the accuracy check; the relative figures are None when
    Delta2 = 0, and then print as undefined, with the absolute mean and standard error kept."""
    lines = []
    for est in ("plain", "reduced"):
        e = acc[est]
        rel = ("undefined (Delta2 = 0)" if e["rel_delta2"] is None else
               f"{100 * e['rel_delta2']:.2f} % +- {100 * e['se_rel_delta2']:.2f} % of Delta2")
        tot = "undefined (f0 + Delta2 = 0)" if e["rel_total"] is None else f"{100 * e['rel_total']:.3f} % of f0 + Delta2"
        lines.append(f"accuracy check, {est}: R = {e['mean']:.3e} +- {e['se']:.1e}  ({rel}; {tot})")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="build, summary, Taylor test, accuracy check; no optimization")
    ap.add_argument("--maxiter", type=int, default=400)
    ap.add_argument("--surface", default=None, help="path of input.LandremanPaul2021_QA")
    ap.add_argument("--nphi", type=int, default=64)
    ap.add_argument("--ntheta", type=int, default=16)
    ap.add_argument("--sigma", type=float, default=SIGMA, help="metres per component per level")
    ap.add_argument("--correlation-length", type=float, default=L, help="curve parameter")
    ap.add_argument("--levels", default="both", choices=("both", "systematic", "statistical"))
    ap.add_argument("--pairs", type=int, default=32, help="mirrored pairs of the accuracy check (check mode)")
    ap.add_argument("--out", default=os.environ.get("PHASE0_RUN_DIR", "output_gaussian_example"))
    a = ap.parse_args()
    surface_file = Path(a.surface) if a.surface else default_surface_file()
    if surface_file is None or not surface_file.exists():
        sys.exit("input.LandremanPaul2021_QA not found; pass --surface")
    s, coils, Jf, JF = build(surface_file, a.nphi, a.ntheta, a.sigma, a.correlation_length, a.levels)
    print(Jf, flush=True)
    out = {"surface_file": str(surface_file), "nphi": a.nphi, "ntheta": a.ntheta, "error_model": Jf.error_model()}
    if a.check:
        t0 = time.perf_counter()
        tt = taylor_test(JF)
        print("Taylor test (central differences), relative error per h:", [f"{r['h']:.0e}: {r['rel_err']:.1e}" for r in tt["rows"]], flush=True)
        acc = Jf.accuracy_check(n_pairs=a.pairs, seed=0)
        for line in accuracy_lines(acc):
            print(line, flush=True)
        print(f"identity: 1/2 tr(Lambda H) = {acc['identity']['half_trace_Lambda_H']:.12e} against Delta2 = {acc['delta2']:.12e} "
              f"(relative {acc['identity']['rel_diff']:.1e}, {'pass' if acc['identity']['pass'] else 'FAIL'})", flush=True)
        lr = acc["law_residual"] or {}
        print("law residual against the stated kernel: " + "; ".join(f"{k} {v['max']:.1e}" for k, v in lr.items())
              + f"; retained_is_stated {acc['retained_is_stated']}", flush=True)
        passed = bool(tt["min_rel_err"] <= 1e-6 and acc["identity"]["pass"] and acc["retained_is_stated"] is True)
        out.update(taylor=tt, accuracy={k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk not in ("per_pair", "f_plus", "f_minus", "q")})
                                        for k, v in acc.items()}, check_passed=passed, seconds=time.perf_counter() - t0)
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "check.json").write_text(json.dumps(out, indent=1, default=float))
        print("check", "passed" if passed else "FAILED")
        sys.exit(0 if passed else 1)
    hist = []

    def fun(dofs):
        JF.x = dofs
        J, grad = JF.J(), JF.dJ()
        hist.append({"J": float(J), "flux": float(Jf.J()), "grad_norm": float(np.linalg.norm(grad))})
        return J, grad
    t0 = time.perf_counter()
    res = minimize(fun, JF.x, jac=True, method="L-BFGS-B", options={"maxiter": a.maxiter, "maxcor": 300}, tol=1e-15)
    JF.x = res.x
    Path(a.out).mkdir(parents=True, exist_ok=True)
    from simsopt import save
    save(coils, str(Path(a.out) / "coils_opt.json"))
    out.update(status=int(res.status), message=str(res.message), nit=int(res.nit), nfev=int(res.nfev), J_final=float(res.fun),
               f0_final=Jf.f0(), delta2_final=Jf.delta2(), seconds=time.perf_counter() - t0, history=hist)
    (Path(a.out) / "summary.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"L-BFGS-B: {res.message}; nit {res.nit}, nfev {res.nfev}; J {res.fun:.6e}, f0 {Jf.f0():.6e}, Delta2 {Jf.delta2():.6e}")


if __name__ == "__main__":
    main()
