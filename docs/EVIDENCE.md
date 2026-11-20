# Evidence for the numbers in USAGE.md

Every number of `docs/USAGE.md` that is not a definition has an entry here.
- **Reproducible here.** Entries R1–R7 are Python blocks that use only this package and SIMSOPT. Run in order, they print
  the numbers the page quotes; the output printed under each block is from the run named at the end of this file.
- **Development measurements.** Entries D1–D3 come from the project in which the package was developed. Their inputs
  (coil designs, timing scripts) and raw outputs are not published; each entry gives the command, seed, settings,
  environment and date as recorded there.

Kernel convention throughout: SIMSOPT's σ² exp(−d²/l²) periodized on [0, 1) with images j = −5…5; σ in metres per
Cartesian component per level. Environment of every run: Python 3.11.14, SIMSOPT 1.10.7.dev473+g0c099d55a (a
development build), JAX 0.10.2, NumPy 2.4.6, SciPy 1.17.1, SymPy 1.14.0, on macOS (Apple silicon, 10 cores).

## Setup used by R2–R7

The coils and surface of the tests (`tests/test_phase4_3.py`, `example()`): 4 base coils of order 5, nfp 2,
stellarator-symmetric, Fourier coefficients perturbed with seed 7, a 32×16 full-torus surface.

```python
import math
import warnings
import numpy as np
from simsopt.field import Current, coils_via_symmetries
from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves
from gaussflux import GaussianSquaredFlux, kl_modes as kl, mc_reference as mc


def example(nq=75, nfp=2, stellsym=True):
    base = create_equally_spaced_curves(4, nfp, stellsym=stellsym, R0=1.0, R1=0.5, order=5, numquadpoints=nq)
    rng = np.random.default_rng(7)
    for c in base:
        c.x = c.x + 0.02 * rng.standard_normal(c.x.shape)
    currents = [Current(1.0) * 1e5 for _ in range(4)]
    currents[0].fix_all()
    coils = coils_via_symmetries(base, currents, nfp, stellsym)
    s = SurfaceRZFourier(nfp=2, stellsym=True, mpol=2, ntor=2, quadpoints_phi=np.linspace(0, 1, 32, endpoint=False),
                         quadpoints_theta=np.linspace(0, 1, 16, endpoint=False))
    s.set_rc(0, 0, 1.0); s.set_rc(1, 0, 0.18); s.set_zs(1, 0, 0.18); s.set_rc(1, 1, 0.03); s.set_zs(1, 1, -0.03)
    return base, coils, s


def quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)
```

## R1. The rms displacement and the cutoff M (USAGE §3)

```python
for sig in (1e-3, 3e-3, 5e-3, 1e-2):
    print("R1 rms [mm]", sig, [round(1e3 * sig * math.sqrt(sum(math.exp(-(j / l) ** 2) for j in range(-5, 6))), 3)
                             for l in (0.05, 0.1, 0.2, 0.5, 1.0)])
print("R1 M (tail 1e-12)", [kl.fourier_cutoff(l, 1e-12) for l in (0.05, 0.1, 0.2, 0.5, 1.0)])
```

Output:

```text
R1 rms [mm] 0.001 [1.0, 1.0, 1.0, 1.018, 1.331]
R1 rms [mm] 0.003 [3.0, 3.0, 3.0, 3.054, 3.994]
R1 rms [mm] 0.005 [5.0, 5.0, 5.0, 5.091, 6.657]
R1 rms [mm] 0.01 [10.0, 10.0, 10.0, 10.182, 13.314]
R1 M (tail 1e-12) [32, 16, 8, 3, 1]
```

## R2. The precision of the kernel-exact factorization at short l (USAGE §4)

```python
base, coils, s = example(nq=128)
G = quiet(GaussianSquaredFlux.from_error_model, s, coils, 2, True, 1e-3, 0.001, levels="systematic")
em = G.error_model()
print("R2 l 0.001, 128 points: law residual", f"{em['law_residual']['systematic']['max']:.2e}",
      "cause", em["law_residual_cause"]["systematic"])
```

Output:

```text
R2 l 0.001, 128 points: law residual 1.24e-09 cause factorization
```

## R3. The smallest tail that keeps the whole Fourier table (USAGE §4, §7)

```python
print("R3 whole-table tail at l 0.006:", kl.fourier_whole_table_tail(0.006), "M:", kl.fourier_cutoff(0.006, kl.fourier_whole_table_tail(0.006)))
```

Output:

```text
R3 whole-table tail at l 0.006: 9.13e-08 M: 200
```

## R4. Levels of different l share one tail (USAGE §4)

```python
base, coils, s = example()
ells = {"systematic": 0.006, "statistical": 0.012}
try:
    quiet(GaussianSquaredFlux.from_error_model, s, coils, 2, True, 1e-3, ells, builder="fourier", tail=1e-15)
    print("R4 tail 1e-15: built")
except ValueError as err:
    print("R4 tail 1e-15: refused:", str(err)[:90])
G = quiet(GaussianSquaredFlux.from_error_model, s, coils, 2, True, 1e-3, ells, builder="fourier", tail=9.13e-08)
print("R4 tail 9.13e-08: built,", G.modes.size, "modes")
```

Output:

```text
R4 tail 1e-15: refused: level systematic: the Fourier table ends at m = 200 and the tail 1e-15 is not reached at l
R4 tail 9.13e-08: built, 14460 modes
```

## R5. The stated symmetry decides which coils share a systematic error (USAGE §4)

```python
base, coils, s = example()
d = {}
for nfp, ss in ((2, True), (2, False), (1, False)):
    d[nfp, ss] = quiet(GaussianSquaredFlux.from_error_model, s, coils, nfp, ss, 3e-3, 0.2).delta2()
for k in ((2, False), (1, False)):
    print("R5 stated", k, f"Delta2 {100 * (d[k] / d[2, True] - 1):+.2f} % against (2, True)")
```

Output:

```text
R5 stated (2, False) Delta2 +0.42 % against (2, True)
R5 stated (1, False) Delta2 +1.32 % against (2, True)
```

## R6. Two kernel-exact factors of the same kernel (USAGE §5)

The (value, first-derivative) covariance of an n_derivs = 1 factor against the leading block of an n_derivs = 2 factor,
largest entry difference over σ² (value block) and over 2σ²/l² (derivative block), 128 points, σ 1 mm.

```python
qp, sig = np.linspace(0, 1, 128, endpoint=False), 1e-3
n = len(qp)
for l in (0.2, 0.5, 1.0, 1.2, 1.45):
    L1 = mc.EighSampler(qp, sig, l, 1).L
    L2 = mc.EighSampler(qp, sig, l, 2).L[:2 * n]
    C1, C2 = L1 @ L1.T, L2 @ L2.T
    print("R6 l", l, f"value {np.abs(C1[:n, :n] - C2[:n, :n]).max() / sig ** 2:.1e}",
          f"derivative {np.abs(C1[n:, n:] - C2[n:, n:]).max() / (2 * sig ** 2 / l ** 2):.1e}")
```

Output:

```text
R6 l 0.2 value 3.2e-11 derivative 1.0e-14
R6 l 0.5 value 1.8e-12 derivative 6.4e-15
R6 l 1.0 value 1.4e-11 derivative 2.1e-10
R6 l 1.2 value 1.5e-08 derivative 3.0e-07
R6 l 1.45 value 5.2e-07 derivative 3.6e-06
```

## R7. A coarse cutoff: the reduced estimate speaks for the retained modes only (USAGE §7)

```python
base, coils, s = example()
G = quiet(GaussianSquaredFlux.from_error_model, s, coils, 2, True, 1e-3, 0.2, rel_cut=0.5)
res = G.accuracy_check(n_pairs=128, seed=194301, identity=False)
p, r, om = res["plain"], res["reduced"], res["omitted_delta2"]
print("R7 reduced", f"{100 * r['rel_delta2']:+.4f} % +- {100 * r['se_rel_delta2']:.4f} %",
      "| plain", f"{100 * p['rel_delta2']:+.2f} % +- {100 * p['se_rel_delta2']:.2f} %",
      "| omitted_delta2", f"{100 * om['over_abs_delta2']:+.2f} %", "of |Delta2|")
```

Output:

```text
R7 reduced +0.0037 % +- 0.0002 % | plain +43.31 % +- 4.08 % | omitted_delta2 +38.29 % of |Delta2|
```

## D1. Plain and reduced estimates on one design (USAGE §7)

- Design: a coils-only stage-two design of the LP-QH configuration, optimized with this objective during development (its
  coils are not published).
- Law: realized modes of SIMSOPT's `GaussianSampler` (n_derivs 1, `rel_cut` 1e-13), σ 5 mm per level, l 0.5, both
  levels, "quadratic flux", 64×16 full-torus surface.
- Plain estimate: 512 mirrored pairs, seed 43302: R = −0.94 % ± 1.40 % of Δ₂.
- Reduced estimate: 128 pairs, seed 94340: R = +0.632 % ± 0.038 % of Δ₂. The two differ by 1.12 combined standard errors.
- Command (development script, not published): `gate_d.py --part d3u42`; run on 2026-11-19 at 12:07 CST, and again with
  equal numbers on 2026-11-19 and 2026-11-20.

## D2. The cost on the example (USAGE §8)

- Problem and protocol as stated in USAGE §8.
- Command (development script, not published): `time_example.py`, run on 2026-11-19 at 13:15 CST.
- Machine: Apple silicon, 10 cores (4 performance, 6 efficiency), macOS, the environment above.
- Peak resident memory 1.59 GiB. The timings are of one machine at one time; they are not benchmarks.

## D3. The validation scope (USAGE §6)

- The development project's plan and review records (not published) list the optimizations run with this objective: the
  coils-only stage-two optimizations of its Phases 3 and 4, of its unit 4.2 (the LP-QA, LP-QH and NCSX configurations of
  the Gallegos et al. study) and the end-to-end runs of a finite-beta driver in its unit 4.3. All used l = 0.5.
- At other l (for example 0.2), only single-design checks were made, without optimization.
- Recorded on 2026-11-20 by the project's reviewer at the close of its unit 4.3.

## The run of R1–R7

The outputs shown under R1–R7 are from one run of the blocks of this page, in order, on 2026-11-20 at 14:13 CST, in the
environment above, from a folder outside the repository. The test of this repository's installation runs them again
(see the README).
