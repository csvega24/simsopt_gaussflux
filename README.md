# simsopt_gaussflux

`gaussflux` computes the expectation of SIMSOPT's `SquaredFlux` over Gaussian coil errors, to second order in the error
amplitude, J = f₀ + Δ₂, with its gradient on the coils' degrees of freedom. It is a drop-in replacement for the sample
mean of `SquaredFlux` over perturbed coil sets used in stochastic stage-two coil optimization: same call pattern, the
same three definitions of the flux objective, and no samples.

- f₀ is the squared flux of the unperturbed coils.
- Δ₂ = ½ Σₐ λₐ ∂²f/∂ξₐ² at ξ = 0, a sum over the Karhunen–Loève modes ψₐ (variance λₐ) of the two-level coil-error law
  (systematic errors shared by each base coil and its symmetry images, statistical errors per coil). The tests check this
  identity against the unexpanded objective to 1e-10 relative.
- The derivation is in a write-up that is not part of this repository.

The user guide is [docs/USAGE.md](docs/USAGE.md). The evidence for its numbers is [docs/EVIDENCE.md](docs/EVIDENCE.md).

## Install

In a Python environment that already has SIMSOPT, JAX, NumPy, SciPy and SymPy, from this folder:

```bash
pip install --no-build-isolation --no-deps -e .
```

`--no-deps` resolves no dependency. `--no-build-isolation` uses the environment's setuptools.

## What it was tested with, and how

- **Only in one environment:** Python 3.11.14, SIMSOPT 1.10.7.dev473+g0c099d55a, JAX 0.10.2, NumPy 2.4.6, SciPy 1.17.1,
  SymPy 1.14.0, on macOS (Apple silicon).
  - That SIMSOPT is a development build, not a release.
  - The package uses SIMSOPT's private modules (`simsopt._core.derivative`, `simsopt._core.optimizable`) and its compiled
    `simsoptpp`, so other SIMSOPT versions may not work.
- **The installation test.** A copy of this repository's files was installed (`pip install --no-index --no-deps
  --no-build-isolation`) into a new virtual environment layered over the packages of the development environment above,
  with the network blocked. In that environment:
  - the shipped tests (`tests/`) pass;
  - the example's `--check` passes;
  - the Python blocks of `docs/USAGE.md` and `docs/EVIDENCE.md` run.
  - This shows that the files of this repository install and pass on their own. It does not show an installation from a
    package index or into another environment.
- **The reviewed code.** The SHA-256 of the eight package files, as reviewed and as published:

  | File | SHA-256 |
  | --- | --- |
  | `gaussflux/__init__.py` | `8e85c1715b9028d50e7653d00c8dbe796b94320caf7235b77370041fe8ca944f` |
  | `gaussflux/analysis.py` | `fc3b988a77bd436a2c1b1a20a103f1f40a9403a685c466fcbc0a6898ec78f79f` |
  | `gaussflux/gaussian_flux.py` | `e22789228e9e43927d86716d8709f5664b80ca8b8bea26e666689bf2aa2a1323` |
  | `gaussflux/kernels.py` | `49f3e8d83f1439694414eee3c47a3d9712e3275f2e4697c78846e8d2ba9ccec0` |
  | `gaussflux/kl_modes.py` | `43d4e52ac66b7ac0f321450528617ae738e3625e16ba0fc3de6f30367c0d4d04` |
  | `gaussflux/mc_reference.py` | `386c69d8acf31d141126385bdf9094689a655e5b327455552bd8f3e4bd791f4d` |
  | `gaussflux/oracles.py` | `82a9302c6424a2381dc8d6a7d90264e997171105b987b6aaa85efe32748f2af3` |
  | `gaussflux/response.py` | `14d3d4714189335bd0d1845a24bca18a282df5b8d05c6f264dc255136efd4bc7` |

## Limits

- **Optimizations.** Every optimization run with this objective during development used the correlation length
  l = 0.5. At other l only single designs were checked.
- **The law residual.** At construction the object measures how closely its retained modes' covariance matches the
  stated kernel, and warns or refuses by that measure. It is a measure of the covariance. It is not a bound on the error
  of Δ₂ or of the remainder R.
- **Long correlation lengths.** For the default (kernel-exact) builder between l ≈ 1.1 and 1.45, SIMSOPT's eleven-image
  kernel is not a covariance. The object warns, and works with the kernel's nearest covariance. From l = 1.5 it refuses.
- **The Fourier builder** does not report the kernel's negative eigenvalue at those l.
- **Levels of different l.** A warning's suggested Fourier tail is that level's own limit. For the whole object the
  smallest accepted tail is the largest of the levels' values, which the accuracy check reports.
- **Second order in σ.** What the expansion leaves out is estimated by the accuracy check at one design, not along an
  optimization.
- **Scope of execution:** coils only (the surface is fixed); one process; the full torus for the "normalized" definition.
- **Open items from development.** A few wording notes of the development reviews were left open. A cost comparison with
  a sample mean on four processes, inside a finite-beta driver that is not part of this repository, left a difference
  that was not explained.

## The example's input file

`examples/stage_two_gaussian_example.py` reads `input.LandremanPaul2021_QA`, which is not part of this repository.
- It is the quasi-axisymmetric configuration of M. Landreman and E. Paul, Phys. Rev. Lett. 128, 035001 (2022),
  arXiv:2108.03711, distributed with SIMSOPT (under SIMSOPT's MIT licence) in SIMSOPT's `tests/test_files/`.
- The example finds it there when SIMSOPT is installed from its source tree. Otherwise, give its path with `--surface`.

## References inside the code

The docstrings and comments of the package and of the tests refer to the development project's internal plan and review
records: for example `records/…`, `PLAN.md §…`, unit and gate names such as "unit 4.3" or "clause (d3)", and item labels
such as "R6", "RC1" or "reading (x)". Those records are not published. The references are kept so that the published code
is byte for byte the code that was reviewed.

The second reviewer read the code and an earlier version of the usage page. It did not read the final usage page, nor the
version of it in this repository.

## How it was made

This package was written by an AI model (Claude Opus 5.5) as developer, under a plan and reviews by two other AI models
(Claude Fable 5.2 and GPT-6 Astra). Every stage was accepted by the project's human lead before the next began.

## Licence

MIT; see [LICENSE](LICENSE).
