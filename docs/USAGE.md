# gaussflux: the Gaussian average of the squared flux, for stage-two coil optimization

`gaussflux.GaussianSquaredFlux` is the expectation of SIMSOPT's `SquaredFlux` over Gaussian coil errors, to second order in
the error amplitude: J = f₀ + Δ₂.
- f₀ is the squared flux of the unperturbed coils.
- Δ₂ = ½ tr(Λ ∇²f) is the second-order correction under the error covariance Λ.

It replaces the sample mean of `SquaredFlux` over perturbed coil sets that stochastic stage-two scripts use: same call
pattern, same three definitions, J and dJ, and no samples. The derivation is in a write-up that is not part of this
repository. Every measured number on this page has an entry in `docs/EVIDENCE.md`: most can be rerun with this package
alone; the others are development measurements whose inputs are not published.

## 1. Install

From a Python environment that already has SIMSOPT, JAX, NumPy, SciPy and SymPy, in the repository's root folder:

```bash
pip install --no-build-isolation --no-deps -e .
```

- Both flags matter. `--no-deps` resolves no dependency. `--no-build-isolation` uses the environment's setuptools instead
  of fetching a build backend.
- Only the package `gaussflux` is installed.

## 2. The one call

```python
import numpy as np
from simsopt.field import Current, coils_via_symmetries
from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves
from gaussflux import GaussianSquaredFlux

nfp = 2
base = create_equally_spaced_curves(4, nfp, stellsym=True, R0=1.0, R1=0.5, order=5, numquadpoints=75)
currents = [Current(1e5) for _ in range(4)]
currents[0].fix_all()
coils = coils_via_symmetries(base, currents, nfp, True)
surface = SurfaceRZFourier(nfp=nfp, stellsym=True, mpol=1, ntor=0,
                           quadpoints_phi=np.linspace(0, 1, 32, endpoint=False),
                           quadpoints_theta=np.linspace(0, 1, 16, endpoint=False))
surface.set_rc(0, 0, 1.0); surface.set_rc(1, 0, 0.18); surface.set_zs(1, 0, 0.18)

Jf = GaussianSquaredFlux.from_error_model(surface, coils, nfp, True, sigma=3e-3, correlation_length=0.2,
                                          levels="both", definition="quadratic flux")
print(Jf)                      # the error model, the modes, the rms displacement, the coil-surface distance
J, dJ = Jf.J(), Jf.dJ()        # f0 + Delta2 and its gradient on the coils' dofs, as SquaredFlux
print(J, Jf.f0(), Jf.delta2())
```

`Jf` is a SIMSOPT `Optimizable` that depends on the coils. Use it where the sample mean was used, for example
`JF = Jf + LENGTH_WEIGHT * sum(CurveLength(c) for c in base) + ...`. The arguments:

| Argument | Meaning |
| --- | --- |
| `sigma` | standard deviation in metres, **per Cartesian component, per level**; one number or `{level: value}` |
| `correlation_length` | l in the curve parameter t ∈ [0, 1), kernel exp(−d²/l²) periodized on [0, 1) (sum over j = −5…5), as SIMSOPT's `GaussianSampler`; one number or `{level: value}` |
| `levels` | `"both"`, `"systematic"` (one error per base coil, carried to its images by the symmetry maps), `"statistical"` (one independent error per coil), or a tuple of the last two |
| `definition`, `target` | as `SquaredFlux`: `"quadratic flux"` (default), `"normalized"`, `"local"`; the target is copied |
| `builder` | `"kernel_exact"` (default: eigen-decomposition of the kernel matrix on the coils' points, rel_cut 1e-10) or `"fourier"` (the kernel's Fourier series, tail 1e-12) |
| `rel_cut`, `tail` | the cutoffs of the two builders (eigenvalues kept above rel_cut × the largest; Fourier tail); each a real scalar in [1e-15, 1): a Python int or float, Decimal, Fraction, a real NumPy scalar or a 0-d real array; anything else refused (§4) |
| `sampler` | a sampler object (SIMSOPT's `GaussianSampler` or `gaussflux.EighSampler`, or `{level: sampler}`); then the objective averages over **that sampler's realized law** (§5) |

The coil list must be `coils_via_symmetries(base, currents, nfp, stellsym)` with the same `nfp` and `stellsym`. A list in
another order, or built for another symmetry, is refused.

## 3. What σ and l mean

The rms displacement per Cartesian component of one level, at every point of the coil, is
σ (Σ_{j=−5…5} exp(−j²/l²))^½. With both levels, the two add in quadrature. The summary prints it. The table gives one level,
in mm:

| σ \ l | 0.05 | 0.1 | 0.2 | 0.5 | 1.0 |
| --- | --- | --- | --- | --- | --- |
| 1 mm | 1.000 | 1.000 | 1.000 | 1.018 | 1.331 |
| 3 mm | 3.000 | 3.000 | 3.000 | 3.054 | 3.994 |
| 5 mm | 5.000 | 5.000 | 5.000 | 5.091 | 6.657 |
| 10 mm | 10.000 | 10.000 | 10.000 | 10.182 | 13.314 |

- The table and the cutoffs M below are computed in `docs/EVIDENCE.md`, R1.
- For l ≥ 0.5 the periodized kernel adds its images, so the displacement exceeds σ.
- A short l makes the error rough along the coil and costs more modes: one level keeps 2M + 1 functions per component.
  - M = 32 at l = 0.05, 16 at l = 0.1, 8 at l = 0.2, 3 at l = 0.5, 1 at l = 1.0 (tail 1e-12).
  - With `builder="kernel_exact"` the count never exceeds 2 × points per coil.

## 4. Reading the summary, and what is refused

The summary lists:
- the stated symmetry: `nfp`, `stellsym` and the number of base coils, with a note when they differ from the surface's own;
- the **law residual** per level (below), and whether the retained modes' covariance agrees with the stated law within 1e-6;
- the levels, σ and l, the builder, and the modes per level;
- the reference cutoff M of the kernel (tail 1e-12) beside the points per coil, and for the Fourier builder the M it
  retained;
- the rms displacement, by the formula and as realized by the mode set (minimum and maximum along the coil);
- the smallest coil-to-surface distance of the current coils ("now") and the ratio of the combined rms to it;
  `error_model()` keeps the values taken at construction under `..._at_construction`;
- the path (the isotropic shortcut when the covariance is isotropic per coil) and its certificate.

**The law residual.** At construction the objective compares the law its modes carry with the stated law, per level:
- the stated law is the eleven-image kernel above, computed from its closed form, for both builders; for a supplied
  sampler it is that sampler's realized law (§5);
- the figure is the largest error per block (values, cross, first derivatives) on the coil, over the pointwise variance,
  the derivative blocks scaled by max(1, 2πM)^k;
- up to 1e-10 nothing is said; above it a warning gives the figure per block, its cause and a remedy; above 1e-6
  `error_model()["retained_is_stated"]` is false, and the accuracy check (§7) says that its reduced estimate and identity
  speak for the retained modes only;
- `retained_is_stated` true means that the retained modes' covariance agrees with the stated law within 1e-6 on these
  scales, and nothing more: it is neither equality of two laws, nor a statement that two estimates have one expectation,
  nor a bound on Δ₂ or on the remainder;
- the residual measures the law, not the objective: it is not a bound on the error of Δ₂.

What causes a residual, and what happens:
- **a cutoff** (`rel_cut`, `tail`) chosen coarse: warned, naming a smaller value; the object is built. The cutoff is named
  only when the tightest admissible cutoff (1e-15) keeps more functions and at least halves the residual;
- **the precision of the factorization**: at short l (below about 0.005 in the cases measured during development) the
  kernel-exact factor is itself precise only to about 1e-10 to 1e-9 on these scales (for example 1.2e-09 at l = 0.001 with
  128 points per coil; `docs/EVIDENCE.md`, R2), so a warning there may report that precision; it names the cause and
  offers the other builder only when that builder is admitted at this l and measured closer;
- **the kernel itself**: between l ≈ 1.1 and 1.5 the kernel-exact warning reports that the eleven-image kernel is no
  longer a covariance (its matrix has a negative eigenvalue, given in the warning) and that the factor is its nearest
  covariance; from l = 1.5 the construction is refused. In that band the object is an approximation that reports its
  residual: the plain estimate draws from the factor (the kernel's nearest covariance), not from the kernel;
- **the end of the Fourier table** (m ≤ 200) at small l: where the whole table falls short of the eleven-image sum by more
  than 1e-6 the construction is refused and the message names the table's end (not the periodization); where the
  tightest tail would need more than the table, a warning compares with the whole table and names the smallest tail the
  constructor accepts at that l (for example 9.13e-08 at l = 0.006; R3), never a value it would refuse. A warning's tail is
  that level's own limit: for an object whose levels have different l, the smallest tail the constructor accepts is the
  largest of the levels' values, which `omitted_delta2["tightest"]` of the accuracy check gives (for example systematic
  l = 0.006 and statistical l = 0.012: the object is refused at tail 1e-15 and built at 9.13e-08; R4);
- **the builder** at a correlation length it cannot represent to 1e-6 with any admissible cutoff: refused, with the cause
  measured at that construction. For large l the eleven-image kernel stops being a covariance (negative eigenvalues), and
  the Fourier builder realizes the infinite periodization rather than the eleven-image sum. For small l the Fourier table
  (m ≤ 200) does not reach the tail, and the Fourier builder refuses, pointing to `builder="kernel_exact"`;
- a mode set given directly to the constructor is not certified: the summary says so and `retained_is_stated` is None.

**`nfp` and `stellsym` decide which coils share a systematic error**: each base coil and its images under the stated
symmetry carry one systematic error. The same coils with another stated symmetry are a different law, and a legitimate
one. On the example of the tests (4 base coils, nfp 2, stellarator-symmetric; σ 3 mm, l 0.2) stating (nfp 2, no
stellarator symmetry) gives 8 independent systematic errors and Δ₂ +0.42 %; stating (nfp 1, none) gives 16 and +1.3 %
(`docs/EVIDENCE.md`, R5). State the symmetry the coils were built with unless you mean the other law.

**Warnings** (reported, not refused):
- **The Fourier builder keeps more functions than the points represent** (2M + 1 > points per coil). The cost grows.
  - Remedy: more points per coil, a longer l, or `builder="kernel_exact"`.
- **The combined rms displacement is at least the smallest coil-to-surface distance.** This is a check of the input's unit
  and scale (σ = 3 meaning 3 mm would trigger it), not a validity test. The absence of this warning certifies nothing; the
  accuracy check (§7) is the test.

**Refused, with a message that names the fault:**
- a coil list that is not the symmetric set of its first coils (for instance two coils swapped, or built for another
  symmetry);
- coils on different points, or a mode set built on other points;
- σ not finite or negative (σ = 0 is valid);
- l not finite or not positive;
- an unknown or empty level, or per-level values that do not cover the levels;
- `rel_cut` or `tail` outside [1e-15, 1), not finite, or not a real scalar. The type decides, before any conversion: a
  bool (also NumPy's), None, a complex value (also NumPy's, even with a zero imaginary part), text or bytes in any form
  (`str`, `bytes`, `bytearray`, `memoryview`, `np.str_`, `np.bytes_`, arrays of those, also a numeric string such as
  "1e-10"), an object array, and an array with dimensions are refused; a Python int or float, `Decimal`, `Fraction`, a
  real NumPy scalar and a 0-d array of a real dtype are accepted;
- a level with σ > 0 and no retained mode, which would silently be the nominal objective;
- a builder that cannot represent the stated kernel to 1e-6 at the given l and points (above);
- a target whose shape is not the surface grid's (quadpoints_phi × quadpoints_theta), at construction, naming both shapes;
  a target of the right size in another shape (transposed or flattened) is refused too, because the order of its entries
  cannot be known.

**A surface changed after construction is refused at the next evaluation.**
- The object would otherwise keep the old geometry for Δ₂.
- Build a new objective, with the target computed for the new surface.

## 5. The exact kernel against a given sampler

By default the objective uses the exact kernel. A sample-mean code draws from the realized law of its sampler, L Lᵀ,
which can differ from the kernel; how much depends on the sampler, σ, the number of points per coil, l and n_derivs.
- With a sampler given, the objective averages over that sampler's realized law. The summary prints the realized rms of
  value and first derivative beside the kernel's, and warns when the draws leave [½, 2] times the kernel's rms, or when
  the sampler's covariance block is determined only to more than 1e-10 of the kernel variance (its smallest computed
  eigenvalue is below −1e-10 of it: roundoff of forming a block whose largest eigenvalue is far above the variance; the
  mode set represents the block to that precision).
- To compare with a sample mean, either give the objective the sampler the sample mean uses, or draw the sample mean with
  `gaussflux.EighSampler`. Neither makes the two methods use the same law:
  - with the sample mean's sampler given, the objective takes its modes from that sampler's (value, first-derivative)
    block, truncated at `rel_cut`: the same sampler covariance up to that truncation, which says nothing about the two
    expectations;
  - with `gaussflux.EighSampler` the sample mean targets the same requested kernel as the builders, through a separate
    factor: the sample mean's is built with its own n_derivs, the objective draws with n_derivs = 1 and retains modes at
    `rel_cut`. The (value, first-derivative) covariances of the n_derivs 2 and n_derivs 1 kernel-exact factors differ by
    the amounts below: the largest entry over σ² (value block) and over 2σ²/l² (derivative block), at 128 points and
    σ 1 mm, measured at these l only (`docs/EVIDENCE.md`, R6). They measure the covariance, not the objective:
    - value block 3.2e-11, 1.8e-12, 1.4e-11 and derivative block 1.0e-14, 6.4e-15, 2.1e-10 at l = 0.2, 0.5, 1.0;
    - value block 1.5e-08 and derivative block 3.0e-07 at l = 1.2; 5.2e-07 and 3.6e-06 at l = 1.45;
  - in both cases the values can be compared subject to the checks of §7 at the design.
- With a sampler of n_derivs = 2, the objective uses the sampler's (value, first-derivative) block. That is the law of
  the (γ, γ′) displacement the flux sees.

```python
from simsopt.geo import GaussianSampler
sampler = GaussianSampler(np.asarray(base[0].quadpoints), 3e-3, 0.2, n_derivs=2)
Jf_sampler_law = GaussianSquaredFlux.from_error_model(surface, coils, nfp, True, sampler=sampler)
print(Jf_sampler_law.delta2(), "against the exact kernel's", Jf.delta2())
```

## 6. Limits

- **Coils only.** The surface and the target are fixed; the surface's dofs are not in the gradient. Single-stage use is
  not supported.
- **Second order in σ.** What the expansion leaves out is estimated by the accuracy check (§7).
- **One process.** No MPI or GPU execution of the Gaussian objective.
- **Full torus for the "normalized" definition.** A half-period evaluation is not exact for it.
- **The law is fixed at construction.** The law (the mode set, its variances and any sampler) and the coil topology are
  fixed when the object is built; to change them, build a new object. The coils' dofs, the currents and the target given
  at construction are the only things that move. The object keeps a private, read-only copy of its mode set and copies of
  its samplers; `draw_samplers()` returns copies, so changing what it returns, or the caller's own mode set or sampler,
  does not reach the object.
- **Validation scope.** Every optimization run with this objective during development was at l = 0.5; at shorter l only
  single-design checks were made (development records, not published; `docs/EVIDENCE.md`, D3). The example's `--check` passes on the Taylor test, the identity of the
  retained law and the law residual (§4); it is not a verdict on the size of the remainder R, which the check prints with
  its standard error.

## 7. The accuracy check

`Jf.accuracy_check(n_pairs, seed)` estimates, at the current coils, what second order leaves out, R = ⟨f⟩ − f₀ − Δ₂, with
its standard error. It gives R also as a fraction of |Δ₂| and of |f₀ + Δ₂| (the sign of R is kept; the standard error is
never negative; with Δ₂ = 0 these are None). It leaves the coils as found. It returns:
- **"plain":** the mean over mirrored pairs ±δ of ½[f(+δ) + f(−δ)] − f₀ − Δ₂.
  - δ is drawn through the **sampler's factor**: for the builders the kernel-exact eigen-factor of the eleven-image
    kernel's matrix (every positive eigenvalue kept, so its covariance is that matrix's nearest covariance). Between
    l ≈ 1.1 and 1.5, where the kernel is not a covariance, the result of the kernel-exact builder says so with the
    eigenvalue (`kernel_min_eigenvalue` and the plain law's text); with the Fourier builder the plain law's text names
    the nearest covariance without the eigenvalue. For a supplied
    sampler that sampler's own factor. f is SIMSOPT's `SquaredFlux` on `CurvePerturbed` coils, and nothing comes from
    the expansion.
- **"reduced":** R with the exact quadratic term subtracted pair by pair (Route 2), drawn from the **retained modes**,
  always. With `retained_is_stated` true (§4) the label adds that the retained covariance agrees with the stated law within
  1e-6 and gives the residual; that is not a statement that "plain" and "reduced" have one expectation, and it is
  not a bound on Δ₂. Its error is much smaller.
  - With explicit coefficients (`xi=`) the draws are the caller's: the result says so, and the estimate is a remainder
    estimate under the retained modes only if the caller drew them as ξ_a = √λ_a Z_a with Z independent standard normal
    variables.
  - Example, on one design measured during development (`docs/EVIDENCE.md`, D1): ±0.038 % of Δ₂ from 128 pairs for
    "reduced", against ±1.40 % from 512 pairs for "plain".
- **"identity":** Δ₂ against ½ Σ_a λ_a H_aa of the unexpanded objective, along the retained modes; the reduced estimate
  rests on it. Tolerance 1e-10. It certifies the retained law, not the stated one.
- **Which law each speaks for:** the result carries `"laws"` (plain: the sampler's factor; reduced and identity: the
  retained mode set), the `"law_residual"` per level, and `"retained_is_stated"`: true, false, or None when the law is not
  certified (an object built from a mode set, or `samplers=` given). When it is true, `speaks_for` on the reduced estimate
  and the identity names the retained mode set with the covariance agreement and the measured residual. When it is
  false, they carry `speaks_for: "retained mode set only"`, a warning says so, and `"omitted_delta2"` gives Δ₂ of the same
  law at the tightest admissible cutoff minus Δ₂ as retained, with the cutoff used in `"tightest"`: `rel_cut` 1e-15 for
  the kernel-exact builder; for the Fourier builder the smallest `tail` the constructor accepts at the levels' l (1e-15
  where the table reaches it, otherwise the tail that keeps the whole table, m ≤ 200, for example 9.13e-08 at l = 0.006).
  It is not available for a supplied sampler, or where no tail is accepted (the result gives the reason).
  When it is true, `"omitted_delta2"` is not computed, and the result says that this is not a bound on the omitted
  contribution.
- **Does the truncation matter at this design?** Compare "plain" and "reduced" with their standard errors (a difference of
  several combined standard errors means it does), or build again with a tighter cutoff (`rel_cut` or `tail` down to
  1e-15, or for the Fourier builder down to the value its warning names) and compare Δ₂.

```python
res = Jf.accuracy_check(n_pairs=8, seed=0)
for est in ("plain", "reduced"):
    e = res[est]   # with Delta2 = 0 the relative figures are None
    rel = ("undefined (Delta2 = 0)" if e["rel_delta2"] is None
           else f"{e['rel_delta2']:+.3%} +- {e['se_rel_delta2']:.3%}")
    print(f"{est:<7} R = {e['mean']:.3e} +- {e['se']:.1e}, R/Delta2 = {rel}")
print("identity", res["identity"]["pass"])
```

- A small |R/Δ₂| means that, at this design and σ, the expansion represents the expectation under the law that estimate
  is drawn from: for "plain" the sampler's factor, for "reduced" the retained modes. For the reduced estimate it says
  nothing about what the truncation leaves out; for that, see "Does the truncation matter at this design?" above. (At
  rel_cut 0.5 on the tests' coils the reduced R is +0.0037 % of Δ₂ while the plain R is +43.31 % ± 4.08 % and
  `omitted_delta2` +38.29 %; `docs/EVIDENCE.md`, R7.)
- It is a diagnostic at one design, not a bound along an optimization.
- Benchmark numbers come from the sample-mean reference code.

## 8. Cost

Measured during development on the problem of `examples/stage_two_gaussian_example.py`, built by its own `build()`:
LandremanPaul2021_QA full torus, 64×16 surface; 4 base coils, nfp 2, stellarator-symmetric, order 24; 16 coils of 360
points; "quadratic flux"; σ 1 mm per level, both levels, kernel-exact law. One value, then one gradient after a relative
1e-13 change of every dof and back; two warm-ups, then the median of five; one process, JAX on its default threads.
Machine, command and date in `docs/EVIDENCE.md`, D2 (peak resident memory 1.59 GiB).

| Row | Modes | J alone [s] | J and dJ [s] | CPU [s] | J and dJ in `SquaredFlux` evaluations |
| --- | --- | --- | --- | --- | --- |
| Gaussian, l = 0.5 | 420 | 0.119 | 0.619 | 4.43 | 30 |
| Gaussian, l = 0.2 | 1020 | 0.167 | 0.808 | 6.04 | 39 |
| `SquaredFlux`, same grid | — | 0.0070 | 0.0209 | 0.021 | 1 |

One sample of a sample mean costs at least one `SquaredFlux` evaluation of a perturbed coil set.

## Change log

- 2026-11-20 14:14 CST — Claude Opus 5.5 (developer): created for the public repository from the development project's usage page (final state of 2026-11-20 12:49 CST): citations of unpublished records replaced by entries of docs/EVIDENCE.md; the statements about a particular sampler's error reduced to a neutral one; the sections on a finite-beta driver that is not part of this repository and its timing removed; the validation scope moved to §6.
