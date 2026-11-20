"""GaussianSquaredFlux: f0 + Delta2 of SIMSOPT's SquaredFlux (PLAN.md §2, §3.1).

Delta2 for a diagonal Lambda (KL basis), with u = B.n - t, w = |B|^2 of the
coil field, surface weights |n| / (n_phi n_theta) as in integral_BdotN.cpp, and
the definition's own factor 1/2:

  quadratic flux (J_B):  1/2 * 1/2 sum_a lam_a  I[ u1a^2 + 2 u0 U2aa ]            (PLAN §2)
  local (f_pw / 2):      1/2 * I[ (1/w0)( S_uu + 2 u0 trU2 - (2u0/w0) S_uw
                                + (u0^2/w0^2) S_ww - (u0^2/w0) trW2 ) ]             (eq. 37)
  normalized (f_r / 2):  1/2 * (1/D0)[ I(S_uu + 2 u0 trU2) - sum_a lam_a N1a eta1a
                                + N0 (sum_a lam_a eta1a^2 - trH2) ]                  (eq. 38)

with S_uu = sum lam u1^2, S_uw = sum lam u1 w1, S_ww = sum lam w1^2,
trU2 = sum lam B2aa.n, trW2 = sum lam (2 B0.B2aa + |B1a|^2), I[.] the weighted
surface integral, N1a = 2 I[u0 u1a], eta1a = I[w1a]/D0, trH2 = I[trW2]/D0.
Isotropic shortcut (certificate of PLAN §2): trU2 = 0, trW2 = sum lam |B1a|^2,
no second-order responses. Eq. (37) and (38) are the complete Delta2 of f_pw
and f_r (e.g. u0 = 0 gives E[f_pw] - f_pw0 = int u1^T Lambda u1 / w0), so
SIMSOPT's local = f_pw/2 and normalized = f_r/2 take 1/2 of them; for
J_B = 1/2 int u^2 the PLAN §2 form Delta2 = 1/2 sum lam int (u1^2 + 2 u0 U2)
is complete. The Hessian-trace oracle (tests) checks these factors.
"""

import copy
import math
import time
import warnings
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from simsopt._core.derivative import derivative_dec
from simsopt._core.optimizable import Optimizable
from simsopt.field import BiotSavart
from simsopt.objectives import SquaredFlux

from . import kernels as kn
from . import kl_modes as kl
from . import mc_reference as mc
from . import response as rsp

DEFINITIONS = ("quadratic flux", "normalized", "local")
# unit 4.1 item 1 (Astra finding 4): every other value is refused before route selection
PATHS = ("auto", "general", "isotropic_shortcut")


def delta2_from_responses(B0, B1, lam, B2, nhat, absn, target, definition):
    """Assemble Delta2 (jnp-compatible). B2 is None on the isotropic shortcut."""
    P = absn.shape[0]
    u0 = jnp.sum(B0 * nhat, axis=1) - target
    u1 = jnp.einsum("apk,pk->ap", B1, nhat)
    w0 = jnp.sum(B0 * B0, axis=1)
    w1 = 2.0 * jnp.einsum("apk,pk->ap", B1, B0)
    Suu = jnp.einsum("a,ap->p", lam, u1 * u1)
    if B2 is None:
        trU2 = jnp.zeros_like(u0)
        trW2 = jnp.einsum("a,ap->p", lam, jnp.sum(B1 * B1, axis=2))
    else:
        trU2 = jnp.einsum("a,apk,pk->p", lam, B2, nhat)
        trW2 = jnp.einsum("a,ap->p", lam, 2.0 * jnp.einsum("apk,pk->ap", B2, B0) + jnp.sum(B1 * B1, axis=2))
    if definition == "quadratic flux":
        return 0.5 * jnp.sum((Suu + 2.0 * u0 * trU2) * absn) / P
    if definition == "local":
        Suw = jnp.einsum("a,ap->p", lam, u1 * w1)
        Sww = jnp.einsum("a,ap->p", lam, w1 * w1)
        integrand = (Suu + 2 * u0 * trU2 - 2 * u0 / w0 * Suw + u0 ** 2 / w0 ** 2 * Sww - u0 ** 2 / w0 * trW2) / w0
        return 0.5 * jnp.sum(integrand * absn) / P
    if definition == "normalized":
        D0 = jnp.sum(w0 * absn)
        N0 = jnp.sum(u0 * u0 * absn)
        N1 = 2.0 * jnp.einsum("p,ap->a", u0 * absn, u1)
        eta1 = jnp.einsum("ap,p->a", w1, absn) / D0
        trH2 = jnp.sum(trW2 * absn) / D0
        val = (jnp.sum((Suu + 2 * u0 * trU2) * absn) - jnp.sum(lam * N1 * eta1)
               + N0 * (jnp.sum(lam * eta1 ** 2) - trH2)) / D0
        return 0.5 * val
    raise ValueError(definition)


def linear_gradient(B0, B1, nhat, absn, target, definition):
    """g_a = d f / d xi_a at 0 (first-order responses only), for control variates."""
    P = absn.shape[0]
    u0 = np.sum(B0 * nhat, axis=1) - target
    u1 = np.einsum("apk,pk->ap", B1, nhat)
    w0 = np.sum(B0 * B0, axis=1)
    w1 = 2.0 * np.einsum("apk,pk->ap", B1, B0)
    if definition == "quadratic flux":
        return np.einsum("ap,p->a", u0 * u1, absn) / P
    if definition == "local":
        return 0.5 * np.einsum("ap,p->a", 2 * u0 * u1 / w0 - u0 ** 2 * w1 / w0 ** 2, absn) / P
    if definition == "normalized":
        D0 = np.sum(w0 * absn); N0 = np.sum(u0 * u0 * absn)
        N1 = 2 * np.einsum("ap,p->a", u1, u0 * absn); D1 = np.einsum("ap,p->a", w1, absn)
        return 0.5 * (N1 / D0 - N0 * D1 / D0 ** 2)
    raise ValueError(definition)


class GaussianSquaredFlux(Optimizable):
    """f0 + Delta2 with the call pattern of SquaredFlux.

    J() = native SquaredFlux (bit-equal at sigma = 0) + Delta2 from JAX
    (include_f0=False: Delta2 alone, for the Phase 3 gradient tests)
    responses. dJ(): native SquaredFlux gradient + reverse-mode JAX gradient
    of Delta2 to (gamma, gamma', I) of every physical coil chained through
    Coil.vjp (correctness gate: G3, Phase 3). order=4 is not available.

    Lifetime (Amendment A, item A4): the law (the mode set, its variances and any sampler) and the coil topology are fixed when
    the object is built; to change them, build a new object. The coils' dofs, the currents and the target given at construction
    are the only things that move. The object keeps a private, read-only copy of the mode set and deep copies of the samplers;
    draw_samplers() returns copies. Rebinding an attribute (G.modes = ...) is outside this contract.
    """

    def __init__(self, surface, coils, modes, definition="quadratic flux", target=None,
                 order=2, path="auto", batch=None, backend="kernel", mode_chunk=8, point_chunk=None,
                 include_f0=True, check_coils=True, _spec=None):
        if order != 2:
            raise NotImplementedError("order=4 is unavailable until Phase 6 supplies its gate")
        if definition not in DEFINITIONS:
            raise ValueError(definition)
        if modes.ncoils != len(coils):
            raise ValueError("mode set and coil list disagree on the number of coils")
        _check_points(coils, modes)                      # unit 4.3 item 2 (ii)
        modes = _frozen_copy(modes)                      # Amendment A, item A4
        self.check_coils = bool(check_coils)
        if self.check_coils:                             # unit 4.3 item 2 (i)
            _check_coil_list(coils, modes)
        self.surface, self.coils, self.modes = surface, list(coils), modes
        self.include_f0 = bool(include_f0)
        self.definition = definition
        X, self.nhat, self.absn = rsp.surface_geometry(surface)
        self.X = X
        # unit 4.3 item 2 (iv): the geometry this object uses; every evaluation refuses a surface changed since construction
        self._gamma0, self._normal0 = np.array(surface.gamma(), copy=True), np.array(surface.normal(), copy=True)
        n_shape = surface.normal().shape[:2]
        # unit 4.3 item 2 (iv): the target is held as a copy (np.ascontiguousarray returned the caller's own array)
        self.target = np.zeros(n_shape) if target is None else np.array(target, dtype=float, copy=True)
        # planner's RR2 (records/phase4_3_review_fable.md §10.4): a target of the grid's size in another shape was used by delta2()
        if self.target.shape != tuple(n_shape):
            raise ValueError(f"the target has shape {self.target.shape} but the surface grid {tuple(n_shape)} "
                             f"(quadpoints_phi x quadpoints_theta) needs a target of that shape; reshaping a target of the same size "
                             f"is not done because the order of its entries cannot be known")
        self.field = BiotSavart(self.coils)
        self.native = SquaredFlux(surface, self.field, target=self.target, definition=definition)
        if path not in PATHS:
            raise ValueError(f"path must be one of {PATHS}, not {path!r}")
        self.cert = modes.certificate()
        if path == "auto":
            path = "isotropic_shortcut" if self.cert["eligible"] else "general"
        if path == "isotropic_shortcut" and not self.cert["eligible"]:
            raise ValueError("isotropic shortcut requested but the certificate fails")
        self.path = path
        nq = len(self.coils[0].curve.quadpoints)
        self.batch = batch if batch is not None else rsp.grad_batch_size(self.X.shape[0], nq, budget=1.0e7)
        self.work = None
        if backend not in ("kernel", "jvp"):
            raise ValueError("backend must be 'kernel' or 'jvp'")
        self.backend = backend
        self.mode_chunk = mode_chunk
        P = self.X.shape[0]
        # unit 1b.1: points per chunk of the kernel programs; None -> automatic (one chunk at 64x16)
        self.point_chunk = int(min(P, max(1, point_chunk if point_chunk is not None else int(4.0e5 // nq))))
        if backend == "kernel" and modes.size > 0:   # unit 4.1 item 3: no tables for an empty (zero-covariance) mode set
            idx, T, Td, kmax = kn.coil_tables(modes)
            self._kt = (jnp.asarray(idx), jnp.asarray(T), jnp.asarray(Td))
            self._kmax = kmax
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=self.coils)
        self._spec = _spec
        self._samplers = None if _spec is None else copy.deepcopy(_spec.get("samplers"))   # Amendment A, item A4
        _check_no_mode(self)                             # Amendment A, item A2
        self._law = _certify(self)                       # Amendment A, items A1 and A3 (refuses by cause, or warns)
        self._report = _report_core(self)
        for w in self._law["warnings"] + self._report.get("warnings", []):
            warnings.warn(w, UserWarning, stacklevel=2)

    # unit 4.3 item (1): the one-call constructor -------------------------------------------------------------------
    @classmethod
    def from_error_model(cls, surface, coils, nfp, stellsym, sigma=None, correlation_length=None, levels="both",
                         definition="quadratic flux", target=None, sampler=None, builder="kernel_exact", rel_cut=1e-10,
                         tail=1e-12, **options):
        """f0 + Delta2 of SquaredFlux under a Gaussian coil-error law given by its parameters.

        sigma: metres, per Cartesian component, per level. correlation_length: curve parameter, kernel exp(-d^2/l^2)
        periodized on [0,1) (sum over j = -5..5). Each is one number or a dict {level: value}. levels: "both", "systematic"
        (one error per base coil, carried to its images by the symmetry maps), "statistical" (one independent error per
        coil), or a tuple of the last two. builder: "kernel_exact" (default; realized modes of the kernel-exact
        eigen-decomposition factor mc_reference.EighSampler, n_derivs 1, at rel_cut) or "fourier" (kl_modes.fourier_modes
        at tail). sampler: an object with SIMSOPT's GaussianSampler interface, or a dict {level: sampler}; the objective then
        averages over that sampler's realized law (n_derivs 1: L L^T; n_derivs 2: its (value, first-derivative) block),
        builder is ignored, and sigma, correlation_length are read from it (if also given they must agree). definition and
        target as SquaredFlux. options go to __init__ (path, backend, mode_chunk, point_chunk, include_f0, check_coils)."""
        nfp, stellsym = int(nfp), bool(stellsym)
        rel_cut, tail = kl.check_control("rel_cut", rel_cut), kl.check_control("tail", tail)   # Amendment A, item A2
        copies = nfp * (2 if stellsym else 1)
        if len(coils) % copies:
            raise ValueError(f"{len(coils)} coils is not a multiple of nfp x (2 if stellsym) = {copies}; build the coil list "
                             f"with coils_via_symmetries(base, currents, nfp, stellsym)")
        nbase = len(coils) // copies
        qp = np.asarray(coils[0].curve.quadpoints)
        if sampler is not None:
            lv = kl.check_error_model(1.0, 1.0, levels)[0]
            per = dict(sampler) if isinstance(sampler, dict) else {lev: sampler for lev in lv}
            if set(per) != set(lv):
                raise ValueError(f"per-level samplers must cover exactly the levels {lv}")
            sig = {lev: float(per[lev].sigma) for lev in lv}
            ell = {lev: float(per[lev].length_scale) for lev in lv}
            for name, given, have in (("sigma", sigma, sig), ("correlation_length", correlation_length, ell)):
                if given is not None:
                    g = given if isinstance(given, dict) else {lev: given for lev in lv}
                    if set(g) != set(lv) or any(float(g[lev]) != have[lev] for lev in lv):
                        raise ValueError(f"{name} {given!r} disagrees with the sampler's {have}; omit it when a sampler is given")
            kl.check_error_model(sig, ell, lv)
            modes = kl.realized_modes(sampler, nfp, stellsym, nbase, levels=lv, rel_cut=rel_cut)
            spec = {"builder": "sampler", "rel_cut": rel_cut, "n_derivs": {lev: int(per[lev].n_derivs) for lev in lv},
                    "sampler_report": {lev: _sampler_report(per[lev]) for lev in lv}}
        else:
            if sigma is None or correlation_length is None:
                raise ValueError("sigma (metres per component) and correlation_length (curve parameter) are required "
                                 "unless a sampler is given")
            lv, sig, ell = kl.check_error_model(sigma, correlation_length, levels)
            scalar = not isinstance(sigma, dict) and not isinstance(correlation_length, dict)
            if builder == "kernel_exact":
                if scalar:   # Phase 4's construction (alm/stochastic_flux.py:146-147), bit for bit
                    one = mc.make_sampler(qp, sigma, correlation_length, 1, "eigh")
                    modes = kl.realized_modes(one, nfp, stellsym, nbase, levels=lv, rel_cut=rel_cut)
                    per = {lev: one for lev in lv}
                else:
                    per = {lev: mc.make_sampler(qp, sig[lev], ell[lev], 1, "eigh") for lev in lv}
                    modes = kl.realized_modes(per, nfp, stellsym, nbase, levels=lv, rel_cut=rel_cut)
                spec = {"builder": "kernel_exact", "rel_cut": rel_cut}
            elif builder == "fourier":
                modes = kl.fourier_modes(qp, nfp, stellsym, nbase, sigma, correlation_length, levels=lv, tail=tail)
                per = None   # the plain accuracy estimate builds kernel-exact factors on first use
                spec = {"builder": "fourier", "tail": tail}
            else:
                raise ValueError(f"builder must be 'kernel_exact' or 'fourier', not {builder!r}")
            spec["n_derivs"] = {lev: 1 for lev in lv}
        spec.update(levels=lv, sigma=sig, l=ell, samplers=per)
        return cls(surface, coils, modes, definition=definition, target=target, _spec=spec, **options)

    # unit 4.3 item (1): reporting -----------------------------------------------------------------------------------
    def error_model(self):
        """The error model and the numbers of the summary, as a dict. The coil-to-surface distance and its ratio are those of the
        current coils (Amendment A, item A6); the values taken at construction are kept under ..._at_construction."""
        rep = {k: v for k, v in self._report.items() if k != "warnings"}
        rep["warnings"] = list(self._law["warnings"]) + list(self._report.get("warnings", []))
        rep.update({k: copy.deepcopy(v) for k, v in self._law.items() if k != "warnings"})
        if "min_coil_surface_distance_m" in rep:
            rep["min_coil_surface_distance_m_at_construction"] = rep["min_coil_surface_distance_m"]
            rep["ratio_rms_to_distance_at_construction"] = rep["ratio_rms_to_distance"]
            dmin = _min_distance(self)
            rep["min_coil_surface_distance_m"] = dmin
            rep["ratio_rms_to_distance"] = rep["rms_formula_m"]["combined"] / dmin if dmin > 0 else float("inf")
        rep.update(_realized_rms(self, rep.get("levels")))
        rep.update({"path": self.path, "certificate_eligible": bool(self.cert["eligible"]), "max_M0_dev": self.cert["max_M0_dev"],
                    "max_M1_dev": self.cert["max_M1_dev"], "definition": self.definition, "coil_list_checked": self.check_coils,
                    "modes": int(self.modes.size)})
        return rep

    def summary(self):
        return _summary_text(self.error_model())

    def __str__(self):
        return self.summary()

    def draw_samplers(self):
        """{level: sampler} of the law the plain accuracy estimate draws from (the object's own law); copies (item A4)."""
        return copy.deepcopy(self._draw_law())

    def _draw_law(self):
        if self._samplers is None:
            if self._spec is None:
                raise ValueError("this object was built from a mode set; pass samplers={level: sampler} to accuracy_check")
            qp = np.asarray(self.coils[0].curve.quadpoints)
            self._samplers = {lev: mc.make_sampler(qp, self._spec["sigma"][lev], self._spec["l"][lev], 1, "eigh")
                              for lev in self._spec["levels"]}
        return self._samplers

    def _check_surface(self):
        """Unit 4.3 item 2 (iv): refuse a surface changed after construction (Delta2 would use the old geometry)."""
        if not (np.array_equal(self.surface.gamma(), self._gamma0) and np.array_equal(self.surface.normal(), self._normal0)):
            raise RuntimeError("the surface changed after this GaussianSquaredFlux was built; Delta2 would use the old geometry. "
                               "Build a new objective with the target computed for the new surface.")

    def __repr__(self):
        return (f"GaussianSquaredFlux(definition={self.definition!r}, modes={self.modes.size} "
                f"({self.modes.representation}), path={self.path}, certificate eligible="
                f"{self.cert['eligible']}, max M0/M1 dev={self.cert['max_M0_dev']:.1e}/"
                f"{self.cert['max_M1_dev']:.1e})")

    def certificate_record(self):
        return {"path": self.path, "backend": self.backend, "point_chunk": self.point_chunk,
                "vector_modes": int(np.sum(np.asarray(self.modes.direction) < 0)),
                "eligible": self.cert["eligible"], "rtol": self.cert["rtol"],
                "max_M0_dev": self.cert["max_M0_dev"], "max_M1_dev": self.cert["max_M1_dev"],
                "per_coil": self.cert["per_coil"]}

    def _arrays(self):
        return rsp.coil_arrays(self.coils)

    def responses(self):
        self._check_surface()
        if self.modes.size == 0:   # unit 4.1 item 3: B0 from the native field at the same points; no mode responses
            self.field.set_points(self.X)
            B0 = np.array(self.field.B())
            self.work = {"empty_mode_set": True}
            return B0, np.zeros((0,) + B0.shape), None
        g, gd, cur = self._arrays()
        general = self.path == "general"
        if self.backend == "jvp":
            B0, B1, B2, work = rsp.responses(g, gd, cur, self.X, self.modes, second=general, batch=self.batch)
            self.work = work
            return B0, B1, B2
        idx, T, Td = self._kt
        B0, B1, B2 = kn.forward(jnp.asarray(g), jnp.asarray(gd), jnp.asarray(cur), jnp.asarray(self.X), T, Td, idx,
                                self.modes.size, general, self.mode_chunk, self.point_chunk)
        self.work = {"backend": "kernel", "coils": len(g), "kmax": self._kmax, "mode_chunk": self.mode_chunk,
                     "point_chunk": self.point_chunk, "second_order": general}
        return np.asarray(B0), np.asarray(B1), (np.asarray(B2) if general else None)

    def u1(self):
        """u1^a = n . B1^a per mode, (A, P)."""
        _, B1, _ = self.responses()
        return np.einsum("apk,pk->ap", B1, self.nhat)

    def b1(self):
        """B1^a per mode, (A, P, 3)."""
        return self.responses()[1]

    def delta2(self):
        self._check_surface()
        if self.modes.size == 0:   # unit 4.1 item 3: zero covariance, Delta2 is exactly zero
            return 0.0
        if self.backend == "kernel":
            g, gd, cur = self._arrays()
            idx, T, Td = self._kt
            val = _kernel_value(jnp.asarray(g), jnp.asarray(gd), jnp.asarray(cur), jnp.asarray(self.X), T, Td, idx,
                                jnp.asarray(self.modes.lam), jnp.asarray(self.nhat), jnp.asarray(self.absn),
                                jnp.asarray(self.target.reshape(-1)), self.modes.size, self.path == "general",
                                self.mode_chunk, self.definition, self.point_chunk)
            return float(val)
        B0, B1, B2 = self.responses()
        return float(delta2_from_responses(jnp.asarray(B0), jnp.asarray(B1), jnp.asarray(self.modes.lam),
                                           None if B2 is None else jnp.asarray(B2), jnp.asarray(self.nhat),
                                           jnp.asarray(self.absn), jnp.asarray(self.target.reshape(-1)),
                                           self.definition))

    def f0(self):
        self._check_surface()
        return float(self.native.J())

    def J(self):
        return (self.f0() if self.include_f0 else 0.0) + self.delta2()

    # unit 4.3 item (3): what second order leaves out, at the current coils ------------------------------------------
    def accuracy_check(self, n_pairs=128, seed=0, estimates=("plain", "reduced"), identity=True, samplers=None, xi=None,
                       identity_batch=4):
        """Estimate R = <f> - f0 - Delta2 at the current coils, with its standard error, also over Delta2 and over f0 + Delta2.

        plain: mean over n_pairs mirrored pairs +-delta_k of 1/2 [f(+delta_k) + f(-delta_k)] - f0 - Delta2, with delta_k drawn
            from the object's own law by the reference path (mc_reference: sample_rng(seed, k, n_pairs), draw_levels,
            samples_from_arrays, perturbed_coils_levels, flux_values = SIMSOPT's SquaredFlux) and nothing from the mode
            responses. samplers={level: sampler} overrides the draw law (required for objects built from a mode set).
        reduced: Route 2 of PLAN.md section 4a, rho_k = 1/2 [f(xi_k) + f(-xi_k)] - f0 - q(xi_k), xi_k = sqrt(lam) Z_k with
            Z = Generator(PCG64DXSM(seed)).standard_normal((n_pairs, A)), f SIMSOPT's native integral of the displaced arrays
            (simsoptpp.biot_savart_B, integral_BdotN) and q(xi) = 1/2 d^2/de^2 f(e xi) at 0 of the JAX objective (exact
            second-order forward derivative). It estimates the remainder under the retained mode set's law, always (PLAN.md
            reading (x)). Far smaller error; valid only with the identity. RC5: with an explicit array xi the coefficients are
            the caller's; the result says so, and it is a remainder estimate under the retained mode set only if the caller drew
            them as xi_a = sqrt(lam_a) Z_a with Z independent standard normal variables.
        identity: Delta2 against 1/2 sum_a lam_a H_aa with H_aa the same second directional derivative along each mode
            (tolerance 1e-10 relative, section 4a); it certifies the retained law only. identity=False records "not checked".
        The result names the law of each estimate ("laws": plain, the sampler's factor it is drawn through; reduced and
        identity, the retained mode set), carries the law residual of the mode set against the stated law ("law_residual", per
        level and block, reading (v)'s scales) and "retained_is_stated": true, false, or None when the law is not certified (an
        object built from a mode set, or samplers= given). True means only that the retained mode set's covariance agrees with
        the stated kernel within 1e-6 on reading (v)'s scales: it is not equality of two laws, and it is not a bound on Delta2
        or on the remainder (RC1). speaks_for on the reduced estimate and the identity names the retained mode set, with that
        agreement and the measured residual when it holds, "retained mode set only" when it does not, "law not certified"
        without a certificate. When it is false, "omitted_delta2" gives Delta2 of the same law at the tightest admissible
        cutoff minus Delta2 as retained (a builder only: rel_cut 1e-15 for kernel_exact; for fourier the smallest tail the
        constructor accepts at the levels' l, kl.fourier_whole_table_tail, RD1; "tightest" names it); when it is true it is not computed, and that is not a bound on the
        omitted contribution: to test whether the truncation matters, compare the plain and the reduced estimates with their
        standard errors, or build again with a tighter cutoff and compare Delta2.
        Relative figures (item A5): rel_delta2 = mean / |Delta2| keeps the sign of the remainder; se_rel_delta2 = se / |Delta2| is
        never negative; rel_total uses |f0 + Delta2|; with Delta2 = 0 the relative figures are None.
        The coils and dofs are left as found. A diagnostic: benchmark numbers come from the reference code."""
        t_all = time.perf_counter()
        f0, d2 = self.f0(), self.delta2()
        out = {"seed": int(seed), "n_pairs": int(n_pairs), "f0": f0, "delta2": d2, "definition": self.definition,
               "error_model": {k: v for k, v in self._report.items() if k in ("levels", "sigma_m", "l", "builder")}}
        x_before = np.array(self.x, copy=True)
        if "plain" in estimates:
            out["plain"] = _plain_estimate(self, n_pairs, seed, f0, d2, samplers)
        if "reduced" in estimates:
            out["reduced"] = _reduced_estimate(self, n_pairs, seed, d2, xi)
        if identity:
            out["identity"] = _identity(self, d2, identity_batch)
        else:
            out["identity"] = "not checked"
        _name_laws(self, out, samplers, d2, xi is not None and "reduced" in estimates)   # item A1, R6, R7; RC1, RC5
        if not np.array_equal(np.array(self.x), x_before):
            raise RuntimeError("accuracy_check changed the dofs; this is a defect")
        out["seconds"] = time.perf_counter() - t_all
        return out

    # gradient ---------------------------------------------------------------
    @derivative_dec
    def dJ(self):
        """Native SquaredFlux gradient + Delta2 gradient chained through Coil.vjp.

        dDelta2/dB0, dDelta2/dB1 (and dB2 on the general path) come from the cheap
        assembly; they are pulled back coil by coil, the mode responses in padded
        batches of grad_batch_size(P, nq) so that reverse-mode memory is bounded.
        """
        self._check_surface()
        if self.modes.size == 0:   # unit 4.1 item 3: the native gradient, or exact zeros without f0
            if self.include_f0:
                return self.native.dJ(partials=True)
            g, gd, cur = self._arrays()
            d = None
            for j, c in enumerate(self.coils):
                v = c.vjp(np.zeros_like(g[j]), np.zeros_like(gd[j]), np.zeros(1))
                d = v if d is None else d + v
            return d
        if self.backend == "kernel":
            return self._dJ_kernel()
        g, gd, cur = self._arrays()
        B0, B1, B2 = self.responses()
        general = B2 is not None
        args = [jnp.asarray(B0), jnp.asarray(B1), jnp.asarray(B2) if general else None]
        lam, nhat, absn = jnp.asarray(self.modes.lam), jnp.asarray(self.nhat), jnp.asarray(self.absn)
        tgt = jnp.asarray(self.target.reshape(-1))
        F = lambda b0, b1, b2: delta2_from_responses(b0, b1, lam, b2, nhat, absn, tgt, self.definition)
        if general:
            cB0, cB1, cB2 = jax.grad(F, argnums=(0, 1, 2))(*args)
        else:
            cB0, cB1 = jax.grad(lambda b0, b1: F(b0, b1, None), argnums=(0, 1))(args[0], args[1])
            cB2 = jnp.zeros_like(cB1)
        X = jnp.asarray(self.X)
        P, nq = X.shape[0], g.shape[1]
        bs = rsp.grad_batch_size(P, nq)
        self.grad_work = {"batch_size": bs, "batches": 0}
        d = self.native.dJ(partials=True) if self.include_f0 else None
        for j, c in enumerate(self.coils):
            gj, gdj, Ij = jnp.asarray(g[j]), jnp.asarray(gd[j]), jnp.asarray(cur[j])
            vg, vgd, vI = (np.asarray(x) for x in rsp.coil_vjp_b0(gj, gdj, Ij, X, cB0))
            idx, dg, dgd = self.modes.tangents(j)
            for s0 in range(0, len(idx), bs):
                sl = idx[s0:s0 + bs]
                pad = bs - len(sl)
                tdg = np.concatenate([dg[s0:s0 + bs], np.zeros((pad,) + dg.shape[1:])])
                tdgd = np.concatenate([dgd[s0:s0 + bs], np.zeros((pad,) + dgd.shape[1:])])
                c1 = jnp.concatenate([cB1[sl], jnp.zeros((pad,) + cB1.shape[1:])])
                c2 = jnp.concatenate([cB2[sl], jnp.zeros((pad,) + cB2.shape[1:])])
                a1, b1, i1 = rsp.coil_vjp_modes(gj, gdj, Ij, X, jnp.asarray(tdg), jnp.asarray(tdgd), c1, c2, general)
                vg, vgd, vI = vg + np.asarray(a1), vgd + np.asarray(b1), vI + np.asarray(i1)
                self.grad_work["batches"] += 1
            v = c.vjp(vg, vgd, np.array([float(vI)]))
            d = v if d is None else d + v
        return d

    def _dJ_kernel(self):
        g, gd, cur = self._arrays()
        general = self.path == "general"
        idx, T, Td = self._kt
        G, GD, I, X = jnp.asarray(g), jnp.asarray(gd), jnp.asarray(cur), jnp.asarray(self.X)
        B0, B1, B2 = kn.forward(G, GD, I, X, T, Td, idx, self.modes.size, general, self.mode_chunk, self.point_chunk)
        lam, nhat, absn = jnp.asarray(self.modes.lam), jnp.asarray(self.nhat), jnp.asarray(self.absn)
        tgt = jnp.asarray(self.target.reshape(-1))
        if general:
            F = lambda b0, b1, b2: delta2_from_responses(b0, b1, lam, b2, nhat, absn, tgt, self.definition)
            cB0, cB1, cB2 = jax.grad(F, argnums=(0, 1, 2))(B0, B1, B2)
        else:  # Phase 2 item 1: no B2 and no zero B2 cotangent on the isotropic path
            F = lambda b0, b1: delta2_from_responses(b0, b1, lam, None, nhat, absn, tgt, self.definition)
            cB0, cB1 = jax.grad(F, argnums=(0, 1))(B0, B1)
            cB2 = None
        vg, vgd, vI = (np.asarray(x) for x in kn.pullback(G, GD, I, X, T, Td, idx, cB0, cB1, cB2, general, self.mode_chunk,
                                                                   self.point_chunk))
        self.grad_work = {"backend": "kernel", "per_coil_vjp": True, "point_chunk": self.point_chunk}
        d = self.native.dJ(partials=True) if self.include_f0 else None
        for j, c in enumerate(self.coils):
            v = c.vjp(vg[j], vgd[j], np.array([float(vI[j])]))
            d = v if d is None else d + v
        return d

    # reference only (tests): Delta2 as one JAX function of all coil arrays; its
    # full reverse-mode gradient is memory-unbounded and is not used by dJ.
    def _delta2_of_arrays(self):
        modes, X = self.modes, jnp.asarray(self.X)
        nhat, absn = jnp.asarray(self.nhat), jnp.asarray(self.absn)
        tgt = jnp.asarray(self.target.reshape(-1))
        lam = jnp.asarray(modes.lam)
        support = [modes.tangents(j) for j in range(modes.ncoils)]
        general = self.path == "general"

        def fun(G, GD, I):
            B0 = rsp.total_field(G, GD, I, X)
            B1 = jnp.zeros((modes.size, X.shape[0], 3))
            B2 = jnp.zeros((modes.size, X.shape[0], 3)) if general else None
            for j, (idx, dg, dgd) in enumerate(support):
                if len(idx) == 0:
                    continue
                f = lambda g, gd, j=j: rsp.coil_field(g, gd, I[j], X)
                _, lin = jax.linearize(f, G[j], GD[j])
                B1 = B1.at[idx].add(jax.vmap(lin)(jnp.asarray(dg), jnp.asarray(dgd)))
                if general:
                    def sec(t, td, j=j):
                        h = lambda e: rsp.coil_field(G[j] + e * t, GD[j] + e * td, I[j], X)
                        return 0.5 * jax.jacfwd(jax.jacfwd(h))(0.0)
                    B2 = B2.at[idx].add(jax.vmap(sec)(jnp.asarray(dg), jnp.asarray(dgd)))
            return delta2_from_responses(B0, B1, lam, B2, nhat, absn, tgt, self.definition)
        return fun


@partial(jax.jit, static_argnames=("A", "general", "mode_chunk", "definition", "point_chunk"))
def _kernel_value(G, GD, I, X, T, Td, idx, lam, nhat, absn, tgt, A, general, mode_chunk, definition, point_chunk=None):
    B0, B1, B2 = kn.forward(G, GD, I, X, T, Td, idx, A, general, mode_chunk, point_chunk)
    return delta2_from_responses(B0, B1, lam, B2 if general else None, nhat, absn, tgt, definition)


# ======================================================================================================================
# Unit 4.3: guards, reporting, accuracy check (records/phase4_3_proposal_opus.md sections 4-7 and 16)
# ======================================================================================================================

def _check_points(coils, modes):
    """Item 2 (ii): every coil on one set of points, the mode set built on the same points."""
    qps = [np.asarray(c.curve.quadpoints, dtype=float) for c in coils]
    if any(q.shape != qps[0].shape or not np.array_equal(q, qps[0]) for q in qps):
        sizes = sorted({len(q) for q in qps})
        raise ValueError(f"the coils do not share one set of quadrature points (points per coil: {sizes}); rebuild the base curves "
                         f"with one numquadpoints and the coil list with coils_via_symmetries")
    if modes.quadpoints is not None:
        if modes.quadpoints.shape != qps[0].shape or not np.array_equal(modes.quadpoints, qps[0]):
            raise ValueError(f"the mode set was built on other quadrature points ({len(modes.quadpoints)} points) than the coils "
                             f"({len(qps[0])}); build it from the coils' own points")
    elif modes.nq != len(qps[0]):
        raise ValueError(f"the mode set was built on {modes.nq} points but the coils have {len(qps[0])}; build it from the coils' "
                         f"own points")


def _check_coil_list(coils, modes):
    """Item 2 (i): the coil list is the symmetric set of its first coils under (nfp, stellsym) in coils_via_symmetries order."""
    copies = int(modes.nfp) * (2 if modes.stellsym else 1)
    remedy = (f"build the coil list with coils_via_symmetries(base, currents, nfp={modes.nfp}, stellsym={modes.stellsym}) and "
              f"the mode set with the same nfp and stellsym")
    if len(coils) // copies != modes.nbase:
        raise ValueError(f"{len(coils)} coils give {len(coils) // copies} base coils under nfp={modes.nfp}, "
                         f"stellsym={modes.stellsym}, but the mode set has {modes.nbase}; {remedy}")
    try:
        mc.base_curves_of(coils, modes.nfp, modes.stellsym)
    except ValueError as err:
        raise ValueError(f"{err} (symmetry check of the coil list); {remedy}") from None


# ---------------------------------------------------------------------------------------------------- Amendment A (D21)
# records/phase4_3_proposal_opus.md §17-18; review records/phase4_3_review_fable.md §15-16; PLAN.md reading (ix).

_MODE_ARRAYS = ("lam", "phi", "dphi", "direction", "level", "owner", "base_psi", "base_dpsi", "quadpoints")


def _frozen_copy(modes):
    """Item A4: a private copy of the mode set, its arrays read-only (same values, so every number is unchanged)."""
    arrays = {}
    for name in _MODE_ARRAYS:
        a = getattr(modes, name)
        if a is not None:
            a = np.array(a, copy=True)
            a.setflags(write=False)
        arrays[name] = a
    return modes._copy(info=copy.deepcopy(modes.info), **arrays)


def _model_levels(obj):
    """{level: sigma} of the object's law when it is recorded (spec or mode-set info), else None."""
    if obj._spec is not None:
        return {lev: float(obj._spec["sigma"][lev]) for lev in obj._spec["levels"]}
    parsed = _model_from_info(obj.modes)
    return None if parsed is None else {lev: float(parsed[1][lev]) for lev in parsed[0]}


def _check_no_mode(obj):
    """Item A2: a level with sigma > 0 and no retained mode would silently be the nominal objective; refused. sigma = 0 stays valid."""
    levels = _model_levels(obj)
    if not levels:
        return
    lv = np.asarray(obj.modes.level)
    for lev, sig in levels.items():
        if sig > 0 and not np.any(lv == kl.LEVELS.index(lev)):
            raise ValueError(f"level {lev} has sigma {sig:g} m > 0 but no mode: the law would be the nominal objective; check rel_cut "
                             f"and tail (each must be in [1e-15, 1))")


def _blocks_text(r):
    return f"{r['value']:.2e} (value), {r['cross']:.2e} (cross), {r['derivative']:.2e} (derivative)"


def _certify(obj):
    """Items A1 and A3 with the planner's R2, R3 (reading (ix)). Per level of a law built by a builder or from a supplied sampler:
    r = law residual of the retained mode set against the stated law; r_builder = what no admissible cutoff improves (kernel_exact:
    the eigen-factor's L L^T, every positive eigenvalue kept; fourier: the whole table m = 0 ... 200; a supplied sampler: its block
    with every positive eigenvalue kept). Builders: r_builder > 1e-6 is refused, naming the cause. Otherwise r > 1e-10 warns, naming
    the cause and the remedy; retained_is_stated = (r <= 1e-6 at every level). A mode set given directly is not certified
    (retained_is_stated None). RA2 (records/phase4_3_review_fable.md §17.4): the cause is the cutoff only when the tightest admissible
    cutoff keeps more functions and at least halves the residual; otherwise it is the factorization's precision (kernel_exact), the
    infinite periodization (fourier) or the sampler's block, and another builder is offered only when it is admitted and closer.
    RB1 (§18.4): for kernel_exact, when the stated kernel's matrix has a negative eigenvalue larger in size than the residual, the
    cause is the kernel ("kernel": not a covariance at this l; the factor is its nearest covariance), not the factorization; its
    eigenvalue is kept (kernel_min_eigenvalue) for the plain estimate's label. RC3 (§20.4): for the Fourier builder the residual of
    the infinite periodization (every coefficient above roundoff) is measured beside the whole table's; where it is small the cause
    is the table's end at m = 200 ("table"), and only otherwise the periodization; no message offers a control value or a builder
    the constructor would refuse (tail 1e-15 is replaced by the whole table, kl.fourier_whole_table_tail, where it needs M > 200;
    the other builder only when its residual at its default control is <= 1e-6). retained_is_stated True means covariance
    agreement within 1e-6 on reading (v)'s scales, not equality of laws and not a bound on Delta2 (RC1, PLAN.md reading (x))."""
    spec = obj._spec
    if spec is None:
        return {"law_certified": False, "stated_law": "not certified (mode set given)", "retained_is_stated": None,
                "law_residual": None, "law_residual_builder": None, "warnings": []}
    b = spec["builder"]
    t = obj.modes.quadpoints if obj.modes.quadpoints is not None else np.asarray(obj.coils[0].curve.quadpoints)
    res, resb, warns, causes, kmins = {}, {}, [], {}, {}
    for lev in spec["levels"]:
        sig, ell = float(spec["sigma"][lev]), float(spec["l"][lev])
        if sig == 0.0:
            res[lev] = resb[lev] = {"value": 0.0, "cross": 0.0, "derivative": 0.0, "max": 0.0}
            continue
        C = kl.retained_law(obj.modes, lev)
        if b in ("kernel_exact", "fourier"):
            K = kl.stated_kernel_matrix(t, sig, ell)
            if b == "kernel_exact":
                L = np.asarray(obj._samplers[lev].L)
                Cb = L @ L.T
            else:
                Cb = kl.fourier_table_matrix(t, sig, ell)
        else:
            smp = obj._samplers[lev]
            n, L = len(smp.points), np.asarray(smp.L)
            K = L @ L.T if int(smp.n_derivs) == 1 else L[:2 * n] @ L[:2 * n].T
            w, V = np.linalg.eigh(K)
            Cb = (V * np.maximum(w, 0.0)) @ V.T
        r, rb = kl.law_residual(C, K, sig, ell), kl.law_residual(Cb, K, sig, ell)
        res[lev], resb[lev] = r, rb
        where = f"level {lev}, l = {ell:g}, {len(t)} points"
        if b in ("kernel_exact", "fourier") and rb["max"] > kl.LAW_MATERIAL:
            var = sig ** 2 * sum(math.exp(-(j / ell) ** 2) for j in range(-5, 6))
            if b == "kernel_exact":
                kmin = float(np.linalg.eigvalsh(K)[0]) / var
                cause = (f"the eleven-image kernel is not a covariance here: the smallest eigenvalue of its (value, first-derivative) "
                         f"matrix is {kmin:.2e} of the variance, and its nearest covariance (the eigen-factor, every positive "
                         f"eigenvalue kept) differs from it by {_blocks_text(rb)}" if kmin < 0 else
                         f"the eigen-factor, every positive eigenvalue kept, differs from the eleven-image kernel by {_blocks_text(rb)}")
                try:
                    other = kl.law_residual(kl.fourier_table_matrix(t, sig, ell), K, sig, ell) if kl.fourier_cutoff(ell, 1e-12) <= kl.FOURIER_TABLE_M else None
                except ValueError:
                    other = None
                alt = "builder='fourier'"
            else:
                rper = _periodization_residual(t, sig, ell, K)
                if rper["max"] <= kl.LAW_MATERIAL:   # RC3(a): the table's end, not the periodization, leaves the residual
                    cause = (f"the Fourier table ends at m = {kl.FOURIER_TABLE_M}: the whole table differs from the eleven-image sum by "
                             f"{_blocks_text(rb)}, while every coefficient above roundoff (m = 0 ... {kl.fourier_mmax(ell)}) would "
                             f"differ by {rper['max']:.2e}; reading (v)'s reference cutoff here is M = {kl.fourier_cutoff(ell, 1e-12)}")
                else:
                    cause = (f"the Fourier builder realizes the infinite periodization, which differs here from the eleven-image sum by "
                             f"{_blocks_text(rper)} (m = 0 ... {kl.fourier_mmax(ell)}; the whole table, m = 0 ... "
                             f"{kl.FOURIER_TABLE_M}, by {rb['max']:.2e})")
                w, V = np.linalg.eigh(K)
                other = kl.law_residual((V * np.maximum(w, 0.0)) @ V.T, K, sig, ell)
                alt = "builder='kernel_exact'"
            remedy = (f"{alt} (it measures {other['max']:.2e} here)" if other is not None and other["max"] <= kl.LAW_MATERIAL else
                      "a smaller correlation length; build there and read error_model()['law_residual']")   # note N3
            raise ValueError(f"{where}: refused, the stated law is not represented to 1e-6 by builder {b!r}: {cause}. Remedy: {remedy}")
        if r["max"] > kl.LAW_EXACT:
            knob = "rel_cut" if b != "fourier" else "tail"
            control = f"{knob} {spec.get(knob):g}"
            # RA2: the cutoff is named only when the tightest admissible cutoff keeps more functions and at least halves the
            # residual; otherwise the residual is the builder's (or the sampler's) own precision and no cutoff remedy is given
            Ct, kt = _tight_law(obj, b, lev, t, sig, ell)
            rt = kl.law_residual(Ct, K, sig, ell)["max"]
            if kt > _retained_count(obj.modes, lev) and rt <= 0.5 * r["max"]:
                causes[lev] = "cutoff"
                left = kt - _retained_count(obj.modes, lev)
                tau = kl.fourier_whole_table_tail(ell) if b == "fourier" else kl.CONTROL_INTERVAL[0]
                if b == "fourier" and tau != kl.CONTROL_INTERVAL[0]:
                    # RC3(b): tail 1e-15 needs more than the table here and the constructor refuses it; compare with the whole table
                    rest = (f" (what remains with the whole table is the table's truncation at m = {kl.FOURIER_TABLE_M})"
                            if rt > kl.LAW_EXACT else "")
                    accepted = (f", which the constructor builds from tail {tau:g} (smaller tails are refused at this l)" if tau is not None
                                else "")
                    why = (f"cause: the cutoff ({control}) left out {left} functions per component that the whole table "
                           f"(m <= {kl.FOURIER_TABLE_M}){accepted} keeps, and with them the residual is {rt:.2e}{rest}; remedy: "
                           + (f"a smaller tail, down to {tau:g} at this l" if tau is not None else "a smaller tail"))
                else:
                    rest = (f" (what remains at {knob} 1e-15 is the precision of the builder at this l and number of points)"
                            if rt > kl.LAW_EXACT else "")
                    why = (f"cause: the cutoff ({control}) left out {left} functions per component that {knob} 1e-15, the smallest "
                           f"admissible value, keeps, and with them the residual is {rt:.2e}{rest}; remedy: a smaller {knob}")
            elif b == "sampler":
                causes[lev] = "sampler"
                # RA2's principle for a supplied sampler: EighSampler is offered only when the sampler is not already one
                fix = ("; it is already an eigen-factor (gaussflux.EighSampler), so this is the precision of its factorization at "
                       "this l and number of points" if isinstance(obj._samplers[lev], mc.EighSampler) else
                       "; a sampler factored more precisely (gaussflux.EighSampler) removes it")
                why = (f"cause: the sampler's block, which no cutoff improves (its nearest covariance differs from it by "
                       f"{rb['max']:.2e}){fix}")
            else:
                if b == "kernel_exact":
                    # RB1: where the stated kernel's own matrix has a negative eigenvalue larger than the residual (l ~ 1.1 to
                    # 1.5), the cause is the kernel, not the factorization's precision; the factor is its nearest covariance
                    var = sig ** 2 * sum(math.exp(-(j / ell) ** 2) for j in range(-5, 6))
                    kmin = float(np.linalg.eigvalsh(K)[0]) / var
                    if kmin < 0 and -kmin > r["max"]:
                        causes[lev] = "kernel"
                        kmins[lev] = kmin
                        what = (f"cause: the eleven-image kernel is not a covariance at this l: the smallest eigenvalue of its (value, "
                                f"first-derivative) matrix is {kmin:.2e} of the variance, and the kernel-exact factor is its nearest "
                                f"covariance (at the tightest admissible cutoff the residual is {rt:.2e}); no cutoff improves it")
                    else:
                        causes[lev] = "factorization"
                        what = (f"cause: the precision of the kernel-exact factorization at this l and number of points (at the "
                                f"tightest admissible cutoff the residual is {rt:.2e}); no cutoff improves it")
                    try:
                        other = (kl.law_residual(kl.fourier_table_matrix(t, sig, ell), K, sig, ell)
                                 if kl.fourier_cutoff(ell, 1e-12) <= kl.FOURIER_TABLE_M else None)
                    except ValueError:
                        other = None
                    alt = "builder='fourier'"
                else:
                    rper = _periodization_residual(t, sig, ell, K)
                    if rper["max"] <= 0.5 * rb["max"]:   # RC3: the table's end leaves the residual, not the periodization
                        causes[lev] = "table"
                        what = (f"cause: the Fourier table ends at m = {kl.FOURIER_TABLE_M}: the whole table differs from the "
                                f"eleven-image sum by {_blocks_text(rb)}, every coefficient above roundoff (m = 0 ... "
                                f"{kl.fourier_mmax(ell)}) would differ by {rper['max']:.2e}; no tail improves it")
                    else:
                        causes[lev] = "builder"
                        what = (f"cause: the Fourier builder realizes the infinite periodization, which differs here from the "
                                f"eleven-image sum by {_blocks_text(rper)}; no cutoff improves it")
                    w, V = np.linalg.eigh(K)
                    other = kl.law_residual((V * np.maximum(w, 0.0)) @ V.T, K, sig, ell)
                    alt = "builder='kernel_exact'"
                # the other builder is offered only when it is admitted here and measured closer (as in the refusal branch)
                closer = other is not None and other["max"] < r["max"] and other["max"] <= kl.LAW_MATERIAL
                if causes[lev] == "kernel":   # RB1: a smaller correlation length, or the other builder only if measured closer
                    why = (f"{what}; remedy: a smaller correlation length, or {alt} (it measures {other['max']:.2e} here)" if closer
                           else f"{what}; remedy: a smaller correlation length (no other builder is closer at this l and number of points)")
                else:
                    why = (f"{what}; remedy: {alt} (it measures {other['max']:.2e} here)" if closer
                           else f"{what}; no other builder is closer at this l and number of points")
            mark = ("; above 1e-6, so the reduced estimate and the identity of accuracy_check speak for the retained mode set only"
                    if r["max"] > kl.LAW_MATERIAL else "")
            stated = "the eleven-image kernel" if b != "sampler" else "the sampler's realized law"
            warns.append(f"{where}: law residual against {stated} {_blocks_text(r)} on reading (v)'s scales, above 1e-10; {why}{mark}")
    ris = all(res[lev]["max"] <= kl.LAW_MATERIAL for lev in spec["levels"])
    stated = ("the eleven-image kernel exp(-(d+j)^2/l^2), j = -5..5" if b in ("kernel_exact", "fourier")
              else "the supplied sampler's realized (value, first-derivative) law")
    return {"law_certified": True, "stated_law": stated, "retained_is_stated": bool(ris), "law_residual": res,
            "law_residual_builder": resb, "law_residual_cause": causes, "kernel_min_eigenvalue": kmins, "warnings": warns}


def _periodization_residual(t, sig, ell, K):
    """RC3: law residual of the infinite periodization (every Fourier coefficient above roundoff, m = 0 ... fourier_mmax) against
    the eleven-image sum; it separates the periodization from the end of the 0 ... 200 table."""
    return kl.law_residual(kl.fourier_table_matrix(t, sig, ell, M=kl.fourier_mmax(ell)), K, sig, ell)


def _retained_count(modes, level):
    """Functions per component of one level (owner 0, direction 0) in a mode set."""
    lev_id = kl.LEVELS.index(level)
    return int(np.sum((np.asarray(modes.level) == lev_id) & (np.asarray(modes.owner) == 0) & (np.asarray(modes.direction) == 0)))


def _tight_law(obj, b, lev, t, sig, ell):
    """RA2: (law, functions per component) of one level at the tightest admissible cutoff -- rel_cut 1e-15 for kernel_exact and a
    supplied sampler (its own factor), tail 1e-15 for the Fourier builder within its table (M <= 200) -- built exactly as the
    builder builds it."""
    lo = kl.CONTROL_INTERVAL[0]
    if b == "fourier":
        M = min(kl.fourier_cutoff(ell, lo), kl.FOURIER_TABLE_M)
        m = kl.fourier_modes(t, 1, False, 1, sig, ell, levels=("statistical",), M=M)
    else:
        m = kl.realized_modes(obj._samplers[lev], 1, False, 1, levels=("statistical",), rel_cut=lo)
    return kl.retained_law(m, "statistical"), _retained_count(m, "statistical")


def _min_distance(obj):
    from scipy.spatial import cKDTree
    pts = np.concatenate([np.asarray(c.curve.gamma()).reshape(-1, 3) for c in obj.coils])
    return float(np.min(cKDTree(obj.X).query(pts)[0]))


def _relative(mean, se, d2, total):
    """Item A5: relative figures over |Delta2| (and |f0 + Delta2|); the sign of the remainder is kept, the standard error is never
    negative; with Delta2 = 0 they are None."""
    if d2 == 0:
        return {"rel_delta2": None, "se_rel_delta2": None, "rel_total": mean / abs(total) if total else None,
                "note": "Delta2 is zero; relative figures over Delta2 undefined"}
    return {"rel_delta2": mean / abs(d2), "se_rel_delta2": se / abs(d2), "rel_total": mean / abs(total) if total else None}


def _plain_law_text(obj, samplers):
    """RC1(c): the law the plain estimate is drawn through, named by the sampler's factor."""
    law = obj._law
    if samplers is not None:
        return "the samplers given (samplers=), drawn through their own factors"
    if not law["law_certified"]:
        return "the law of draw_samplers()"
    if obj._spec["builder"] == "sampler":
        return "the supplied sampler's own factor L (draws L z): the law that sampler realizes"
    text = ("the kernel-exact eigen-factor of the eleven-image kernel's matrix (mc_reference.EighSampler: L = V sqrt(max(w, 0)), "
            "every positive eigenvalue kept), whose covariance L L^T is that matrix's nearest covariance")
    km = law.get("kernel_min_eigenvalue") or {}
    if km:
        text += ("; at this l the eleven-image kernel is not a covariance (smallest eigenvalue of its (value, first-derivative) "
                 "matrix: " + ", ".join(f"{lev} {v:.2e}" for lev, v in km.items()) + " of the variance), so the factor's law is "
                 "the kernel's nearest covariance, not the kernel")
    return text


def _name_laws(obj, out, samplers, d2, explicit_xi=False):
    """Item A1 with the planner's R6 and R7, and RC1/RC5 (records/phase4_3_review_fable.md §20.4; PLAN.md reading (x)): the law each
    estimate is drawn from (plain: the sampler's factor; reduced and identity: the retained mode set, always), the law residual,
    retained_is_stated (None without a certificate or with samplers given; True = covariance agreement within 1e-6 on reading (v)'s
    scales, not equality of laws, not a bound on Delta2 or on the remainder), and omitted_delta2 when that agreement fails."""
    law = obj._law
    ris = law["retained_is_stated"] if (law["law_certified"] and samplers is None) else None
    out["laws"] = {"plain": _plain_law_text(obj, samplers), "reduced": "retained mode set", "identity": "retained mode set"}
    out["law_residual"] = copy.deepcopy(law["law_residual"])
    out["retained_is_stated"] = ris
    if ris:
        rmax = max(v["max"] for v in law["law_residual"].values())
        stated = "the eleven-image kernel" if obj._spec["builder"] != "sampler" else "the supplied sampler's realized law"
        speaks = (f"retained mode set (its covariance agrees with {stated} within 1e-6 on reading (v)'s scales, law residual "
                  f"{rmax:.2e}; not equality of laws, not a bound on Delta2 or on the remainder)")
    else:
        speaks = {False: "retained mode set only", None: "law not certified"}[ris]
    for key in ("reduced", "identity"):
        if isinstance(out.get(key), dict):
            out[key]["speaks_for"] = speaks
    if explicit_xi and isinstance(out.get("reduced"), dict):   # RC5: the draws are the caller's
        out["reduced"]["speaks_for"] = (f"the caller's coefficients (xi given): a remainder estimate under the retained mode set only if "
                                        f"the caller drew them as xi_a = sqrt(lam_a) Z_a with Z independent standard normal variables; "
                                        f"the mode set: {speaks}")
    if ris is None:
        out["omitted_delta2"] = "not computed: the law is not certified"
        return
    if ris:
        out["omitted_delta2"] = ("not computed: the law residual is at most 1e-6 (the retained mode set's covariance agrees with the "
                                 "stated law within 1e-6 on reading (v)'s scales); this is not a bound on the omitted contribution. To "
                                 "measure it, build again with a tighter cutoff and compare Delta2, or compare the plain and the reduced "
                                 "estimates with their standard errors")
        return
    rmax = max(v["max"] for v in law["law_residual"].values())
    out["warning"] = (f"the retained mode set's covariance differs from the stated law by more than 1e-6 (law residual {rmax:.2e}): the "
                      f"plain estimate is drawn through the sampler's factor (laws['plain']), the reduced estimate and the identity "
                      f"speak for the retained mode set only, and their expectations can differ")
    spec = obj._spec
    if spec["builder"] not in ("kernel_exact", "fourier"):
        out["omitted_delta2"] = "not available for a supplied sampler"
        return
    if spec["builder"] == "kernel_exact":
        tight = {"rel_cut": kl.CONTROL_INTERVAL[0]}
    else:
        # RD1 (records/phase4_3_review_fable.md §21.5): the smallest tail the constructor accepts at every level's l (tail 1e-15 needs
        # M > 200 at short l and is refused there); one tail serves all levels, so the largest of the levels' smallest tails
        taus = {lev: kl.fourier_whole_table_tail(spec["l"][lev]) for lev in spec["levels"]}
        if any(t is None for t in taus.values()):
            out["omitted_delta2"] = ("not available: no tail the constructor accepts was found for level(s) "
                                     + ", ".join(lev for lev, t in taus.items() if t is None))
            return
        tight = {"tail": max(taus.values())}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            T = GaussianSquaredFlux.from_error_model(obj.surface, obj.coils, obj.modes.nfp, obj.modes.stellsym, sigma=dict(spec["sigma"]),
                                                     correlation_length=dict(spec["l"]), levels=tuple(spec["levels"]),
                                                     definition=obj.definition, target=obj.target, builder=spec["builder"],
                                                     check_coils=obj.check_coils, **tight)
        d2t = T.delta2()
        out["omitted_delta2"] = {"absolute": d2t - d2, "over_abs_delta2": (d2t - d2) / abs(d2) if d2 else None, "tightest": tight,
                                 "delta2_tightest": d2t}
    except ValueError as err:
        out["omitted_delta2"] = f"not available: {err}"


def _rms_formula(sigma, l):
    return float(sigma) * math.sqrt(sum(math.exp(-(j / l) ** 2) for j in range(-5, 6)))


def _model_from_info(modes):
    """(levels, sigma, l, builder dict) read from a mode set's info, or None when it does not record them."""
    info = modes.info
    if "subset" in info:
        return None
    scale = math.sqrt(info["scaled_by"]) if "scaled_by" in info else 1.0
    present = tuple(kl.LEVELS[i] for i in sorted(set(int(v) for v in np.asarray(modes.level))))
    if modes.representation == "fourier":
        lv = tuple(info["sigma"])
        return lv, {k: scale * v for k, v in info["sigma"].items()}, dict(info["l"]), {"builder": "fourier", "tail": info.get("tail")}
    if "per_level" in info:
        per = info["per_level"]
        lv = tuple(per)
        return lv, {k: scale * per[k]["sigma"] for k in lv}, {k: per[k]["l"] for k in lv}, {"builder": "realized", "rel_cut": per[lv[0]]["rel_cut"]}
    if "sigma" in info and present:
        return present, {k: scale * info["sigma"] for k in present}, {k: info["l"] for k in present}, {"builder": "realized", "rel_cut": info.get("rel_cut")}
    return None


def _kernel_rms(sigma, l):
    """Kernel rms per component of the value and of the first derivative: sigma (sum_j e^{-j^2/l^2})^1/2 and
    sigma (sum_j (2/l^2 - 4 j^2/l^4) e^{-j^2/l^2})^1/2 (= sqrt(-k''(0)) of k(d) = sigma^2 sum_j e^{-(d+j)^2/l^2}), j = -5..5."""
    v = sum(math.exp(-(j / l) ** 2) for j in range(-5, 6))
    d = sum((2.0 / l ** 2 - 4.0 * j * j / l ** 4) * math.exp(-(j / l) ** 2) for j in range(-5, 6))
    return float(sigma) * math.sqrt(v), float(sigma) * math.sqrt(max(d, 0.0))


def _sampler_report(sampler):
    """Planner's ruling on R2 (b) (records/phase4_3_review_fable.md §9): the law a given sampler realizes, described beside its kernel.
    The (value, first-derivative) block C of L L^T (all of it for n_derivs 1): largest and smallest eigenvalue over the kernel's
    pointwise variance; realized rms of value and first derivative (min-max over the coil, from diag C) against the kernel's."""
    n = len(sampler.points)
    L = np.asarray(sampler.L)
    C = L @ L.T if int(sampler.n_derivs) == 1 else L[:2 * n] @ L[:2 * n].T
    kv, kd = _kernel_rms(sampler.sigma, sampler.length_scale)
    var = kv ** 2
    rep = {"n_derivs": int(sampler.n_derivs), "points": n, "kernel_rms_value_m": kv, "kernel_rms_derivative_m": kd}
    if var == 0.0:
        rep["zero_variance"] = True
        return rep
    w = np.linalg.eigvalsh(C)
    dg = np.diag(C)
    rv, rd = np.sqrt(np.maximum(dg[:n], 0.0)), np.sqrt(np.maximum(dg[n:2 * n], 0.0))
    rep.update(lam_max_over_var=float(w.max() / var), lam_min_over_var=float(w.min() / var),
               realized_rms_value_m=[float(rv.min()), float(rv.max())], realized_rms_derivative_m=[float(rd.min()), float(rd.max())])
    return rep


def _report_core(obj):
    """Numbers computed once at construction: the error model, rms displacement by the formula, smallest coil-to-surface
    distance and the ratio, resolution, and the warnings of items 2 (v) and (vi)."""
    rep = {"warnings": []}
    try:
        modes, nq = obj.modes, int(obj.modes.nq)
        # planner's RR3 (records/phase4_3_review_fable.md §10.4): the stated symmetry decides which coils share a systematic error
        rep.update(nfp=int(modes.nfp), stellsym=bool(modes.stellsym), base_coils=int(modes.nbase), coils=len(obj.coils),
                   surface_nfp=getattr(obj.surface, "nfp", None), surface_stellsym=getattr(obj.surface, "stellsym", None))
        if obj._spec is not None:
            lv, sig, ell = obj._spec["levels"], obj._spec["sigma"], obj._spec["l"]
            b = {k: obj._spec[k] for k in ("builder", "rel_cut", "tail", "n_derivs") if k in obj._spec}
        else:
            parsed = _model_from_info(modes)
            if parsed is None:
                rep.update(levels=None, note="error model not recorded by this mode set (subset)")
                return rep
            lv, sig, ell, b = parsed
        rep.update(levels=list(lv), sigma_m=dict(sig), l=dict(ell), points_per_coil=nq, **b)
        rms = {k: _rms_formula(sig[k], ell[k]) for k in lv}
        rms["combined"] = math.sqrt(sum(rms[k] ** 2 for k in lv))
        rep["rms_formula_m"] = rms
        res = {}
        for k in lv:
            M = kl.fourier_cutoff(ell[k], 1e-12)
            lev_id = kl.LEVELS.index(k)
            nfun = int(np.sum((np.asarray(modes.level) == lev_id) & (np.asarray(modes.owner) == 0) & (np.asarray(modes.direction) == 0)))
            # Amendment A, item A6: the reference cutoff (tail 1e-12) and, for the Fourier builder, the retained one
            Mret = modes.info.get("cutoff", {}).get(k) if modes.representation == "fourier" else None
            res[k] = {"cutoff_M": int(M), "reference_cutoff_M": int(M), "retained_cutoff_M": None if Mret is None else int(Mret),
                      "functions_2M_plus_1": int(2 * M + 1), "modes": int(np.sum(np.asarray(modes.level) == lev_id)),
                      "functions_per_component": nfun}
            if b.get("builder") == "fourier" and Mret is not None and 2 * Mret + 1 > nq:
                rep["warnings"].append(f"level {k}: the Fourier builder keeps 2M+1 = {2 * Mret + 1} harmonic functions (l = {ell[k]}) "
                                       f"on {nq} points per coil, more than the points represent; the value is not affected in the "
                                       f"cases measured but the cost grows; use at least {2 * Mret + 1} points per coil, a longer "
                                       f"correlation length, or builder='kernel_exact'")
        rep["resolution"] = res
        srep = (obj._spec or {}).get("sampler_report")
        if srep:
            info = modes.info
            for k, r in srep.items():
                tr = info["per_level"][k].get("truncation_residual_rel") if "per_level" in info else info.get("truncation_residual_rel")
                r["truncation_residual_rel"] = tr
                if r.get("zero_variance"):
                    continue
                if r["lam_min_over_var"] < -1e-10:
                    # planner's RN1 (§10.5): the block is PSD in exact arithmetic; a negative eigenvalue measures the roundoff of forming it
                    rep["warnings"].append(f"level {k}: the sampler's covariance block is determined only to about "
                                           f"{abs(r['lam_min_over_var']):.1e} of the kernel variance (its smallest computed eigenvalue is "
                                           f"{r['lam_min_over_var']:.2e}, roundoff at its largest, {r['lam_max_over_var']:.2e}); the mode set "
                                           f"represents it to that precision")
                lo, hi = r["realized_rms_value_m"]
                kv = r["kernel_rms_value_m"]
                if lo < 0.5 * kv or hi > 2.0 * kv:
                    rep["warnings"].append(f"level {k}: the sampler's draws have rms {1e3 * lo:.4g}-{1e3 * hi:.4g} mm where its kernel gives "
                                           f"{1e3 * kv:.4g} mm; it does not realize its kernel")
            rep["sampler_report"] = srep
        from scipy.spatial import cKDTree
        pts = np.concatenate([np.asarray(c.curve.gamma()).reshape(-1, 3) for c in obj.coils])
        dmin = float(np.min(cKDTree(obj.X).query(pts)[0]))
        rep["min_coil_surface_distance_m"] = dmin
        rep["ratio_rms_to_distance"] = rms["combined"] / dmin if dmin > 0 else float("inf")
        if rep["ratio_rms_to_distance"] >= 1.0:
            rep["warnings"].append(f"rms displacement {rms['combined']:.3g} m >= the smallest coil-surface distance {dmin:.3g} m: "
                                   f"check that sigma is in metres per component. This is a check of unit and scale, not a "
                                   f"validity test; use accuracy_check()")
    except Exception as err:   # reporting never stops an evaluation
        rep["report_error"] = f"{type(err).__name__}: {err}"
    return rep


def _realized_rms(obj, levels):
    """Realized rms displacement per component on coil 0, per level: min and max over t of (sum_a lam_a |psi_a(t)|^2 / 3)^1/2."""
    if not levels:
        return {}
    out = {}
    for k in levels:
        lev_id = kl.LEVELS.index(k)
        v = np.zeros(obj.modes.nq)
        for a in np.where(np.asarray(obj.modes.level) == lev_id)[0]:
            w = obj.modes.on_coil(int(a), 0)
            if w is not None:
                v += obj.modes.lam[a] * np.sum(w[0] ** 2, axis=1)
        r = np.sqrt(v / 3.0)
        out[k] = [float(r.min()), float(r.max())]
    return {"rms_realized_m": out}


def _summary_text(em):
    lines = [f"GaussianSquaredFlux: f0 + Delta2 of SquaredFlux ({em['definition']!r}), order 2"]
    if "nfp" in em:   # planner's RR3 (§10.4)
        sym = (f"  stated symmetry: nfp {em['nfp']}, stellsym {em['stellsym']}; {em['base_coils']} base coils of {em['coils']} "
               f"(each base coil and its images share one systematic error)")
        if em.get("surface_nfp") is not None and (em["surface_nfp"], em["surface_stellsym"]) != (em["nfp"], em["stellsym"]):
            sym += f"; differs from the surface's nfp {em['surface_nfp']}, stellsym {em['surface_stellsym']}"
        lines.append(sym)
    if em.get("report_error"):   # planner's RN3 (§10.5)
        lines.append(f"  report failed: {em['report_error']}")
    if em.get("law_certified") is False:   # Amendment A, R7
        lines.append("  law not certified (mode set given): retained_is_stated None")
    elif em.get("law_residual"):           # Amendment A, item A1
        for k, r in em["law_residual"].items():
            lines.append(f"  law residual against the stated law ({k}): {_blocks_text(r)}")
        lines.append("  retained mode set: covariance within 1e-6 of the stated law on reading (v)'s scales (law residual <= 1e-6); "
                     "not equality of laws, not a bound on Delta2" if em.get("retained_is_stated") else
                     "  retained mode set: covariance differs from the stated law by more than 1e-6 (law residual > 1e-6): the reduced "
                     "estimate and the identity speak for the retained mode set only")
    if not em.get("levels"):
        lines.append(f"  error model: {em.get('note', 'not recorded')}; {em['modes']} modes")
    else:
        b = em.get("builder")
        bdesc = {"kernel_exact": f"kernel_exact (rel_cut {em.get('rel_cut')})", "fourier": f"fourier (tail {em.get('tail')})",
                 "sampler": f"sampler's realized law (rel_cut {em.get('rel_cut')}, n_derivs {em.get('n_derivs')})",
                 "realized": f"realized modes of a sampler (rel_cut {em.get('rel_cut')})"}.get(b, str(b))
        lines.append(f"  error model: kernel exp(-d^2/l^2) periodized on [0,1); builder {bdesc}")
        lines.append("  level        sigma [mm]  l       modes  functions/component  reference kernel cutoff M (2M+1)  points/coil")
        for k in em["levels"]:
            r = em["resolution"][k]
            ret = "" if r.get("retained_cutoff_M") is None else f"  (retained M {r['retained_cutoff_M']})"
            lines.append(f"  {k:<12} {1e3 * em['sigma_m'][k]:<11.4g} {em['l'][k]:<7.4g} {r['modes']:>5}  {r['functions_per_component']:<20} "
                         f"{r['cutoff_M']} ({r['functions_2M_plus_1']}){'':<26} {em['points_per_coil']}{ret}")
        rf, rr = em["rms_formula_m"], em.get("rms_realized_m", {})
        parts = [f"{k} {1e3 * rf[k]:.4g} (realized {1e3 * rr[k][0]:.4g}-{1e3 * rr[k][1]:.4g})" for k in em["levels"] if k in rr]
        lines.append(f"  rms displacement per component [mm]: {', '.join(parts)}, combined {1e3 * rf['combined']:.4g}")
        for k, r in (em.get("sampler_report") or {}).items():
            if r.get("zero_variance"):
                continue
            lines.append(f"  sampler ({k}, n_derivs {r['n_derivs']}): rms value {1e3 * r['realized_rms_value_m'][0]:.4g}-"
                         f"{1e3 * r['realized_rms_value_m'][1]:.4g} mm (kernel {1e3 * r['kernel_rms_value_m']:.4g}), first derivative "
                         f"{1e3 * r['realized_rms_derivative_m'][0]:.4g}-{1e3 * r['realized_rms_derivative_m'][1]:.4g} mm (kernel "
                         f"{1e3 * r['kernel_rms_derivative_m']:.4g}); block eigenvalues / kernel variance: max {r['lam_max_over_var']:.3g}, "
                         f"min {r['lam_min_over_var']:.3g}; truncation residual {r.get('truncation_residual_rel')}")
        if "min_coil_surface_distance_m" in em:
            lines.append(f"  smallest coil-surface distance [m] (now): {em['min_coil_surface_distance_m']:.4g}; combined rms / "
                         f"distance: {em['ratio_rms_to_distance']:.3g}")
    lines.append(f"  path {em['path']}; certificate {'eligible' if em['certificate_eligible'] else 'NOT eligible'} "
                 f"(max M0/M1 dev {em['max_M0_dev']:.1e}/{em['max_M1_dev']:.1e})"
                 + ("" if em["coil_list_checked"] else "; coil list not checked"))
    for w in em.get("warnings", []):
        lines.append(f"  WARNING: {w}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------- accuracy check

def _stats(v):
    v = np.asarray(v, dtype=float)
    return float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float("nan")


def _plain_estimate(obj, n_pairs, seed, f0, d2, samplers):
    t0 = time.perf_counter()
    smp = obj._draw_law() if samplers is None else dict(samplers)
    modes = obj.modes
    base = mc.base_curves_of(obj.coils, modes.nfp, modes.stellsym)
    nbase, ncoils = len(base), len(obj.coils)
    fp, fm = np.empty(n_pairs), np.empty(n_pairs)
    for k in range(n_pairs):
        sy, st = mc.draw_levels(smp, nbase, ncoils, mc.sample_rng(seed, k, n_pairs))
        vals = []
        for sgn in (1.0, -1.0):
            psy = None if sy is None else mc.samples_from_arrays(smp["systematic"], mc.sample_arrays(sy), sgn)
            pst = None if st is None else mc.samples_from_arrays(smp["statistical"], mc.sample_arrays(st), sgn)
            pc = mc.perturbed_coils_levels(obj.coils, base, modes.nfp, modes.stellsym, psy, pst)
            vals.append(mc.flux_values(obj.surface, pc, obj.target, (obj.definition,))[obj.definition])
        fp[k], fm[k] = vals
    r = 0.5 * (fp + fm) - f0 - d2
    mean, se = _stats(r)
    return {"mean": mean, "se": se, **_relative(mean, se, d2, f0 + d2), "per_pair": r.tolist(), "f_plus": fp.tolist(), "f_minus": fm.tolist(),
            "levels": sorted(smp), "seconds": time.perf_counter() - t0}


def _native_flux(obj):
    """f(xi): SIMSOPT's native integral (SquaredFlux.J's arithmetic) of the arrays displaced by sum_a xi_a psi_a."""
    import simsoptpp as sopp
    g, gd, cur = rsp.coil_arrays(obj.coils)
    g, gd = np.asarray(g), np.asarray(gd)
    cur = [float(c) for c in np.asarray(cur)]
    tabs = [obj.modes.tangents(j) for j in range(len(g))]
    n = obj.surface.normal()
    X = np.ascontiguousarray(obj.X)

    def f(xi):
        dG, dGD = np.zeros_like(g), np.zeros_like(gd)
        for j, (idx, dg, dgd) in enumerate(tabs):
            if len(idx):
                dG[j] = np.einsum("a,aqk->qk", xi[idx], dg)
                dGD[j] = np.einsum("a,aqk->qk", xi[idx], dgd)
        B = sopp.biot_savart_B(X, [np.ascontiguousarray(x) for x in g + dG], [np.ascontiguousarray(x) for x in gd + dGD], cur)
        return float(sopp.integral_BdotN(np.ascontiguousarray(B.reshape(n.shape)), obj.target, n, obj.definition))
    return f, g, gd, np.asarray(cur), tabs


def _field_scan(G, GD, I, X):
    """Sum of coil fields, one coil at a time (memory of one coil's (P, nq, 3) intermediate, also under nested forward AD)."""
    def body(acc, c):
        return acc + rsp.coil_field(c[0], c[1], c[2], X), None
    B, _ = jax.lax.scan(body, jnp.zeros_like(X), (G, GD, I))
    return B


@partial(jax.jit, static_argnames=("definition",))
def _second_directional(Brest, Gt, GDt, It, Tt, TDt, X, nhat, absn, target, definition):
    """1/2 d^2/de^2 f at e = 0 for B(e) = Brest + sum over the given coils of their field displaced by e (Tt, TDt)."""
    def F(e):
        return rsp.objective(Brest + _field_scan(Gt + e * Tt, GDt + e * TDt, It, X), nhat, absn, target, definition)
    d1 = lambda e: jax.jvp(F, (e,), (jnp.ones_like(e),))[1]
    zero = jnp.zeros((), dtype=X.dtype)
    return 0.5 * jax.jvp(d1, (zero,), (jnp.ones_like(zero),))[1]


def _reduced_estimate(obj, n_pairs, seed, d2, xi):
    t0 = time.perf_counter()
    lam = np.asarray(obj.modes.lam)
    A = lam.size
    explicit = xi is not None
    if xi is None:
        Z = np.random.Generator(np.random.PCG64DXSM(seed)).standard_normal((n_pairs, A))
        xi = np.sqrt(lam)[None, :] * Z
    xi = np.asarray(xi, dtype=float)
    f, g, gd, cur, tabs = _native_flux(obj)
    f0n = f(np.zeros(A))
    X, nhat, absn = jnp.asarray(obj.X), jnp.asarray(obj.nhat), jnp.asarray(obj.absn)
    tgt = jnp.asarray(obj.target.reshape(-1))
    G, GD, I = jnp.asarray(g), jnp.asarray(gd), jnp.asarray(cur)
    zeroB = jnp.zeros_like(X)
    rho, q = np.empty(len(xi)), np.empty(len(xi))
    for k, x in enumerate(xi):
        T, TD = np.zeros_like(g), np.zeros_like(gd)
        for j, (idx, dg, dgd) in enumerate(tabs):
            if len(idx):
                T[j] = np.einsum("a,aqk->qk", x[idx], dg)
                TD[j] = np.einsum("a,aqk->qk", x[idx], dgd)
        q[k] = float(_second_directional(zeroB, G, GD, I, jnp.asarray(T), jnp.asarray(TD), X, nhat, absn, tgt, obj.definition))
        rho[k] = 0.5 * (f(x) + f(-x)) - f0n - q[k]
    mean, se = _stats(rho)
    return {"mean": mean, "se": se, **_relative(mean, se, d2, f0n + d2), "per_pair": rho.tolist(), "q": q.tolist(), "f0_native": f0n,
            "draws": "explicit" if explicit else "sqrt(lam) * PCG64DXSM(seed).standard_normal((n_pairs, A))",
            "seconds": time.perf_counter() - t0}


def _identity(obj, d2, batch):
    """Delta2 against 1/2 sum_a lam_a H_aa, H_aa the exact second directional derivative of the JAX objective along mode a;
    only the coils mode a moves are displaced (the others enter through B0), modes grouped by the number of coils they move."""
    t0 = time.perf_counter()
    modes = obj.modes
    g, gd, cur = (np.asarray(v) for v in rsp.coil_arrays(obj.coils))
    X, nhat, absn = jnp.asarray(obj.X), jnp.asarray(obj.nhat), jnp.asarray(obj.absn)
    tgt = jnp.asarray(obj.target.reshape(-1))
    B0 = _field_scan(jnp.asarray(g), jnp.asarray(gd), jnp.asarray(cur), X)
    groups = {}
    for a in range(modes.size):
        sup = [(j, modes.on_coil(a, j)) for j in range(modes.ncoils)]
        sup = [(j, v) for j, v in sup if v is not None]
        groups.setdefault(len(sup), []).append((a, sup))
    H = np.zeros(modes.size)
    fn = jax.vmap(lambda Br, Gt, GDt, It, Tt, TDt: _second_directional(Br, Gt, GDt, It, Tt, TDt, X, nhat, absn, tgt, obj.definition))
    for kcoils, items in groups.items():
        for s0 in range(0, len(items), batch):
            chunk = items[s0:s0 + batch]
            idx = [[j for j, _ in sup] for _, sup in chunk]
            Gt = np.stack([g[i] for i in idx]); GDt = np.stack([gd[i] for i in idx]); It = np.stack([cur[i] for i in idx])
            Tt = np.stack([np.stack([v[0] for _, v in sup]) for _, sup in chunk])
            TDt = np.stack([np.stack([v[1] for _, v in sup]) for _, sup in chunk])
            Brest = jnp.stack([B0 - _field_scan(jnp.asarray(Gt[m]), jnp.asarray(GDt[m]), jnp.asarray(It[m]), X) for m in range(len(chunk))])
            val = fn(Brest, jnp.asarray(Gt), jnp.asarray(GDt), jnp.asarray(It), jnp.asarray(Tt), jnp.asarray(TDt))
            for m, (a, _) in enumerate(chunk):
                H[a] = 2.0 * float(val[m])     # _second_directional returns 1/2 f''
    tr = 0.5 * float(np.asarray(modes.lam) @ H)
    return {"half_trace_Lambda_H": tr, "delta2": d2, "rel_diff": (tr - d2) / d2 if d2 else float("nan"),
            "pass": bool(abs(tr - d2) <= 1e-10 * abs(d2)) if d2 else bool(tr == 0.0), "tol": 1e-10,
            "seconds": time.perf_counter() - t0}
