"""Karhunen-Loeve mode sets for the two-level coil error model (PLAN.md §2, §3.1).

A mode a is a scalar function phi_a(t) on the coil quadrature grid, its
derivative-channel function dphi_a(t), a Cartesian direction d_a, a level
(systematic or statistical), an owner (base coil for systematic, physical coil
for statistical) and a variance lam_a (sigma^2 included). The displacement it
induces on physical coil j is

    Xi_j(t)  = A_j e_d phi_a(t),   Xi'_j(t) = A_j e_d dphi_a(t),

with A_j = the image map M_j = F R_z(phi_k) of coil j (systematic modes, coil j
an image of the owner; read in curve.py:686-707, x -> F R_z x on column
vectors) or the identity (statistical modes, which act in the lab frame on the
owner only). Lambda is diagonal in this basis.

Representations:
  * "fourier": the kernel's KL basis (write-up eq. 47 in parameter units):
    c_m = l sqrt(pi) exp(-pi^2 m^2 l^2), Var A_d0 = sigma^2 c_0,
    Var A_dm = Var B_dm = 2 sigma^2 c_m; dphi = d phi / dt exactly; truncation
    M = smallest integer with sum_{m>M} 2 c_m / (c_0 + 2 sum_{m>=1} c_m) <= tail.
  * "realized": eigenpairs of the sampler's own C = L L^T (value and
    derivative blocks jointly, one 2n x 2n problem shared by the three
    Cartesian components); every eigenvalue > rel_cut * largest is kept
    (default 1e-13: 1e-15 keeps 50-80 roundoff-level eigenvectors per
    component; 1e-13 keeps 7-8 at l = 0.5 with block residual <= 5e-13). Here
    dphi is NOT d phi / dt.
Every mode is normalized so that max_t |phi_a| = 1 (lam rescaled), except a
mode with negligible value channel, normalized by its derivative channel.
"""

import math
from decimal import Decimal
from fractions import Fraction
from dataclasses import dataclass, field

import numpy as np

LEVELS = ("systematic", "statistical")


def check_error_model(sigma, l, levels=LEVELS):
    """Unit 4.3 item 2 (iii): validate an error model and return (levels, {level: sigma}, {level: l}).

    levels: "both", "systematic", "statistical", or a tuple/list of the last two (no repeats). sigma (metres, per Cartesian
    component, per level) and l (curve parameter, kernel exp(-d^2/l^2) periodized on [0,1)) are each one number or a dict with
    exactly the keys of levels. sigma must be finite and >= 0 (0 is a valid zero-covariance model); l finite and > 0."""
    if isinstance(levels, str):
        if levels == "both":
            levels = LEVELS
        elif levels in LEVELS:
            levels = (levels,)
        else:
            raise ValueError(f"unknown error level {levels!r}: use 'both', 'systematic', 'statistical' or a tuple of the last two")
    levels = tuple(levels)
    if not levels or any(lev not in LEVELS for lev in levels) or len(set(levels)) != len(levels):
        raise ValueError(f"error levels must be a nonempty, repeat-free subset of {LEVELS}, not {levels!r}")
    out = []
    for name, value, ok, rule in (("sigma", sigma, lambda v: v >= 0, "finite and >= 0, in metres per component"),
                                  ("correlation length", l, lambda v: v > 0, "finite and > 0, in the curve parameter on [0,1)")):
        if isinstance(value, dict):
            if set(value) != set(levels):
                raise ValueError(f"per-level {name} must cover exactly the levels {levels}, not {sorted(value)}")
            d = dict(value)
        else:
            d = {lev: value for lev in levels}
        for lev, v in d.items():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                raise ValueError(f"{name} of level {lev!r} is not a number: {v!r}") from None
            if not math.isfinite(fv) or not ok(fv):
                raise ValueError(f"{name} of level {lev!r} must be {rule}; got {v!r}")
        out.append(d)
    return levels, out[0], out[1]


# Amendment A (records/phase4_3_proposal_opus.md §17-18; PLAN.md reading (ix)) ------------------------------------------------
CONTROL_INTERVAL = (1e-15, 1.0)   # rel_cut and tail: real, finite, in [1e-15, 1)
FOURIER_TABLE_M = 200             # the Fourier builder's coefficient table, m = 0 ... 200
LAW_EXACT, LAW_MATERIAL = 1e-10, 1e-6   # law residual: the gate's tolerance; above it the retained law is not the stated one


def _is_real_scalar(value):
    """RC2 (records/phase4_3_review_fable.md §20.4): decided by type, before any conversion. Real scalars: Python int and float,
    Decimal, Fraction, a real NumPy integer or floating scalar, a 0-d array of a real integer or floating dtype. Everything else is
    not: bool in every container, every complex type whatever its imaginary part, text and bytes in every container (str, bytes,
    bytearray, memoryview, np.str_, np.bytes_, arrays of those dtypes), object arrays, arrays with one or more dimensions, None."""
    if isinstance(value, (bool, np.bool_)):
        return False
    if isinstance(value, np.ndarray):
        return value.ndim == 0 and value.dtype.kind in "iuf"
    if isinstance(value, np.generic):
        return value.dtype.kind in "iuf"
    return isinstance(value, (int, float, Decimal, Fraction))


def check_control(name, value):
    """Item A2: rel_cut and tail are real finite numbers in [1e-15, 1). RC2: whether the value is a real scalar is decided by its
    type before any conversion (_is_real_scalar); so a bool, None, a string, bytes, a memoryview, a complex value (NumPy's too, even
    with a zero imaginary part), a textual, byte or object array and an array with dimensions are refused (RA1: also a numeric
    string such as "1e-10", which float() would otherwise accept)."""
    lo, hi = CONTROL_INTERVAL
    ok = _is_real_scalar(value)
    try:
        v = float(value) if ok else float("nan")
    except (TypeError, ValueError, OverflowError, ArithmeticError):
        ok, v = False, float("nan")
    if not (ok and math.isfinite(v) and lo <= v < hi):
        raise ValueError(f"{name} must be a real number in [1e-15, 1); got {value!r}")
    return v


def fourier_coefficients(l, mmax=200):
    m = np.arange(mmax + 1)
    return l * np.sqrt(np.pi) * np.exp(-np.pi ** 2 * m ** 2 * l ** 2)


def fourier_cutoff(l, tail=1e-12):
    """Smallest M with sum_{m>M} 2 c_m / (c_0 + 2 sum_{m>=1} c_m) <= tail. Item A3: the table reaches m_max = max(200,
    ceil(6 / (pi l))), so every coefficient above roundoff (exp(-36 pi^-2 ...) < 3e-16 c_0) enters; for l >= 6 / (200 pi) the table
    and so M are those of 0 ... 200, entry for entry. The result may exceed 200; the Fourier builder refuses that case."""
    mmax = max(200, int(math.ceil(6.0 / (math.pi * float(l)))))
    if mmax > 2_000_000:
        raise ValueError(f"correlation length {l!r} is too small for the Fourier cutoff (it needs more than 2e6 coefficients)")
    c = fourier_coefficients(l, mmax)
    total = c[0] + 2 * c[1:].sum()
    if mmax == 200:   # the table of 0 ... 200 and the arithmetic of the code before Amendment A
        for M in range(len(c)):
            if 2 * c[M + 1:].sum() / total <= tail:
                return M
        raise ValueError("cutoff not reached")
    tails = 2 * (np.cumsum(c[::-1])[::-1] - c) / total
    hit = np.nonzero(tails <= tail)[0]
    if not len(hit):
        raise ValueError("cutoff not reached")
    return int(hit[0])


def fourier_mmax(l):
    """The number of Fourier coefficients above roundoff, as fourier_cutoff counts them: max(200, ceil(6 / (pi l)))."""
    return max(FOURIER_TABLE_M, int(math.ceil(6.0 / (math.pi * float(l)))))


def fourier_whole_table_tail(l):
    """RC3 (records/phase4_3_review_fable.md §20.4): the smallest tail the Fourier builder accepts at l and with which it keeps its whole
    table (fourier_cutoff(l, tail) = 200), written with at most six significant digits and rounded up so that the printed value is
    itself accepted; the interval's lower end 1e-15 when that already needs no more than the table. Verified by fourier_cutoff;
    None if no such printed value is found (then no value is offered)."""
    lo = CONTROL_INTERVAL[0]
    if fourier_cutoff(l, lo) <= FOURIER_TABLE_M:
        return lo
    c = fourier_coefficients(l, fourier_mmax(l))
    exact = 2 * c[FOURIER_TABLE_M + 1:].sum() / (c[0] + 2 * c[1:].sum())
    for digits in (3, 4, 6):
        tau = float(f"{exact:.{digits - 1}e}")
        k = 0
        while tau < exact and k < 3:   # round up in the last printed digit
            tau = float(f"{tau * (1 + 10.0 ** (1 - digits)):.{digits - 1}e}")
            k += 1
        if lo <= tau < CONTROL_INTERVAL[1] and fourier_cutoff(l, tau) == FOURIER_TABLE_M and float(f"{tau:g}") == tau:
            return tau
    return None


def stated_kernel_matrix(points, sigma, l):
    """Item A3: the stated law, SIMSOPT's eleven-image kernel k(d) = sigma^2 sum_{j=-5..5} exp(-(d+j)^2/l^2), d = t_i - t_k, with its
    (value, first-derivative) blocks [[k, -k'], [k', -k'']], from the closed form (independent of the builders, which factor
    mc_reference.kernel_matrix or use the Fourier coefficients)."""
    t = np.asarray(points, dtype=float)
    d = t[:, None] - t[None, :]
    k = np.zeros_like(d); k1 = np.zeros_like(d); k2 = np.zeros_like(d)
    for j in range(-5, 6):
        x = d + j
        e = np.exp(-x ** 2 / l ** 2)
        k += e; k1 += -2.0 * x / l ** 2 * e; k2 += (4.0 * x ** 2 / l ** 4 - 2.0 / l ** 2) * e
    n = len(t)
    K = np.empty((2 * n, 2 * n))
    K[:n, :n], K[:n, n:], K[n:, :n], K[n:, n:] = k, -k1, k1, -k2
    return sigma ** 2 * K


def fourier_table_matrix(points, sigma, l, M=FOURIER_TABLE_M):
    """Item A3: the law of the Fourier builder with its whole table (m = 0 ... M), sigma^2 [c_0 + 2 sum_m c_m cos 2 pi m d] and its
    derivative blocks; what no tail can improve (the infinite periodization, truncated at M)."""
    t = np.asarray(points, dtype=float)
    d = t[:, None] - t[None, :]
    c = fourier_coefficients(l, M)
    k = np.full_like(d, c[0]); k1 = np.zeros_like(d); k2 = np.zeros_like(d)
    for m in range(1, M + 1):
        if c[m] == 0.0:
            break
        w = 2 * np.pi * m
        cs, sn = np.cos(w * d), np.sin(w * d)
        k += 2 * c[m] * cs; k1 += -2 * c[m] * w * sn; k2 += -2 * c[m] * w * w * cs
    n = len(t)
    K = np.empty((2 * n, 2 * n))
    K[:n, :n], K[:n, n:], K[n:, :n], K[n:, n:] = k, -k1, k1, -k2
    return sigma ** 2 * K


def law_residual(C, K, sigma, l):
    """Item A1/A3: reading (v)'s per-block error of a (value, first-derivative) covariance C against the stated K: max-abs entry error
    over the pointwise variance sigma^2 sum_j exp(-j^2/l^2); cross blocks also over max(1, 2 pi M), the derivative block over
    max(1, 2 pi M)^2, M = fourier_cutoff(l, 1e-12)."""
    n = C.shape[0] // 2
    var = float(sigma) ** 2 * sum(math.exp(-(j / l) ** 2) for j in range(-5, 6))
    s = max(1.0, 2 * math.pi * fourier_cutoff(l, 1e-12))
    E = np.abs(np.asarray(C) - np.asarray(K))
    out = {"value": float(E[:n, :n].max() / var), "cross": float(max(E[:n, n:].max(), E[n:, :n].max()) / (var * s)),
           "derivative": float(E[n:, n:].max() / (var * s * s))}
    out["max"] = max(out.values())
    return out


def retained_law(modes, level):
    """Item A1/A3: the covariance of one level's retained modes on one coil (owner 0, direction 0), sum_a lam_a [phi; dphi][phi; dphi]^T.
    Every owner and direction of a level carries the same functions (_assemble)."""
    lev_id = LEVELS.index(level)
    sel = np.where((np.asarray(modes.level) == lev_id) & (np.asarray(modes.owner) == 0) & (np.asarray(modes.direction) == 0))[0]
    F = np.concatenate([np.asarray(modes.phi)[sel], np.asarray(modes.dphi)[sel]], axis=1)
    return (F.T * np.asarray(modes.lam)[sel]) @ F


def image_maps(nfp, stellsym, nbase):
    """A_j and base index for the coils_via_symmetries ordering (coil.py:482-509)."""
    flips = [False, True] if stellsym else [False]
    maps, base = [], []
    F = np.diag([1.0, -1.0, -1.0])
    for k in range(nfp):
        phi = 2 * np.pi * k / nfp
        R = np.array([[np.cos(phi), -np.sin(phi), 0.0], [np.sin(phi), np.cos(phi), 0.0], [0.0, 0.0, 1.0]])
        for flip in flips:
            for i in range(nbase):
                maps.append(F @ R if flip else R)
                base.append(i)
    return np.array(maps), np.array(base)


@dataclass
class ModeSet:
    representation: str
    nq: int
    nfp: int
    stellsym: bool
    nbase: int
    lam: np.ndarray        # (A,)
    phi: np.ndarray        # (A, nq) scalar value function (metadata; base_psi is authoritative)
    dphi: np.ndarray       # (A, nq)
    direction: np.ndarray  # (A,) in {0,1,2}, or -1 for a vector mode
    level: np.ndarray      # (A,) 0 systematic, 1 statistical
    owner: np.ndarray      # (A,) base index (systematic) or coil index (statistical)
    info: dict = field(default_factory=dict)
    base_psi: np.ndarray = None   # (A, nq, 3) lab-frame displacement on the owner / base curve
    base_dpsi: np.ndarray = None  # (A, nq, 3)
    quadpoints: np.ndarray = None  # unit 4.3: the curve parameter points the modes were built on (None: older mode sets)

    def __post_init__(self):
        self.maps, self.image_base = image_maps(self.nfp, self.stellsym, self.nbase)
        self.ncoils = len(self.maps)
        if self.base_psi is None:
            E = np.eye(3)[np.asarray(self.direction)]
            self.base_psi = np.asarray(self.phi)[:, :, None] * E[:, None, :]
            self.base_dpsi = np.asarray(self.dphi)[:, :, None] * E[:, None, :]

    @property
    def size(self):
        return len(self.lam)

    def support(self, a):
        """[(coil j, 3x3 map A_j)] on which mode a acts."""
        if self.level[a] == 1:
            return [(int(self.owner[a]), np.eye(3))]
        return [(j, self.maps[j]) for j in range(self.ncoils) if self.image_base[j] == self.owner[a]]

    def on_coil(self, a, j):
        """(psi, dpsi) of mode a on coil j in the lab frame, or None."""
        if self.level[a] == 1:
            if self.owner[a] != j:
                return None
            return self.base_psi[a], self.base_dpsi[a]
        if self.image_base[j] != self.owner[a]:
            return None
        A = self.maps[j]
        return self.base_psi[a] @ A.T, self.base_dpsi[a] @ A.T

    def tangents(self, j):
        """All modes acting on coil j: (indices, dgamma (k,nq,3), dgammadash (k,nq,3))."""
        idx, dg, dgd = [], [], []
        for a in range(self.size):
            v = self.on_coil(a, j)
            if v is not None:
                idx.append(a); dg.append(v[0]); dgd.append(v[1])
        if not idx:
            return np.array([], int), np.zeros((0, self.nq, 3)), np.zeros((0, self.nq, 3))
        return np.array(idx), np.array(dg), np.array(dgd)

    def _copy(self, **kw):
        d = dict(representation=self.representation, nq=self.nq, nfp=self.nfp, stellsym=self.stellsym,
                 nbase=self.nbase, lam=self.lam, phi=self.phi, dphi=self.dphi, direction=self.direction,
                 level=self.level, owner=self.owner, info=dict(self.info), base_psi=self.base_psi,
                 base_dpsi=self.base_dpsi, quadpoints=self.quadpoints)
        d.update(kw)
        return ModeSet(**d)

    def scaled(self, factor):
        """Same modes with all variances times factor (sigma -> sqrt(factor) sigma)."""
        return self._copy(lam=self.lam * factor, info=dict(self.info, scaled_by=factor))

    def subset(self, indices):
        idx = np.asarray(indices, dtype=int)   # unit 4.1 item 3: an empty index list is an integer index
        return self._copy(lam=self.lam[idx], phi=self.phi[idx], dphi=self.dphi[idx],
                          direction=self.direction[idx], level=self.level[idx], owner=self.owner[idx],
                          base_psi=self.base_psi[idx], base_dpsi=self.base_dpsi[idx],
                          info=dict(self.info, subset=idx.tolist()))

    # ------------------------------------------------------------------
    # covariance and isotropy
    # ------------------------------------------------------------------
    def reconstruct(self, i, j, cache=None):
        """Covariance between coils i and j: array (2, nq, 3, 2, nq, 3) indexed
        [channel_i, t, comp, channel_j, s, comp], channel 0 = value, 1 = derivative."""
        def mat(c):
            if cache is not None and c in cache:
                return cache[c]
            idx, dg, dgd = self.tangents(c)
            out = (idx, np.stack([dg, dgd], axis=1).reshape(len(idx), 2 * self.nq * 3))   # unit 4.1 item 3: explicit width (zero rows)
            if cache is not None:
                cache[c] = out
            return out
        ii, Pi = mat(i)
        jj, Pj = mat(j)
        common, ai, aj = np.intersect1d(ii, jj, return_indices=True)
        R = np.zeros((2 * self.nq * 3, 2 * self.nq * 3))
        if len(common):
            R = (Pi[ai] * self.lam[common][:, None]).T @ Pj[aj]
        return R.reshape(2, self.nq, 3, 2, self.nq, 3)

    def certificate(self, rtol=1e-12):
        key = ("_cert", rtol)
        if getattr(self, "_cert_cache", None) and key in self._cert_cache:
            return self._cert_cache[key]
        out = self._certificate(rtol)
        self._cert_cache = {key: out}
        return out

    def _certificate(self, rtol):
        """Isotropy certificate of PLAN §2 per physical coil, applied pointwise in t (unit 4.1 item 2, Astra finding 5;
        records/phase4_1_review_fable.md ruling 1).

        On coil j, over the K_j modes acting on it:
          M0(t) = sum_a lam_a psi_a psi_a^T,  M1(t) = sum_a lam_a psi'_a psi_a^T,
          S0(t) = sum_a |lam_a| |psi_a|^2,    S1(t) = sum_a |lam_a| |psi'_a| |psi_a|,
          dev_k(t) = max-abs of M_k(t) - (tr M_k(t) / 3) I.
        Point t passes when dev_k(t) <= rtol * tr M0(t) + 2 gamma_{K_j+5} S_k(t), k = 0, 1, with
        gamma_n = n eps / (1 - n eps), eps the float64 unit roundoff. The first term is the plan's tolerance; the second
        is the roundoff allowance: an element of M_k carries at most gamma_{K_j+1} S_k, tr M_k / 3 at most
        gamma_{K_j+4} S_k / 3, so their difference about (4/3) gamma_{K_j+5} S_k. With lam >= 0, S0(t) = tr M0(t), so the
        M0 test is the pointwise relative bound dev0(t) <= (rtol + 2 gamma) tr M0(t).
        Zero variance: S0(t) = 0 means every psi_a(t) with lam_a != 0 vanishes, so M0(t) = M1(t) = 0 exactly and the
        point passes (0 I is a scalar multiple of I); such points are counted. The coil is eligible when every point passes.
        Reported per coil: M0_dev, M1_dev = max_t dev_k(t) / tr M0(t) (0/0 := 0), failing_points, zero_variance_points,
        allowance_rel_max = max_t 2 gamma S0(t) / tr M0(t).
        """
        eps = np.finfo(float).eps
        I = np.eye(3)
        rows = []
        for j in range(self.ncoils):
            M0 = np.zeros((self.nq, 3, 3))
            M1 = np.zeros((self.nq, 3, 3))
            S0 = np.zeros(self.nq)
            S1 = np.zeros(self.nq)
            K = 0
            for a in range(self.size):
                v = self.on_coil(a, j)
                if v is None:
                    continue
                K += 1
                M0 += self.lam[a] * np.einsum("tp,tq->tpq", v[0], v[0])
                M1 += self.lam[a] * np.einsum("tp,tq->tpq", v[1], v[0])
                n0, n1 = np.linalg.norm(v[0], axis=1), np.linalg.norm(v[1], axis=1)
                S0 += abs(self.lam[a]) * n0 * n0
                S1 += abs(self.lam[a]) * n1 * n0
            n = K + 5
            allowance = 2.0 * n * eps / (1.0 - n * eps)
            tr0 = np.trace(M0, axis1=1, axis2=2)
            d0 = np.max(np.abs(M0 - (tr0 / 3)[:, None, None] * I), axis=(1, 2))
            d1 = np.max(np.abs(M1 - (np.trace(M1, axis1=1, axis2=2) / 3)[:, None, None] * I), axis=(1, 2))
            ok = (d0 <= rtol * tr0 + allowance * S0) & (d1 <= rtol * tr0 + allowance * S1)
            pos = tr0 > 0
            safe = np.where(pos, tr0, 1.0)
            rel = lambda d: np.where(pos, d / safe, np.where(d == 0, 0.0, np.inf))
            rows.append({"coil": j, "M0_dev": float(np.max(rel(d0))), "M1_dev": float(np.max(rel(d1))),
                         "failing_points": int(np.sum(~ok)), "zero_variance_points": int(np.sum(S0 == 0)),
                         "allowance_rel_max": float(np.max(np.where(pos, allowance * S0 / safe, 0.0))), "modes": K})
        eligible = all(r["failing_points"] == 0 for r in rows)
        return {"eligible": bool(eligible), "rtol": rtol, "rule": "pointwise; allowance 2 gamma_{K+5} S_k(t)",
                "max_M0_dev": max(r["M0_dev"] for r in rows),
                "max_M1_dev": max(r["M1_dev"] for r in rows), "per_coil": rows}


def _normalize(phi, dphi, lam):
    s = np.max(np.abs(phi))
    if s < 1e-12 * max(np.max(np.abs(dphi)), 1e-300):
        s = np.max(np.abs(dphi))
    return phi / s, dphi / s, lam * s * s


def _assemble(rep, nq, nfp, stellsym, nbase, functions, levels, info, quadpoints=None):
    """functions: dict level -> list of (phi, dphi, var); builds all modes."""
    ncoils = nbase * nfp * (2 if stellsym else 1)
    lam, phi, dphi, direction, level, owner = [], [], [], [], [], []
    for lev, funcs in functions.items():
        lev_id = LEVELS.index(lev)
        owners = range(nbase) if lev == "systematic" else range(ncoils)
        for o in owners:
            for d in range(3):
                for (p, dp, v) in funcs:
                    lam.append(v); phi.append(p); dphi.append(dp)
                    direction.append(d); level.append(lev_id); owner.append(o)
    # unit 4.1 item 3: integer index arrays and (0, nq) function arrays also when no function is retained
    return ModeSet(rep, nq, nfp, stellsym, nbase, np.array(lam, dtype=float), np.array(phi, dtype=float).reshape(-1, nq),
                   np.array(dphi, dtype=float).reshape(-1, nq), np.array(direction, dtype=int), np.array(level, dtype=int),
                   np.array(owner, dtype=int), info, quadpoints=None if quadpoints is None else np.array(quadpoints, dtype=float))


def fourier_modes(quadpoints, nfp, stellsym, nbase, sigma, l, levels=LEVELS, tail=1e-12, M=None):
    """Kernel KL modes; sigma, l may be dicts per level (validated by check_error_model, unit 4.3)."""
    t = np.asarray(quadpoints)
    levels, _, _ = check_error_model(sigma, l, levels)
    tail = check_control("tail", tail)   # Amendment A, item A2
    sig = sigma if isinstance(sigma, dict) else {lev: sigma for lev in levels}
    ell = l if isinstance(l, dict) else {lev: l for lev in levels}
    functions, info = {}, {"cutoff": {}, "sigma": {}, "l": {}, "tail": tail}
    for lev in levels:
        Mlev = fourier_cutoff(ell[lev], tail) if M is None else M
        if Mlev > FOURIER_TABLE_M:   # Amendment A, item A3: never a cutoff whose tail is unmet
            raise ValueError(f"level {lev}: the Fourier table ends at m = {FOURIER_TABLE_M} and the tail {tail:g} is not reached at "
                             f"l = {ell[lev]:g} (it needs M = {Mlev}); use builder='kernel_exact'")
        c = fourier_coefficients(ell[lev])
        funcs = [_normalize(np.ones_like(t), np.zeros_like(t), sig[lev] ** 2 * c[0])]
        for m in range(1, Mlev + 1):
            w = 2 * np.pi * m
            funcs.append(_normalize(np.cos(w * t), -w * np.sin(w * t), 2 * sig[lev] ** 2 * c[m]))
            funcs.append(_normalize(np.sin(w * t), w * np.cos(w * t), 2 * sig[lev] ** 2 * c[m]))
        functions[lev] = funcs
        info["cutoff"][lev] = Mlev
        info["sigma"][lev] = sig[lev]
        info["l"][lev] = ell[lev]
    return _assemble("fourier", len(t), nfp, stellsym, nbase, functions, levels, info, quadpoints=t)


def _realized_functions(sampler, rel_cut):
    """(functions, info) of one sampler's realized covariance C = L L^T.

    n_derivs = 1: C = L L^T exactly as before unit 4.3. n_derivs = 2 (unit 4.3, the driver's sample-mean sampler): the leading
    (value, first-derivative) block C = L[:2n] L[:2n]^T, the covariance of the (gamma, gamma') displacement SquaredFlux sees."""
    check_error_model(sampler.sigma, sampler.length_scale, ("statistical",))
    rel_cut = check_control("rel_cut", rel_cut)   # Amendment A, item A2
    n = len(sampler.points)
    if sampler.n_derivs == 1:
        C = sampler.L @ sampler.L.T
    elif sampler.n_derivs == 2:
        C = sampler.L[:2 * n] @ sampler.L[:2 * n].T
    else:
        raise ValueError(f"realized_modes expects a sampler with n_derivs 1 or 2, not {sampler.n_derivs}")
    if not np.any(C):   # unit 4.1 item 3 (Astra finding 6): zero covariance, no mode, exact zero residual
        return [], {"retained": 0, "rel_cut": rel_cut, "sigma": float(sampler.sigma), "l": float(sampler.length_scale),
                    "truncation_residual_rel": 0.0, "zero_covariance": True}
    w, V = np.linalg.eigh(C)
    if not w.max() > 0:
        raise ValueError("the realized covariance is nonzero but has no positive eigenvalue")
    keep = w > rel_cut * w.max()
    w, V = w[keep], V[:, keep]
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    funcs = [_normalize(V[:n, k], V[n:, k], w[k]) for k in range(len(w))]
    Ct = (V * w) @ V.T
    info = {"retained": int(len(w)), "rel_cut": rel_cut, "sigma": float(sampler.sigma),
            "l": float(sampler.length_scale),
            "truncation_residual_rel": float(np.max(np.abs(Ct - C)) / np.max(np.abs(C)))}
    return funcs, info


def realized_modes(sampler, nfp, stellsym, nbase, levels=LEVELS, rel_cut=1e-13):
    """KL modes of a sampler's realized covariance C = L L^T (n_derivs = 1).

    `sampler` is one sampler used for every level, or a dict {level: sampler} for
    per-level error models (Phase 3, carried item 4); the dict path builds each
    level's modes from its own sampler and records the per-level info."""
    if isinstance(sampler, dict):
        if set(sampler) != set(levels):
            raise ValueError("per-level samplers must cover exactly the levels")
        npts = {len(sampler[lev].points) for lev in levels}
        if len(npts) != 1:
            raise ValueError("per-level samplers must share the quadrature points")
        functions, per = {}, {}
        for lev in levels:
            functions[lev], per[lev] = _realized_functions(sampler[lev], rel_cut)
        pts = [np.asarray(sampler[lev].points, dtype=float) for lev in levels]
        if any(not np.array_equal(p, pts[0]) for p in pts):
            raise ValueError("per-level samplers must share the quadrature points")
        return _assemble("realized", npts.pop(), nfp, stellsym, nbase, functions, levels, {"per_level": per}, quadpoints=pts[0])
    funcs, info = _realized_functions(sampler, rel_cut)
    return _assemble("realized", len(sampler.points), nfp, stellsym, nbase, {lev: funcs for lev in levels}, levels, info,
                     quadpoints=sampler.points)


def realized_basis(sampler, rel_cut=1e-13):
    """(V, scales) of realized_modes' functions: columns of V are the retained
    eigenvectors of L L^T (descending), scales the normalization of _normalize."""
    n = len(sampler.points)
    C = sampler.L @ sampler.L.T
    w, V = np.linalg.eigh(C)
    keep = w > rel_cut * w.max()
    w, V = w[keep], V[:, keep]
    V = V[:, np.argsort(w)[::-1]]
    scales = []
    for k in range(V.shape[1]):
        p, dp = V[:n, k], V[n:, k]
        s = np.max(np.abs(p))
        if s < 1e-12 * max(np.max(np.abs(dp)), 1e-300):
            s = np.max(np.abs(dp))
        scales.append(s)
    return V, np.array(scales)


def kl_coordinates_realized(basis, sample_arrays):
    """xi = V^T (L z) per Cartesian component, scaled to realized_modes' normalization.

    basis: realized_basis(sampler). sample_arrays: (ncurves, 2, nq, 3).
    Returns (ncurves, 3, K) coordinates and the relative reconstruction residual.
    """
    V, scales = basis
    x = np.concatenate([sample_arrays[:, 0], sample_arrays[:, 1]], axis=1)  # (c, 2n, 3)
    xi = np.einsum("nk,cnd->cdk", V, x) * scales[None, None, :]
    resid = x - np.einsum("nk,cdk->cnd", V / scales[None, :], xi)
    return xi, float(np.max(np.abs(resid)) / max(np.max(np.abs(x)), 1e-300))


def kl_coordinates_fourier(modes_level_funcs, sample_arrays):
    """Least-squares projection of draws onto Fourier mode functions (value and derivative jointly).

    modes_level_funcs: (phi (K, nq), dphi (K, nq)) of one level and direction-free functions.
    Returns (ncurves, 3, K) coordinates and the relative residual.
    """
    phi, dphi = modes_level_funcs
    Bm = np.concatenate([phi, dphi], axis=1).T  # (2nq, K)
    x = np.concatenate([sample_arrays[:, 0], sample_arrays[:, 1]], axis=1)  # (c, 2nq, 3)
    coef, *_ = np.linalg.lstsq(Bm, x.transpose(1, 0, 2).reshape(Bm.shape[0], -1), rcond=None)
    xi = coef.reshape(Bm.shape[1], x.shape[0], 3).transpose(1, 2, 0)
    resid = x - np.einsum("nk,cdk->cnd", Bm, xi)
    return xi, float(np.max(np.abs(resid)) / max(np.max(np.abs(x)), 1e-300))
