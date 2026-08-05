"""
Complete-only Hermite moment density estimator.

This module is intentionally separate from ``moment_density_estimator_hermite``:
it exposes only the ``moment_qp_metric="complete"`` workflow and contains no
Fisher-local / max-entropy dependencies.
"""
from __future__ import annotations

import csv
import json
import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import osqp
import scipy.sparse as sp
from scipy.stats import norm
from numpy.polynomial.hermite import Hermite, herm2poly, hermval


def hermite_approx(n, x):
    """Evaluate physicist Hermite polynomial H_n(x)."""
    n = int(n)
    coeffs = np.zeros(n + 1, dtype=float)
    coeffs[n] = 1.0
    return hermval(x, coeffs)


def dirac(x, a):
    """Gaussian envelope N(0, a^2 / 2)."""
    return norm.pdf(x, loc=0.0, scale=float(a) / np.sqrt(2.0))


def hermite_coefficient(moments, a, k):
    """Map raw moments to Hermite coefficients for the Gaussian-envelope basis."""
    list1 = []
    for j in range(int(k)):
        d = 0.0
        norm_j = (1 / 2) ** j
        for l in range(j + 1):
            if j % 2 == 0:
                if l % 2 == 0:
                    c = (
                        math.factorial(j)
                        * ((2 / a) ** l)
                        * ((-1) ** (int(j / 2) - int(l / 2)))
                        / (
                            math.factorial(l)
                            * math.factorial(int(j / 2) - int(l / 2))
                        )
                    )
                else:
                    c = 0.0
            else:
                if l % 2 == 0:
                    c = 0.0
                else:
                    c = (
                        math.factorial(j)
                        * ((2 / a) ** l)
                        * ((-1) ** (int((j - 1) / 2) - int((l - 1) / 2)))
                        / (
                            math.factorial(l)
                            * math.factorial(int((j - 1) / 2) - int((l - 1) / 2))
                        )
                    )
            d += c * moments[l]
        list1.append(d * norm_j / math.factorial(j))
    return list1


def plot_by_weights_final(x, weights, a=2.5):
    """Evaluate the Hermite-Gaussian density curve on ``x``."""
    x = np.asarray(x, dtype=float).ravel()
    weights = np.asarray(weights, dtype=float).ravel()
    H = np.zeros((x.size, weights.size), dtype=float)
    for j in range(weights.size):
        H[:, j] = Hermite.basis(j)(x / float(a))
    return (H @ weights) * dirac(x, a)


def _trapezoid_rule_weights(x):
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 2:
        raise ValueError("need at least two grid points")
    w = np.zeros(x.size, dtype=float)
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    for i in range(1, x.size - 1):
        w[i] = 0.5 * (x[i + 1] - x[i - 1])
    return w


def _trapezoid(y, x):
    try:
        from scipy.integrate import trapezoid

        return float(trapezoid(y, x))
    except Exception:
        return float(np.trapz(y, x))


def _univariate_normal_raw_moment(mu, sigma, n):
    n = int(n)
    if n < 0:
        raise ValueError("n must be nonnegative")
    s = 0.0
    for i in range(0, n + 1, 2):
        m = i // 2
        ez2m = 1.0
        for t in range(1, m + 1):
            ez2m *= 2.0 * t - 1.0
        s += math.comb(n, i) * (mu ** (n - i)) * (sigma**i) * ez2m
    return float(s)


class BimodalNormal:
    """Simple two-component Gaussian mixture used in the demos."""

    def __init__(self, mu1=-2.0, sigma1=1.0, mu2=2.0, sigma2=1.0, w=0.5):
        self.mu1, self.sigma1 = float(mu1), float(sigma1)
        self.mu2, self.sigma2 = float(mu2), float(sigma2)
        self.w = float(w)
        self.dist = type("Mock", (object,), {"name": "Bimodal Normal"})()

    def pdf(self, x):
        return self.w * norm.pdf(x, self.mu1, self.sigma1) + (1.0 - self.w) * norm.pdf(
            x, self.mu2, self.sigma2
        )

    def moment(self, n, *args, **kwargs):
        m1 = _univariate_normal_raw_moment(self.mu1, self.sigma1, n)
        m2 = _univariate_normal_raw_moment(self.mu2, self.sigma2, n)
        return float(self.w * m1 + (1.0 - self.w) * m2)

    def rvs(self, size):
        selectors = np.random.rand(int(size)) < self.w
        samples = np.zeros(int(size), dtype=float)
        samples[selectors] = np.random.normal(
            self.mu1, self.sigma1, size=int(np.sum(selectors))
        )
        samples[~selectors] = np.random.normal(
            self.mu2, self.sigma2, size=int(np.sum(~selectors))
        )
        return samples


def empirical_moments_from_samples(data, max_order):
    """Raw empirical moments E[X^k], k=0,...,max_order, from a 1D sample."""
    data = np.asarray(data, dtype=float).ravel()
    if data.size < 1:
        raise ValueError("need at least one sample")
    mo = int(max_order)
    if mo < 0:
        raise ValueError("max_order must be nonnegative")
    powers = np.arange(mo + 1, dtype=float)
    return np.mean(data[:, np.newaxis] ** powers, axis=0)


def _gaussian_raw_moment(k, sigma):
    k = int(k)
    if k < 0:
        raise ValueError("k must be nonnegative")
    if k % 2:
        return 0.0
    df = 1.0
    for i in range(1, k, 2):
        df *= i
    return df * (float(sigma) ** k)


def _gaussian_raw_moments_through(max_p, sigma):
    max_p = int(max_p)
    if max_p < 0:
        raise ValueError("max_p must be nonnegative")
    G = np.zeros(max_p + 1, dtype=np.float64)
    for k in range(0, max_p + 1, 2):
        G[k] = _gaussian_raw_moment(k, sigma)
    return G


def _hermite_basis_j_coeffs_in_x(j, a):
    j = int(j)
    h_asc = np.zeros(j + 1, dtype=float)
    h_asc[j] = 1.0
    poly_t = herm2poly(h_asc)
    c_x = np.zeros(j + 1, dtype=float)
    for m in range(j + 1):
        c_x[m] = poly_t[m] / (float(a) ** m)
    return c_x


def hermite_weights_power_moments(weights, a, max_order):
    """Closed-form power moments of the Hermite-Gaussian series."""
    mo = int(max_order)
    if mo < 0:
        raise ValueError("max_order must be nonnegative")
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.size == 0:
        return np.zeros(mo + 1, dtype=np.float64)
    sigma = float(a) / np.sqrt(2.0)
    pmax = mo + w.size - 1
    G = _gaussian_raw_moments_through(pmax, sigma)
    M = np.zeros((mo + 1, w.size), dtype=np.float64)
    for j in range(w.size):
        c_x = _hermite_basis_j_coeffs_in_x(j, a)
        j1 = j + 1
        Lg = np.arange(mo + 1, dtype=np.intp)[:, None] + np.arange(j1, dtype=np.intp)[
            None, :
        ]
        M[:, j] = np.sum(c_x[None, :j1] * G[Lg], axis=1)
    return M @ w


def delta_ij(i, j, a):
    """Coefficient of x^j in physicists' H_i(x/a)."""
    if (i - j) % 2 != 0 or j > i:
        return 0.0
    r = (i - j) // 2
    return (
        ((-1) ** r)
        * ((2 / a) ** j)
        * math.factorial(i)
        / (math.factorial(r) * math.factorial(j))
    )


def mise_estimator(mu, a, n, m, M):
    """Asymptotic MISE proxy used for bandwidth selection."""
    mu = np.asarray(mu, dtype=float).ravel()

    def cov_mu(i, j):
        return (mu[i + j] - mu[i] * mu[j]) / n

    var_term = 0.0
    for i in range(m + 1):
        for j in range(m + 1):
            cov_ij = cov_mu(i, j)
            Fi = 0.0
            for k in range(max(i, j), m + 1):
                Fi += delta_ij(k, i, a) * delta_ij(k, j, a) / (
                    2**k * math.factorial(k)
                )
            var_term += cov_ij * Fi

    bias_term = 0.0
    for j in range(m + 1, M + 1):
        coeff = 0.0
        for i in range(j + 1):
            coeff += delta_ij(j, i, a) * mu[i]
        bias_term += (coeff**2) / (2**j * math.factorial(j))
    return float(var_term + bias_term)


def compute_mise_curve(moments, a_grid, n, m, M):
    return np.asarray(
        [mise_estimator(moments, float(a), int(n), int(m), int(M)) for a in a_grid],
        dtype=float,
    )


def hermite_gram_matrix_density_l2_dirac_sq(
    a, n_basis, *, n_grid=4001, x_range_factor=12.0
):
    """Density L2 Gram matrix for psi_j(x)=dirac(x,a) H_j(x/a)."""
    a = float(a)
    nb = int(n_basis)
    if nb < 1:
        raise ValueError("n_basis must be positive")
    R = float(x_range_factor) * max(abs(a), 1e-12)
    xs = np.linspace(-R, R, int(n_grid), dtype=np.float64)
    phi = np.asarray(dirac(xs, a), dtype=np.float64)
    Hm = np.zeros((xs.size, nb), dtype=np.float64)
    for j in range(nb):
        Hm[:, j] = Hermite.basis(j)(xs / a)
    w = _trapezoid_rule_weights(xs)
    G = np.zeros((nb, nb), dtype=np.float64)
    for i in range(nb):
        for j in range(i, nb):
            s = float(np.sum(w * Hm[:, i] * Hm[:, j] * phi * phi))
            G[i, j] = G[j, i] = s
    return G


def _coefficient_to_moment_matrix(n_m, n_c, a):
    A = np.zeros((int(n_m), int(n_c)), dtype=np.float64)
    for j in range(int(n_c)):
        e = np.zeros(int(n_c), dtype=np.float64)
        e[j] = 1.0
        A[:, j] = hermite_weights_power_moments(e, a, int(n_m) - 1)
    return A


def moment_space_osqp_complete(
    mu_target,
    a,
    x,
    *,
    ridge_G=1e-10,
    ridge_P=0.0,
    gram_n_grid=4001,
    gram_x_range_factor=12.0,
    nonnegative_mode="density",
    verbose=False,
    osqp_max_iter=500_000,
    osqp_eps_abs=1e-6,
    osqp_eps_rel=1e-6,
    osqp_check_termination=25,
    osqp_scaling=10,
    constrain_unit_mass=True,
    unit_mass_value=1.0,
    extra_hermite_terms=0,
    moment_qp_complete_alpha=1.0,
    moment_qp_complete_lambda=1e-6,
    moment_qp_complete_coeff_prior_mult=1.0,
):
    """
    Complete-only OSQP solver.

    Exact regime (``extra_hermite_terms=0``): minimize
    ``(c-c_ref)^T G (c-c_ref)``.

    Extended regime: minimize
    ``alpha*(A c-mu)^T W (A c-mu) + lambda_eff*(c-c_ref)^T G (c-c_ref)``.
    """
    mu_target = np.asarray(mu_target, dtype=np.float64).ravel()
    x = np.asarray(x, dtype=np.float64).ravel()
    n_m = int(mu_target.size)
    if n_m < 1:
        raise ValueError("mu_target must be nonempty")
    if x.size < 1:
        raise ValueError("x must be nonempty")
    if nonnegative_mode not in ("density", "hermite"):
        raise ValueError("nonnegative_mode must be 'density' or 'hermite'")
    if ridge_G < 0 or ridge_P < 0:
        raise ValueError("ridge_G and ridge_P must be nonnegative")

    extra = int(extra_hermite_terms)
    if extra < 0:
        raise ValueError("extra_hermite_terms must be nonnegative")
    n_c = n_m + extra
    alpha = float(moment_qp_complete_alpha)
    lam = float(moment_qp_complete_lambda)
    coeff_mult = float(moment_qp_complete_coeff_prior_mult)
    if alpha < 0 or lam < 0 or coeff_mult < 0:
        raise ValueError("complete objective weights must be nonnegative")
    lam_eff = lam * coeff_mult
    mass_target = float(unit_mass_value)
    if not np.isfinite(mass_target):
        raise ValueError("unit_mass_value must be finite")

    A = _coefficient_to_moment_matrix(n_m, n_c, a)
    G = hermite_gram_matrix_density_l2_dirac_sq(
        a, n_c, n_grid=gram_n_grid, x_range_factor=gram_x_range_factor
    )
    G_use = G + float(ridge_G) * np.eye(n_c, dtype=np.float64)

    c_ref = np.zeros(n_c, dtype=np.float64)
    c_ref[:n_m] = np.asarray(hermite_coefficient(mu_target, float(a), n_m), dtype=float)
    complete_is_exact = extra == 0
    W_complete = None

    if complete_is_exact:
        P = 2.0 * G_use
        q = -2.0 * (G_use @ c_ref)
    else:
        M_sq = A[:, :n_m]
        G_sq = G_use[:n_m, :n_m]
        try:
            Minv = np.linalg.inv(M_sq)
        except np.linalg.LinAlgError:
            Minv = np.linalg.pinv(M_sq, rcond=1e-12)
        W_complete = Minv.T @ G_sq @ Minv
        W_complete = 0.5 * (W_complete + W_complete.T)
        P = 2.0 * alpha * (A.T @ W_complete @ A) + 2.0 * lam_eff * G_use
        q = -2.0 * alpha * (A.T @ W_complete @ mu_target) - 2.0 * lam_eff * (
            G_use @ c_ref
        )

    P = 0.5 * (P + P.T) + float(ridge_P) * np.eye(n_c, dtype=np.float64)

    H = np.zeros((x.size, n_c), dtype=np.float64)
    for j in range(n_c):
        col = Hermite.basis(j)(x / float(a))
        if nonnegative_mode == "density":
            col = col * dirac(x, a)
        H[:, j] = col
    A_ineq = sp.csc_matrix(H)
    l_ineq = np.zeros(x.size, dtype=np.float64)
    u_ineq = np.full(x.size, np.inf, dtype=np.float64)

    if bool(constrain_unit_mass):
        A_mass = sp.csc_matrix(np.asarray(A[0, :], dtype=float).reshape(1, n_c))
        A_con = sp.vstack([A_ineq, A_mass], format="csc")
        l_con = np.concatenate([l_ineq, np.array([mass_target], dtype=np.float64)])
        u_con = np.concatenate([u_ineq, np.array([mass_target], dtype=np.float64)])
    else:
        A_con = A_ineq
        l_con = l_ineq
        u_con = u_ineq

    try:
        prob = osqp.OSQP()
        prob.setup(
            P=sp.csc_matrix(P),
            q=np.asarray(q, dtype=np.float64).ravel(),
            A=A_con,
            l=l_con,
            u=u_con,
            verbose=bool(verbose),
            max_iter=int(osqp_max_iter),
            eps_abs=float(osqp_eps_abs),
            eps_rel=float(osqp_eps_rel),
            check_termination=int(osqp_check_termination),
            scaling=int(osqp_scaling),
        )
        res = prob.solve()
    except Exception as exc:
        c0 = np.zeros(n_c, dtype=np.float64)
        return c0, False, {
            "status": f"error:{exc!s}",
            "A": A,
            "density_l2_basis_gram": G_use,
            "coefficient_reference": c_ref,
            "moment_qp_metric": "complete",
            "moment_qp_complete_alpha": alpha,
            "moment_qp_complete_lambda": lam,
            "moment_qp_complete_coeff_prior_mult": coeff_mult,
            "moment_qp_complete_lambda_effective": lam_eff,
            "complete_is_exact": complete_is_exact,
            "complete_moment_weight_W": W_complete,
            "objective_moment_J": None,
            "gram_moment_distance": None,
            "mu_tilde": None,
            "residual_moments": None,
            "implied_int_mu0": None,
            "n_moment_target": n_m,
            "n_hermite_coeffs": n_c,
            "extra_hermite_terms": extra,
        }

    c_opt = np.asarray(res.x, dtype=np.float64).ravel()
    status = getattr(res.info, "status", None) or getattr(res.info, "status_val", "unknown")
    success = str(status) == "solved"
    mu_tilde = A @ c_opt
    resid = mu_target - mu_tilde
    dc = c_opt - c_ref
    if complete_is_exact:
        J = float(dc @ G_use @ dc)
    else:
        J = float(alpha * (resid @ W_complete @ resid) + lam_eff * (dc @ G_use @ dc))
    return c_opt, success, {
        "status": status,
        "A": A,
        "density_l2_basis_gram": G_use,
        "coefficient_reference": c_ref,
        "moment_qp_metric": "complete",
        "moment_qp_complete_alpha": alpha,
        "moment_qp_complete_lambda": lam,
        "moment_qp_complete_coeff_prior_mult": coeff_mult,
        "moment_qp_complete_lambda_effective": lam_eff,
        "complete_is_exact": complete_is_exact,
        "complete_moment_weight_W": W_complete,
        "objective_moment_J": J,
        "gram_moment_distance": float(np.sqrt(max(J, 0.0))),
        "mu_tilde": mu_tilde,
        "residual_moments": resid,
        "implied_int_mu0": float(mu_tilde[0]),
        "constrain_unit_mass": bool(constrain_unit_mass),
        "unit_mass_value": mass_target,
        "n_moment_target": n_m,
        "n_hermite_coeffs": n_c,
        "extra_hermite_terms": extra,
        "osqp_info": res.info,
    }


def _tight_layout_quiet(fig):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="This figure includes Axes that are not compatible with tight_layout.*",
            category=UserWarning,
        )
        fig.tight_layout()


def hermite_complete_chain_report(
    initial_moments,
    dist,
    n_samples,
    *,
    x=None,
    a_grid=None,
    fixed_a=None,
    m_mise=40,
    M_mise_cap=60,
    plot_mise_bandwidth=True,
    moment_space_ridge_G=1e-10,
    moment_space_ridge_P=0.0,
    gram_n_grid=4001,
    gram_x_range_factor=12.0,
    nonnegative_mode="density",
    verbose_moment_osqp=False,
    plot_metric_curves=True,
    plot_summary_figure=True,
    constrain_unit_mass=True,
    unit_mass_value=1.0,
    extra_hermite_terms=0,
    moment_qp_complete_alpha=1.0,
    moment_qp_complete_lambda=1e-6,
    moment_qp_complete_coeff_prior_mult=1.0,
):
    """Complete-only raw Hermite vs nonnegative QP report."""
    mu0 = np.asarray(initial_moments, dtype=float).ravel()
    n0 = int(mu0.size)
    if n0 < 1:
        raise ValueError("initial_moments must be nonempty")
    if x is None:
        x = np.linspace(-8.0, 8.0, 161, dtype=float)
    else:
        x = np.asarray(x, dtype=float).ravel()
    if a_grid is None:
        a_grid = np.linspace(1.5, 4.0, 80, dtype=float)
    else:
        a_grid = np.asarray(a_grid, dtype=float).ravel()

    if fixed_a is None:
        M_mise = min(int(M_mise_cap), n0 - 1)
        m_eff = min(max(1, n0 // 2), max(1, (n0 - 1) // 2), int(m_mise), M_mise)
        mise_curve = compute_mise_curve(mu0, a_grid, int(n_samples), int(m_eff), int(M_mise))
        idx = int(np.argmin(mise_curve))
        a_star = float(a_grid[idx])
        mise_min = float(mise_curve[idx])
    else:
        a_star = float(fixed_a)
        if not (a_star > 0.0) or not np.isfinite(a_star):
            raise ValueError("fixed_a must be finite and positive")
        m_eff = None
        M_mise = None
        mise_curve = None
        mise_min = None

    has_true_density = dist is not None and hasattr(dist, "pdf")
    y_true = np.asarray(dist.pdf(x), dtype=float).ravel() if has_true_density else None
    dist_name = (
        getattr(getattr(dist, "dist", None), "name", None)
        or (type(dist).__name__ if dist is not None else "moments only")
    )
    w_raw = np.asarray(hermite_coefficient(mu0, a_star, n0), dtype=float)
    y_raw = plot_by_weights_final(x, w_raw, a_star)
    global_min_raw = float(np.min(y_raw))
    mu_raw = hermite_weights_power_moments(w_raw, a_star, n0 - 1)
    l1_raw = float(_trapezoid(np.abs(y_raw - y_true), x)) if has_true_density else float("nan")

    extra_h = int(extra_hermite_terms)
    if extra_h < 0:
        raise ValueError("extra_hermite_terms must be nonnegative")
    ks_sweep = list(range(extra_h + 1))
    table_rows = []
    y_ms_curves = []
    sweep_dist = []
    sweep_l1 = []
    sweep_gmin = []
    sweep_ok = []
    info_ms = {}
    c_ms = None
    mu_ms = mu_raw.copy()

    for step, k_try in enumerate(ks_sweep):
        c_ms, ok_ms, info_ms = moment_space_osqp_complete(
            mu0,
            a_star,
            x,
            ridge_G=moment_space_ridge_G,
            ridge_P=moment_space_ridge_P,
            gram_n_grid=gram_n_grid,
            gram_x_range_factor=gram_x_range_factor,
            nonnegative_mode=nonnegative_mode,
            verbose=verbose_moment_osqp,
            constrain_unit_mass=constrain_unit_mass,
            unit_mass_value=unit_mass_value,
            extra_hermite_terms=k_try,
            moment_qp_complete_alpha=moment_qp_complete_alpha,
            moment_qp_complete_lambda=moment_qp_complete_lambda,
            moment_qp_complete_coeff_prior_mult=moment_qp_complete_coeff_prior_mult,
        )
        y_ms = plot_by_weights_final(x, c_ms, a_star)
        y_ms_curves.append(np.asarray(y_ms, dtype=float).copy())
        mu_ms = hermite_weights_power_moments(c_ms, a_star, n0 - 1)
        l1_ms = float(_trapezoid(np.abs(y_ms - y_true), x)) if has_true_density else float("nan")
        gmin_ms = float(np.min(y_ms))
        gd = info_ms.get("gram_moment_distance")
        sweep_dist.append(float(gd) if gd is not None and np.isfinite(gd) else float("nan"))
        sweep_l1.append(l1_ms)
        sweep_gmin.append(gmin_ms)
        sweep_ok.append(bool(ok_ms))

        diff_raw = np.asarray(mu_raw[:n0], dtype=float) - mu0
        diff_ms = np.asarray(mu_ms[:n0], dtype=float) - mu0
        table_rows.append(
            {
                "step": step,
                "moment_qp_metric": "complete",
                "extra_hermite_terms": int(k_try),
                "n_prefix": n0,
                "n_hermite_coeffs": int(info_ms.get("n_hermite_coeffs", n0 + k_try)),
                "optimal_a": float(a_star),
                "L1_raw": l1_raw,
                "L1_complete": l1_ms,
                "global_min_raw": global_min_raw,
                "global_min_complete": gmin_ms,
                "complete_ok": bool(ok_ms),
                "complete_objective_J": info_ms.get("objective_moment_J"),
                "complete_distance": info_ms.get("gram_moment_distance"),
                "implied_mu0_raw": float(mu_raw[0]),
                "implied_mu0_complete": float(mu_ms[0]),
                "max_abs_err_prefix_raw": float(np.max(np.abs(diff_raw))),
                "max_abs_err_prefix_complete": float(np.max(np.abs(diff_ms))),
                "rmse_prefix_raw": float(np.sqrt(np.mean(diff_raw**2))),
                "rmse_prefix_complete": float(np.sqrt(np.mean(diff_ms**2))),
                "trapezoid_integral_complete": float(_trapezoid(y_ms, x)),
                "moment_qp_complete_alpha": moment_qp_complete_alpha,
                "moment_qp_complete_lambda": moment_qp_complete_lambda,
                "moment_qp_complete_coeff_prior_mult": moment_qp_complete_coeff_prior_mult,
                "moment_qp_complete_lambda_effective": info_ms.get(
                    "moment_qp_complete_lambda_effective"
                ),
            }
        )

    y_ms_final = y_ms_curves[-1]

    figure = Figure(figsize=(11.5, 4.0))
    ax = figure.add_subplot(1, 1, 1)
    if has_true_density:
        ax.plot(x, y_true, "k-", lw=2, label="true PDF")
    ax.plot(x, y_raw, color="C0", lw=1.25, label="raw Hermite")
    if extra_h <= 0:
        ax.plot(x, y_ms_final, color="C1", lw=1.5, label="complete QP")
    else:
        cmap = plt.cm.viridis
        for i, k_try in enumerate(ks_sweep):
            ax.plot(
                x,
                y_ms_curves[i],
                color=cmap(i / max(1, extra_h)),
                lw=2.0 if i == extra_h else 1.0,
                alpha=0.95 if i == extra_h else 0.75,
                label=f"complete k={k_try}" + (" final" if i == extra_h else ""),
            )
    ax.axhline(0.0, color="0.2", ls=":", lw=0.8)
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.set_title(f"Complete-only Hermite QP - {dist_name}, a={a_star:.4g}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    _tight_layout_quiet(figure)

    figure_metrics = None
    if plot_metric_curves:
        figure_metrics = Figure(figsize=(11.5, 4.0))
        ax1 = figure_metrics.add_subplot(1, 2, 1)
        ax2 = figure_metrics.add_subplot(1, 2, 2)
        if has_true_density:
            ax1.plot(ks_sweep, sweep_l1, "o-", color="C1", label="complete")
            ax1.axhline(l1_raw, color="C0", ls="--", label="raw")
            ax1.set_ylabel("L1 error")
            ax1.legend(fontsize=8)
        else:
            ax1.text(0.5, 0.5, "true density not supplied", ha="center", va="center")
            ax1.set_ylabel("L1 error unavailable")
        ax1.set_xlabel("extra Hermite terms")
        ax1.set_title("L1 curve error")
        ax1.grid(True, alpha=0.3)
        ax2.plot(ks_sweep, sweep_dist, "s-", color="C3")
        ax2.set_xlabel("extra Hermite terms")
        ax2.set_ylabel("log 10 scale")
        ax2.set_yscale("log", base=10)
        ax2.set_title("(A c - mu*)^T W (A c - mu*) / complete distance")
        ax2.grid(True, alpha=0.3, which="both")
        _tight_layout_quiet(figure_metrics)

    figure_density_and_mise = None
    if plot_mise_bandwidth:
        if mise_curve is None:
            figure_density_and_mise = figure
        else:
            figure_density_and_mise = Figure(figsize=(11.5, 4.0))
            axd = figure_density_and_mise.add_subplot(1, 2, 1)
            axm = figure_density_and_mise.add_subplot(1, 2, 2)
            if has_true_density:
                axd.plot(x, y_true, "k-", lw=2, label="true PDF")
            axd.plot(x, y_ms_final, color="C1", lw=1.5, label="complete final")
            axd.set_xlabel("x")
            axd.set_ylabel("density")
            axd.grid(True, alpha=0.3)
            axd.legend(fontsize=8)
            axm.plot(a_grid, mise_curve, color="C0", lw=1.5)
            axm.axvline(a_star, color="C3", ls="--", lw=1.0)
            axm.set_xlabel("bandwidth a")
            axm.set_ylabel("MISE proxy")
            axm.set_title("Bandwidth selection")
            axm.grid(True, alpha=0.3)
            _tight_layout_quiet(figure_density_and_mise)

    figure_summary = None
    if plot_summary_figure:
        figure_summary = Figure(figsize=(11.5, 7.5))
        gs = figure_summary.add_gridspec(2, 2, hspace=0.35, wspace=0.30)
        ax0 = figure_summary.add_subplot(gs[0, 0])
        ax1 = figure_summary.add_subplot(gs[0, 1])
        ax2 = figure_summary.add_subplot(gs[1, :])
        if mise_curve is not None:
            ax0.plot(a_grid, mise_curve, color="C0")
            ax0.axvline(a_star, color="C3", ls="--")
            ax0.set_title("MISE vs bandwidth")
            ax0.set_xlabel("a")
            ax0.set_ylabel("MISE proxy")
        else:
            ax0.text(0.5, 0.5, f"fixed a = {a_star:.4g}", ha="center", va="center")
            ax0.set_title("Fixed bandwidth")
        ax0.grid(True, alpha=0.3)
        ax1.plot(ks_sweep, sweep_dist, "s-", color="C3")
        ax1.set_yscale("log", base=10)
        ax1.set_xlabel("extra Hermite terms")
        ax1.set_ylabel("log 10 scale")
        ax1.set_title("(A c - mu*)^T W (A c - mu*), Target distance")
        ax1.grid(True, alpha=0.3, which="both")
        if has_true_density:
            ax2.plot(x, y_true, "k-", lw=2, label="true PDF")
        ax2.plot(x, y_raw, color="C0", ls="--", label="raw Hermite")
        ax2.plot(x, y_ms_final, color="C1", label="complete final")
        ax2.axhline(0.0, color="0.2", ls=":", lw=0.8)
        ax2.set_xlabel("x")
        ax2.set_ylabel("density")
        ax2.set_title("Density curves")
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8)
        _tight_layout_quiet(figure_summary)

    return {
        "table_rows": table_rows,
        "figure": figure,
        "figure_metrics": figure_metrics,
        "figure_density_and_mise": figure_density_and_mise,
        "figure_summary": figure_summary,
        "mise_curve_a_grid": None if mise_curve is None else np.asarray(a_grid).copy(),
        "mise_curve_values": None if mise_curve is None else np.asarray(mise_curve).copy(),
        "mise_minimum_value": mise_min,
        "optimal_a": a_star,
        "target_moments": mu0.copy(),
        "implied_moments_raw": np.asarray(mu_raw).copy(),
        "implied_moments_complete": np.asarray(mu_ms).copy(),
        "moment_space_osqp": info_ms,
        "success": bool(sweep_ok[-1]),
    }


def save_report_outputs(result, output_dir, *, prefix="complete_report"):
    """Save table/metadata/figures from ``hermite_complete_chain_report``."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    table_path = out / f"{prefix}_table.csv"
    rows = list(result.get("table_rows", []))
    if rows:
        keys = list(rows[0].keys())
        with table_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    meta = {
        "success": bool(result.get("success")),
        "optimal_a": result.get("optimal_a"),
        "mise_minimum_value": result.get("mise_minimum_value"),
        "target_moments": np.asarray(result.get("target_moments", [])).tolist(),
        "implied_moments_complete": np.asarray(
            result.get("implied_moments_complete", [])
        ).tolist(),
    }
    meta_path = out / f"{prefix}_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    figure_paths = {}
    figure_objs = []
    for key in ("figure", "figure_metrics", "figure_density_and_mise", "figure_summary"):
        fig = result.get(key)
        if fig is not None:
            figure_objs.append(fig)
            path = out / f"{prefix}_{key}.png"
            fig.savefig(path, dpi=160, bbox_inches="tight")
            figure_paths[key] = str(path)

    pdf_path = out / f"{prefix}_report.pdf"
    if figure_objs:
        from matplotlib.backends.backend_pdf import PdfPages

        with PdfPages(pdf_path) as pdf:
            for fig in figure_objs:
                pdf.savefig(fig, bbox_inches="tight")

    return {
        "table": str(table_path),
        "metadata": str(meta_path),
        "pdf": str(pdf_path) if figure_objs else None,
        "figures": figure_paths,
    }


__all__ = [
    "BimodalNormal",
    "empirical_moments_from_samples",
    "hermite_complete_chain_report",
    "moment_space_osqp_complete",
    "save_report_outputs",
    "plot_by_weights_final",
    "hermite_coefficient",
    "hermite_weights_power_moments",
]
