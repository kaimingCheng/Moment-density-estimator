"""
Command-line runner for the complete-only Hermite moment density estimator.

Examples
--------
Synthetic bimodal example:

    python run_moment_density_estimator_complete.py --demo --output-dir reports/demo

Moments from a CSV/NPY/TXT file:

    python run_moment_density_estimator_complete.py --moments moments.csv --prefix-len 8 --output-dir reports/moments

Samples from a CSV/NPY/TXT file:

    python run_moment_density_estimator_complete.py --samples data.csv --prefix-len 8 --max-moment-order 40 --output-dir reports/data
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.stats as st

from moment_density_estimator_complete import (
    BimodalNormal,
    empirical_moments_from_samples,
    hermite_complete_chain_report,
    save_report_outputs,
)


DEFAULT_A_GRIDS = {
    "bimodal": (1.2, 2.9, 50),
    "normal": (1.4, 2.0, 50),
    "logistic": (1.9, 2.7, 50),
    "skewed": (2.0, 3.2, 50),
    "skewed_su": (2.0, 3.2, 50),
    "su": (2.0, 3.2, 50),
}


def _load_vector(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    elif path.suffix.lower() == ".json":
        arr = np.asarray(json.loads(path.read_text(encoding="utf-8")), dtype=float)
    else:
        arr = np.loadtxt(path, delimiter="," if path.suffix.lower() == ".csv" else None)
    return np.asarray(arr, dtype=float).ravel()


def _make_dist(args):
    if args.dist is None:
        return None
    if args.dist == "normal":
        return st.norm(loc=float(args.normal_loc), scale=float(args.normal_scale))
    if args.dist == "bimodal":
        return BimodalNormal(
            mu1=float(args.mu1),
            sigma1=float(args.sigma1),
            mu2=float(args.mu2),
            sigma2=float(args.sigma2),
            w=float(args.mix_weight),
        )
    if args.dist == "logistic":
        return st.logistic(loc=0.0, scale=np.sqrt(3.0) / np.pi)
    if args.dist in {"skewed", "skewed_su"}:
        return st.johnsonsu(1.08, 2.18, loc=1.0, scale=1.76)
    if args.dist == "su":
        return st.johnsonsu(0.0, 1.8, loc=0.0, scale=1.6)
    raise ValueError(f"unknown --dist {args.dist!r}")


def _make_a_grid(args, dist_key):
    if args.a_min is None and args.a_max is None and args.a_points is None:
        lo, hi, n = DEFAULT_A_GRIDS.get(dist_key, (0.9, 2.2, 32))
    else:
        lo = 0.9 if args.a_min is None else float(args.a_min)
        hi = 2.2 if args.a_max is None else float(args.a_max)
        n = 32 if args.a_points is None else int(args.a_points)
    return np.linspace(float(lo), float(hi), int(n))


def parse_args():
    p = argparse.ArgumentParser(
        description="Run the complete-only Hermite moment density estimator."
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--demo", action="store_true", help="Use synthetic bimodal samples.")
    src.add_argument("--moments", type=str, help="Path to moments vector (.csv/.txt/.npy/.json).")
    src.add_argument("--samples", type=str, help="Path to 1D samples (.csv/.txt/.npy).")

    p.add_argument("--output-dir", type=str, default="complete_report", help="Output directory.")
    p.add_argument("--prefix", type=str, default="complete_report", help="Output file prefix.")

    p.add_argument("--prefix-len", type=int, default=8, help="Number of target moments to use.")
    p.add_argument("--max-moment-order", type=int, default=40, help="Moments computed from samples up to this order.")
    p.add_argument("--n-samples", type=int, default=50000, help="Sample size used for demo or MISE annotation.")
    p.add_argument("--seed", type=int, default=41, help="Random seed for demo sampling.")

    p.add_argument(
        "--dist",
        choices=["bimodal", "normal", "logistic", "skewed", "skewed_su", "su"],
        default=None,
        help="Optional built-in distribution for samples and true-density plots.",
    )
    p.add_argument("--normal-loc", type=float, default=0.0)
    p.add_argument("--normal-scale", type=float, default=1.0)
    p.add_argument("--mu1", type=float, default=-2.0)
    p.add_argument("--sigma1", type=float, default=1.0)
    p.add_argument("--mu2", type=float, default=2.0)
    p.add_argument("--sigma2", type=float, default=1.0)
    p.add_argument("--mix-weight", type=float, default=0.5)

    p.add_argument("--x-min", type=float, default=-8.0)
    p.add_argument("--x-max", type=float, default=8.0)
    p.add_argument("--x-points", type=int, default=129)
    p.add_argument("--a-min", type=float, default=None)
    p.add_argument("--a-max", type=float, default=None)
    p.add_argument("--a-points", type=int, default=None)
    p.add_argument("--fixed-a", type=float, default=None)

    p.add_argument("--extra-hermite-terms", type=int, default=3)
    p.add_argument("--alpha", type=float, default=1.0, help="Complete metric prefix moment weight.")
    p.add_argument("--lambda", dest="lambda_", type=float, default=1e-6, help="Complete metric coefficient anchor base weight.")
    p.add_argument("--coeff-prior-mult", type=float, default=1.0, help="Multiplier on the coefficient anchor weight.")
    p.add_argument("--ridge-g", type=float, default=1e-9)
    p.add_argument("--ridge-p", type=float, default=0.0)
    p.add_argument("--gram-n-grid", type=int, default=1001)
    p.add_argument("--gram-x-range-factor", type=float, default=12.0)
    p.add_argument("--no-unit-mass", action="store_true", help="Disable unit-mass equality.")
    p.add_argument("--unit-mass-value", type=float, default=1.0)
    p.add_argument("--verbose-osqp", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(int(args.seed))
    dist = _make_dist(args)
    dist_key = args.dist

    if args.demo:
        if dist is None:
            dist = BimodalNormal()
            dist_key = "bimodal"
        samples = dist.rvs(int(args.n_samples))
        moments_full = empirical_moments_from_samples(samples, int(args.max_moment_order))
        n_samples = int(args.n_samples)
    elif args.samples:
        samples = _load_vector(args.samples)
        moments_full = empirical_moments_from_samples(samples, int(args.max_moment_order))
        n_samples = int(samples.size)
    elif args.moments:
        moments_full = _load_vector(args.moments)
        n_samples = int(args.n_samples)
    else:
        if dist is None:
            dist = BimodalNormal()
            dist_key = "bimodal"
        samples = dist.rvs(int(args.n_samples))
        moments_full = empirical_moments_from_samples(samples, int(args.max_moment_order))
        n_samples = int(args.n_samples)

    prefix_len = int(args.prefix_len)
    if prefix_len < 1:
        raise ValueError("--prefix-len must be positive")
    if moments_full.size < prefix_len:
        raise ValueError("moments vector is shorter than --prefix-len")

    x = np.linspace(float(args.x_min), float(args.x_max), int(args.x_points))
    a_grid = _make_a_grid(args, dist_key)
    result = hermite_complete_chain_report(
        moments_full[:prefix_len].copy(),
        dist,
        n_samples,
        x=x,
        a_grid=a_grid,
        fixed_a=args.fixed_a,
        moment_space_ridge_G=float(args.ridge_g),
        moment_space_ridge_P=float(args.ridge_p),
        gram_n_grid=int(args.gram_n_grid),
        gram_x_range_factor=float(args.gram_x_range_factor),
        verbose_moment_osqp=bool(args.verbose_osqp),
        constrain_unit_mass=not bool(args.no_unit_mass),
        unit_mass_value=float(args.unit_mass_value),
        extra_hermite_terms=int(args.extra_hermite_terms),
        moment_qp_complete_alpha=float(args.alpha),
        moment_qp_complete_lambda=float(args.lambda_),
        moment_qp_complete_coeff_prior_mult=float(args.coeff_prior_mult),
    )
    paths = save_report_outputs(result, args.output_dir, prefix=args.prefix)
    print(json.dumps(paths, indent=2))


if __name__ == "__main__":
    main()
