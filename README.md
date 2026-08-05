# Moment density estimator

Density reconstruction from **empirical moments** using a **generalized Hermite polynomial** estimator, **bandwidth selection** via an asymptotic MISE proxy, and **convex post-processing** (OSQP) to enforce approximate nonnegativity of the estimated density on a grid. Optional **moment completion** extends a truncated moment sequence.

## Pipeline (high level)

1. **Moments** — sample moments from a target distribution, or pass a fixed moment vector.
2. **Bandwidth `a`** — grid search minimizing `mise_estimator` (variance + squared bias proxy), or **smoothness in the MISE plateau**: restrict to the near-minimum MISE band on the same grid, scan `a` there, and pick the post-processed curve with smallest ∫(f'')² on the evaluation grid (optionally preferring nonnegative curves); see `bandwidth_selection='smoothness_in_mise_region'` in `postprocess_density_curve`.
3. **Hermite coefficients** — map moments to coefficients with `hermite_coefficient`.
4. **Post-process** — `negative_density_post_process` iteratively projects toward coefficients whose Hermite series (times the Gaussian envelope) stays nonnegative on a 1D grid (OSQP). Separately, **`l2_pdf_projection_on_grid`** projects a **grid curve** (e.g. the **raw** Hermite density before OSQP) to the closest nonnegative curve in L² (trapezoid objective) subject to **∫ g dx ≤ 1** on the grid. **`repair_density_nonnegativity`** can call that as `mode="l2_nonneg_mass"` / `"l2_pdf_qp"`. None of these grid projections preserve Hermite moments.
5. **Plots** — compare the true PDF, the raw Gram estimator, and the post-processed estimate.

## Repository layout

| File | Role |
|------|------|
| `moment_density_estimator_complete.py` | Lightweight complete-only Hermite moment QP pipeline with no Fisher-local / max-entropy paths. |
| `moment_density_estimator_complete_demo.ipynb` | Paired notebook for the complete-only pipeline. |
| `run_moment_density_estimator_complete.py` | Command-line runner that loads moments or samples and writes CSV/JSON/PNG/PDF report outputs; pass `--dist` only when a true PDF is known. |

## Requirements

- Python 3.8+
- NumPy, SciPy, Matplotlib, Seaborn
- [CVXPY](https://www.cvxpy.org/) and [OSQP](https://osqp.org/) (for the positivity QP)

Install (example):

```bash
pip install numpy scipy matplotlib seaborn cvxpy osqp
```

## Quick start (local)

Download or clone the repository, then keep these three files in the same folder:

- `moment_density_estimator_complete.py`
- `moment_density_estimator_complete_demo.ipynb`
- `run_moment_density_estimator_complete.py`

### Notebook demo

Open `moment_density_estimator_complete_demo.ipynb` in Jupyter, VS Code, or Colab and run the cells from top to bottom. The notebook demonstrates the complete-only Hermite QP pipeline, including bandwidth selection, nonnegative density estimation, diagnostic plots, and report output.

### Command-line usage

Run the estimator from a moments file:

```bash
python run_moment_density_estimator_complete.py --moments moments.csv --prefix-len 8 --output-dir results --prefix my_report
```

Run the estimator from a one-dimensional samples file:

```bash
python run_moment_density_estimator_complete.py --samples samples.csv --prefix-len 8 --max-moment-order 40 --output-dir results
```

Run a built-in demo:

```bash
python run_moment_density_estimator_complete.py --demo --output-dir results/demo
```

The script writes a CSV table, JSON metadata, PNG figures, and a PDF report. If the true distribution is unknown, omit `--dist`; if it is known, add `--dist normal` or `--dist bimodal` to include the true PDF in the plots.
The notebook’s first code cell tries **`cwd`** and **`cwd.parent`** automatically; if the error persists, add the directory that **actually contains** `density_estimator_pipeline.py` to `sys.path` manually.

## License

Add a `LICENSE` file in this repository if you want to specify terms (e.g. MIT).
