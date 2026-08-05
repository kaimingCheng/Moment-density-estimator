# Moment density estimator

Density reconstruction from **empirical moments** using a **generalized Hermite polynomial** estimator, **bandwidth selection** via an asymptotic MISE proxy, and **convex post-processing** (OSQP) to enforce approximate nonnegativity of the estimated density on a grid. Optional **moment completion** extends a truncated moment sequence.

## Pipeline (high level)

1. **Moments** — sample moments from a target distribution, or pass a fixed moment vector.
2. **Bandwidth `a`** — grid search minimizing `mise_estimator` (variance + squared bias proxy).
3. **Hermite coefficients** — map moments to coefficients with `hermite_coefficient`.
4. **Post-process** — `negative_density_post_process` iteratively projects toward coefficients whose Hermite series (times the Gaussian envelope) stays nonnegative on a 1D grid (OSQP).
5. **Plots** — compare the true PDF, the raw Gram estimator, and the post-processed estimate.

## Repository layout

| File | Role |
|------|------|
| `density_estimator_pipeline.py` | Core implementation: kernels, MISE, QP loop, `run_hermite_estimation_pipeline`, moment completion helpers. |
| `density_estimator_postprocess.ipynb` | Short notebook: imports, standard-normal example, optional Johnson SU stub, moment-completion. |

## Requirements

- Python 3.8+
- NumPy, SciPy, Matplotlib, Seaborn
- [CVXPY](https://www.cvxpy.org/) and [OSQP](https://osqp.org/) (for the positivity QP)

Install (example):

```bash
pip install numpy scipy matplotlib seaborn cvxpy osqp
```

## Quick start (local)

Clone the repo, then from the project directory:

```bash
python -c "from density_estimator_pipeline import run_hermite_estimation_pipeline; print(run_hermite_estimation_pipeline.__doc__)"
```

Or open `density_estimator_postprocess.ipynb` in Jupyter / VS Code and run cells top to bottom.

Main entry point:

```python
from scipy.stats import norm
from density_estimator_pipeline import run_hermite_estimation_pipeline
import numpy as np

dist = norm(0, 1)
best_a, weights = run_hermite_estimation_pipeline(
    dist,
    n=100_000,
    m=50,
    M=100,
    a_grid=np.linspace(1.5, 4.0, 80),
    x_range=np.linspace(-8, 8, 161),
    dist_name="Standard normal",
)
```

You can pass **`moments=...`** instead of sampling from `dist` (see docstring in `density_estimator_pipeline.py`).

# Hermite moment density estimator

Density reconstruction from **empirical moments** using a **Hermite–Gaussian** (Gram) estimator, **bandwidth selection** via an asymptotic MISE proxy, and **convex post-processing** (OSQP) to enforce approximate nonnegativity of the estimated density on a grid. Optional **moment completion** extends a truncated moment sequence before re-running the same pipeline.

## Pipeline (high level)

1. **Moments** — sample moments from a target distribution, or pass a fixed moment vector.
2. **Bandwidth `a`** — grid search minimizing `mise_estimator` (variance + squared bias proxy), or **smoothness in the MISE plateau**: restrict to the near-minimum MISE band on the same grid, scan `a` there, and pick the post-processed curve with smallest ∫(f'')² on the evaluation grid (optionally preferring nonnegative curves); see `bandwidth_selection='smoothness_in_mise_region'` in `postprocess_density_curve`.
3. **Hermite coefficients** — map moments to coefficients with `hermite_coefficient`.
4. **Post-process** — `negative_density_post_process` iteratively projects toward coefficients whose Hermite series (times the Gaussian envelope) stays nonnegative on a 1D grid (OSQP). Separately, **`l2_pdf_projection_on_grid`** projects a **grid curve** (e.g. the **raw** Hermite density before OSQP) to the closest nonnegative curve in L² (trapezoid objective) subject to **∫ g dx ≤ 1** on the grid. **`repair_density_nonnegativity`** can call that as `mode="l2_nonneg_mass"` / `"l2_pdf_qp"`. None of these grid projections preserve Hermite moments.
5. **Plots** — compare the true PDF, the raw Gram estimator, and the post-processed estimate.

## Repository layout

| File | Role |
|------|------|
| `density_estimator_pipeline.py` | Core implementation: kernels, MISE, QP loop, `run_hermite_estimation_pipeline`, moment completion helpers. |
| `density_estimator_postprocess.ipynb` | Short notebook: imports, standard-normal example, optional Johnson SU stub, moment-completion + rerun. |
| `moment_density_estimator_hermite.py` | Self-contained GitHub-facing Hermite moment-density estimator module; regenerated from canonical source via **`_build_moment_space_fisher_standalone.py`**. |
| `moment_density_estimator_hermite_demo.ipynb` | Focused standalone demo using the renamed module and a single selected `moment_qp_metric`. |
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

Clone the repo, then from the project directory:

```bash
python -c "from density_estimator_pipeline import run_hermite_estimation_pipeline; print(run_hermite_estimation_pipeline.__doc__)"
```

Or open `density_estimator_postprocess.ipynb` in Jupyter / VS Code and run cells top to bottom. The comparison section includes **Johnson SU** shapes (standardized to mean 0, variance 1) via **`StandardizedFrozen`** in `density_estimator_pipeline.py`.

Main entry point:

```python
from scipy.stats import norm
from density_estimator_pipeline import run_hermite_estimation_pipeline
import numpy as np

dist = norm(0, 1)
best_a, weights = run_hermite_estimation_pipeline(
    dist,
    n=100_000,
    m=50,
    M=100,
    a_grid=np.linspace(1.5, 4.0, 80),
    x_range=np.linspace(-8, 8, 161),
    dist_name="Standard normal",
)
```

You can pass **`moments=...`** instead of sampling from `dist` (see docstring in `density_estimator_pipeline.py`).

**Nonnegative post-processing:** `negative_density_post_process` defaults to **`method="iterative"`** (constraints at negative local minima, loop until stable). Use **`method="dense"`** for a **single** OSQP with \(f(x)\ge 0\) on a dense grid. Optional **`postprocess_ridge`** (and `run_hermite_estimation_pipeline(..., postprocess_ridge=...)`) adds Tikhonov regularization \(\|b\|^2\) on Hermite coefficients.

If the **evaluated** curve still has tiny negatives (grid vs. constraint grid, or numerical noise), use **`repair_density_nonnegativity(x, y, ...)`** — clip and/or linearly bridge negative runs, optionally **renormalize** to integrate to 1 on `x`. This **does not** preserve Hermite moments; see the function docstring for the honest trade-off.

### Moment completion (optional)

The notebook shows completing the first 41 empirical moments up to order 61 with `iterative_moment_completion`, then calling `run_hermite_estimation_pipeline` again with the extended vector and **`known=41`** for MISE bookkeeping.

**Alternative:** `iterative_moment_completion_orthogonal_density` **repeats** for each new order: **MISE-optimal `a`** on the **current** moment vector (same `n` as the experiment), **Hermite weights** `hermite_coefficient(μ, a, k)` with `k = len(μ)`, then the next **μ_L** from **closed-form** Gaussian–Hermite integrals (optionally normalized by mass at that step). Optional **`fixed_a`** skips MISE. This differs from **`w_j = 0`** algebraic completion.

**Chain study:** `completion_chain_osqp_compare` starts from an **even** empirical prefix, plots **true / raw Hermite / OSQP** on a grid, records **L¹** errors and **global minima** of raw/OSQP on that grid, then **adds two moments** (default **5** rounds per chain) on **orthogonal** vs **algebraic** chains. By default OSQP uses a **single QP** enforcing **f(x) ≥ 0** at grid points (not **strict** positivity; between-node dips possible). Returns density **figure**, optional **2×2 metrics** (horizontal **raw @ prefix** vs **OSQP** vs add-step for L¹ and global min, orthogonal and algebraic), and **table rows**.

### Comparing QP post-processing across completion orders

`compare_postprocess_across_completion_orders` reuses the same empirical prefix, completes to several target orders, and for each run runs MISE bandwidth search + OSQP nonnegativity projection. Use `plot_postprocess_completion_comparison` (and `_log`) to overlay **post-processed** densities against the true PDF—useful to see whether more completed moments improve the enforced-nonnegative estimate.

`postprocess_comparison_metrics_table` prints a **table** of the post-processed **global minimum**, **L¹ norms** of true vs. estimated density on the grid (integrated mass on the truncated interval), **|‖f̂‖₁ − ‖f‖₁|**, and the **L¹ curve error** ∫|f̂ − f| dx.

`completion_study_until_nonnegative` fixes prefix length **l** and Hermite truncation **k**, sweeps completion target order with **iterative** post-processing, records **min f̂ on the evaluation grid** and **L¹ error** vs. true density, optionally stops when the grid minimum is nonnegative, and `plot_completion_study_results` plots L¹ error and **post_global_min** vs. **moments added**.

`compare_osqp_vs_l2_projection_table` uses the same setup (**l=60**, **k=40**, up to **10** extra completed moments), shared MISE **a**, and reports a table of **grid minimum** and **L¹ error to the true PDF** for **OSQP** vs. **L² PDF projection** (from the raw Hermite curve). Pass **`return_curve_rows=True`** and call **`plot_osqp_vs_l2_completion_sequence`** to plot **true**, **raw Hermite**, **OSQP**, and **L²** curves in a panel grid per completion level.

**Iterative OSQP** stops when there are no negative local minima in `find_roots`, or when the smallest negative minimum exceeds **`postprocess_termination`** (default **`-1e-7`**), or after **`postprocess_max_iter`** (default **30**), or on QP failure (previous iterate kept). See `negative_density_post_process`.

## Google Colab

1. Open the notebook in Colab and run the **first code cell** first. It installs `osqp`, `cvxpy`, and `seaborn`, puts `/content` on `sys.path`, and opens an **upload** dialog if `density_estimator_pipeline.py` is not already there.
2. Alternatively, upload **`density_estimator_pipeline.py`** yourself via **Files** (left sidebar) into `/content` so it sits next to the notebook; then the upload step is skipped.
3. After that, the `import density_estimator_pipeline` line should work.

### `No module named 'density_estimator_pipeline'`

Python only sees modules on **`sys.path`**. The file **`density_estimator_pipeline.py`** must exist in the **current working directory** (or a folder you add to `sys.path`).

| Situation | What to do |
|-----------|------------|
| **Colab** | Upload `density_estimator_pipeline.py` to `/content` (same place as the notebook). If you use Drive, `sys.path.insert(0, "/content/drive/MyDrive/your/project")`. |
| **Notebook in a subfolder** | `cd` to the project root in the terminal, or open the folder that contains the `.py` as the workspace root, or run `import sys; sys.path.insert(0, r"C:\path\to\moment")` before importing. |
| **VS Code / Jupyter** | Start Jupyter from the repo folder, or use the path logic in the first notebook cell (adds `cwd` and `cwd.parent`). |

The notebook’s first code cell tries **`cwd`** and **`cwd.parent`** automatically; if the error persists, add the directory that **actually contains** `density_estimator_pipeline.py` to `sys.path` manually.

## License

Add a `LICENSE` file in this repository if you want to specify terms (e.g. MIT).
