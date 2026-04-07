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

### Moment completion (optional)

The notebook shows completing the first 41 empirical moments up to order 61 with `iterative_moment_completion`, then calling `run_hermite_estimation_pipeline` again with the extended vector and **`known=41`** for MISE bookkeeping.

## Google Colab

1. Upload **`density_estimator_pipeline.py`** to the Colab session (Files sidebar or `google.colab.files.upload()`).
2. Ensure dependencies are available, e.g. `!pip install -q osqp cvxpy seaborn` if needed.
3. Copy the notebook cells or upload the `.ipynb` and run; imports resolve when the `.py` file sits next to the notebook (default `/content`).

## License

Add a `LICENSE` file in this repository if you want to specify terms (e.g. MIT).
