# src/base — traditional statistical baselines (method 1 of 8), series S1 (driver x day)

Two fixed methods, one shared interface (`fit(data)` on train drivers, `sample(n, cond, seed)` -> raw-unit windows):

| File | Method | Conditions used |
|---|---|---|
| `var_sarima.py` | M2 pooled VAR(p) per vehicle_model x fleet group, dow + holiday terms, driver random effect | group, day of week, holiday |
| `copula_ar.py` | M3 Gaussian copula + per-feature AR(p), empirical or NB/lognormal marginals, randomized PIT | group, day of week, holiday |
| `common.py` | interface, post-processing (clip/round, rush <= total, offered >= completed + cancelled), constraint + memorization diagnostics | |
| `configs/*.yaml` | search grid per method | |
| `run.py` | validation search -> selection -> 5-seed test evaluation | |
| `figures.py` | the 9 figure types in `results/base/figures/` | |

Shared, not specific to this method (reuse them for the other 7 methods):
- `src/data/realdata_v1.py`: S1 loader (`load_s1()`), driver-level split, train-only min-max scaler, conditions.
- `utils/evaluate.py::evaluate_all(real_norm, fake_norm, seed)`: the 4 metrics (discriminative, predictive, Context-FID, correlational).

## Reproduce (one command, from the repo root)

```bash
uv sync && uv run python -m src.base.run --stage all && uv run python -m src.base.figures
```

About 1 h on a 16-core CPU (4 worker processes x 4 torch threads; `--workers/--threads` to change).
The run resumes: finished validation trials in `results/base/trials.csv` are skipped. Data root defaults to
`data/mock/realdata-v1` (override with `REALDATA_ROOT`); the joined panel is cached in `data/derived/s1-panel-v1.parquet`.

Tests:

```bash
uv run pytest
```

## Outputs (`results/base/`)
`trials.csv` (every trial: method, config, split, seed, 4 metrics, runtime, peak memory), `best_params.yaml`,
`final_metrics.json`, `samples/*.npy` (raw units, (2400, 28, 6) float32, one per method and seed, plus
nearest-train distances), `figures/`, `BASELINE_REPORT.md`.
