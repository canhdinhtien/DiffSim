# realdata-v1 EDA

Read-only EDA of `data/mock/realdata-v1` for driver-tracking synthetic data. Results: [REPORT.md](REPORT.md), [final_field_spec.yaml](final_field_spec.yaml).

## Setup
Python 3.12 (the system interpreter; the project `.venv` has no data libs).

```bash
python -m pip install --user duckdb==1.* h3==4.* pyyaml pandas pyarrow scikit-learn
```

Override the data location with `REALDATA_ROOT` (defaults to `D:\Projects\DiffSim\data\mock\realdata-v1`).

## Run (in order, from this folder)
| Script | What it does | Output (`out/`) | Time* |
|---|---|---|---|
| `00_schema.py` | file sizes, row counts, schemas (parquet metadata only) | `00_schema.json` | s |
| `01_profile.py` | per-column null %, distinct, quantiles, top values; IDs/coords/hexes masked | `01_profile.json` | ~3 min |
| `02_relations.py` | keys, join coverage, identities between daily tables and trips, hex-tracking ↔ telemetry | `02_relations.json` | ~3 min |
| `02b_followups.py` | commission/cancel mismatch, rush-flag rule, stoppoints, hex ↔ ping join | `02b_followups.json` | ~2 min |
| `03_telemetry.py` | full telemetry quality pass, one daily file at a time | `03_tel_daily.parquet`, `03_tel_driverday.parquet`, `03_tel_hist.json`, `03_tel_totals.json` | ~15 min |
| `03b_summarize.py` | re-summarise 03 outputs without rescanning | `03_tel_totals.json` | s |
| `04_states.py` | driver-state reconstruction (telemetry × trips), H3 checks, dwell (3 sample days) | `04_states.json` | ~1 min |
| `05_patterns.py` | seasonality, trend, cross-signal relations, k-means driver clusters (needs 03) | `05_patterns.json` | ~2 min |
| `06_extra.py` | field-level checks behind specific verdicts | `06_extra.json` | <1 min |

\*16-core / 16 GB machine. `03_telemetry.py` honours `TEL_LIMIT=N` (first N days) for quick tests; `04_states.py` honours `STATE_DAYS=YYYY-MM-DD,...`.

Nothing writes to `realdata-v1`. No script prints raw rows; identifiers, coordinates and hexes are reported only as types, shapes and distributions.
`_peek.py <table>...` prints a compact view of `out/01_profile.json`.
