"""Re-summarise 03 outputs without re-scanning telemetry (same logic as the end of 03_telemetry.py)."""
import pandas as pd
from common import OUT, save
daily = pd.read_parquet(f"{OUT}/03_tel_daily.parquet")
nonadd = [k for k in daily.columns if k.startswith(("dt_", "corr_", "heading_err", "range_per", "speed_max"))]
tot = {"sum": daily.drop(columns=["day"] + nonadd).sum(numeric_only=True).to_dict(), "per_day_stats": daily[nonadd].describe().to_dict(),
       "low_volume_days": daily.nsmallest(5, "n")[["day", "n", "dt_med"]].to_dict(orient="records")}
save("03_tel_totals.json", tot)
print(daily[nonadd].describe().round(3).to_string()); print(tot["low_volume_days"])
