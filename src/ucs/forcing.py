"""Hourly weather forcing from ERA5 via the Open-Meteo historical archive (free, no key).

Open-Meteo radiation values are means over the preceding hour, so they are centred on the
half hour before their time stamp; instantaneous variables are valid at the time stamp.
"""
import numpy as np
import pandas as pd

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
VARS = ["temperature_2m", "dew_point_2m", "surface_pressure", "wind_speed_10m", "cloud_cover",
        "shortwave_radiation", "direct_radiation", "diffuse_radiation"]
RADIATION = ["shortwave_radiation", "direct_radiation", "diffuse_radiation"]


def window(cfg: dict) -> tuple[str, str]:
    day = pd.Timestamp(cfg["sun"]["date"])
    start = day - pd.Timedelta(days=cfg["heat"]["spinup_days"] + 1)  # UTC covers the local day
    end = day + pd.Timedelta(days=1)
    return f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}"


def build_params(cfg: dict) -> dict:
    lon, lat = cfg["center"]
    start, end = window(cfg)
    return {"latitude": f"{lat:.5f}", "longitude": f"{lon:.5f}", "start_date": start, "end_date": end,
            "hourly": ",".join(VARS), "wind_speed_unit": "ms", "models": cfg["heat"]["era5_model"],
            "timezone": "GMT"}


def parse(payload: dict) -> pd.DataFrame:
    h = payload["hourly"]
    df = pd.DataFrame({"time": pd.to_datetime(h["time"], utc=True)})
    for v in VARS:
        df[v] = pd.to_numeric(pd.Series(h[v]), errors="coerce")
    return df.dropna().reset_index(drop=True)


def fetch(cfg: dict) -> pd.DataFrame:
    import requests

    r = requests.get(ARCHIVE_URL, params=build_params(cfg), timeout=120)
    r.raise_for_status()
    return parse(r.json())


def interpolator(df: pd.DataFrame):
    """Return f(t_utc) -> dict of forcing values, linear in time."""
    times = pd.to_datetime(df["time"], utc=True)
    t = (times - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()
    half = 1800.0
    cols = {}
    for v in VARS:
        tv = t - half if v in RADIATION else t
        cols[v] = (tv, df[v].to_numpy(dtype=float))

    def at(ts: float) -> dict:
        return {v: float(np.interp(ts, tv, y)) for v, (tv, y) in cols.items()}

    return at
