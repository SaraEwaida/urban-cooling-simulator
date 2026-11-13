"""Milestone 3: surface temperature through one day from the energy balance."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from . import classify as C
from . import forcing as F
from . import geohash
from .config import out_path
from .energy import SurfaceModel
from .grid import Grid


def load_forcing(cfg: dict) -> pd.DataFrame:
    path = Path(cfg["paths"]["forcing"])
    if not path.exists():
        df = F.fetch(cfg)
        df.to_csv(out_path(cfg, "forcing"), index=False)
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


class SunlitByTime:
    """Sunlit fraction at any time of day, interpolated between the hourly shadow frames."""

    def __init__(self, sun: xr.Dataset, utc_offset_h: float):
        self.frames = sun["sunlit"].values.astype("float32") / 100.0
        t = pd.DatetimeIndex(sun.time.values)
        self.hours = (t.hour + t.minute / 60).to_numpy(dtype=float)
        self.offset = utc_offset_h

    def __call__(self, t_utc: pd.Timestamp) -> np.ndarray:
        local = t_utc + pd.Timedelta(hours=self.offset)
        h = local.hour + local.minute / 60 + local.second / 3600
        if h <= self.hours[0]:
            return self.frames[0]
        if h >= self.hours[-1]:
            return self.frames[-1]
        i = int(np.searchsorted(self.hours, h)) - 1
        w = (h - self.hours[i]) / (self.hours[i + 1] - self.hours[i])
        return (1 - w) * self.frames[i] + w * self.frames[i + 1]


def simulate(scene: xr.Dataset, sun: xr.Dataset, forcing: pd.DataFrame, cfg: dict, verbose=True):
    hc, sc = cfg["heat"], cfg["sun"]
    model = SurfaceModel(scene["albedo"].values, scene["emissivity"].values, scene["evap"].values,
                         sun["sky_view_factor"].values, scene["surface_class"].values, C.CODE,
                         cfg["thermal"], hc["wind_a"], hc["wind_b"], hc["env_albedo"])
    at = F.interpolator(forcing)
    sunlit = SunlitByTime(sun, sc["utc_offset_h"])
    off = pd.Timedelta(hours=sc["utc_offset_h"])
    day0 = pd.Timestamp(sc["date"], tz="UTC") - off            # local midnight of the target day, in UTC
    start = day0 - pd.Timedelta(days=hc["spinup_days"])
    end = day0 + pd.Timedelta(days=1)
    dt = float(hc["dt_s"])

    model.init(at(start.timestamp())["temperature_2m"], hc["deep_offset_c"])
    cls = scene["surface_class"].values
    out_t, frames, air, class_rows = [], [], [], []
    t = start
    while t <= end:
        f = at(t.timestamp())
        if t >= day0 and (t - day0).total_seconds() % 3600 == 0:
            terms, _ = model.fluxes(f, sunlit(t), model.ts)
            out_t.append((t + off).tz_localize(None))
            frames.append(model.ts.copy())
            air.append(f["temperature_2m"])
            for name, code in C.CODE.items():
                m = cls == code
                if m.any():
                    class_rows.append({"time": out_t[-1], "class": name, "ts": float(model.ts[m].mean()),
                                       **{k: float(np.asarray(v)[m].mean()) for k, v in terms.items()}})
            if verbose and out_t[-1].hour % 3 == 0:
                print(f"  {out_t[-1]:%H:%M}  air {f['temperature_2m']:5.1f} C  surface mean {model.ts.mean():5.1f} C")
        model.step(f, sunlit(t), dt)
        t += pd.Timedelta(seconds=dt)

    ts = np.stack(frames[:24])
    ds = xr.Dataset(
        {"surface_temperature": (("time", "y", "x"), ts.astype("float32")),
         "surface_temperature_max": (("y", "x"), ts.max(axis=0)),
         "air_temperature": (("time",), np.array(air[:24]))},
        coords={"time": np.array(out_t[:24], dtype="datetime64[ns]"), "x": scene.x.values, "y": scene.y.values},
        attrs={**{k: scene.attrs[k] for k in ("name", "crs", "res_m", "transform")},
               "time_zone": sun.attrs["time_zone"], "forcing": f"ERA5 ({hc['era5_model']}) via Open-Meteo"},
    )
    for v in ("surface_temperature", "surface_temperature_max", "air_temperature"):
        ds[v].attrs["units"] = "degC"
    return ds, pd.DataFrame(class_rows)


def geohash_table(heat: xr.Dataset, cfg: dict) -> pd.DataFrame:
    grid = Grid.from_config(cfg)
    lon, lat = grid.lonlat()
    df = pd.DataFrame({"geohash": geohash.encode(lon.ravel(), lat.ravel(), cfg["geohash_precision"])})
    hours = pd.DatetimeIndex(heat.time.values).hour
    for h in cfg["heat"]["map_hours"]:
        df[f"ts_{h:02d}h"] = heat["surface_temperature"].values[list(hours).index(h)].ravel()
    df["ts_max"] = heat["surface_temperature_max"].values.ravel()
    out = df.groupby("geohash").mean()
    out.columns = [f"mean_{c}" for c in out.columns]
    return out.reset_index()


def summarize(heat: xr.Dataset, by_class: pd.DataFrame, cfg: dict) -> dict:
    hours = pd.DatetimeIndex(heat.time.values).hour
    out = {"date": cfg["sun"]["date"], "time_zone": heat.attrs["time_zone"],
           "air_temperature": {"min": round(float(heat.air_temperature.min()), 2),
                               "max": round(float(heat.air_temperature.max()), 2)},
           "by_class": {}}
    bc = by_class.copy()
    bc["hour"] = pd.DatetimeIndex(bc["time"]).hour
    for name, g in bc[bc["hour"] < 24].groupby("class", sort=False):
        g = g.drop_duplicates("hour").set_index("hour")
        row = {f"ts_{h:02d}h": round(float(g.loc[h, "ts"]), 2) for h in cfg["heat"]["map_hours"] if h in g.index}
        row["ts_max"] = round(float(g["ts"].max()), 2)
        h15 = g.loc[15] if 15 in g.index else None
        if h15 is not None:
            row["fluxes_15h_wm2"] = {k: round(float(h15[k]), 1) for k in ("k_abs", "l_net", "h", "le")}
        out["by_class"][name] = row
    out["scene_mean"] = {f"ts_{h:02d}h": round(float(heat.surface_temperature.values[list(hours).index(h)].mean()), 2)
                         for h in cfg["heat"]["map_hours"]}
    return out


def run(cfg: dict, verbose=True):
    scene = xr.open_zarr(cfg["paths"]["scene"]).load()
    sun = xr.open_zarr(cfg["paths"]["sun"]).load()
    forcing = load_forcing(cfg)
    heat, by_class = simulate(scene, sun, forcing, cfg, verbose)
    heat.to_zarr(out_path(cfg, "heat"), mode="w", zarr_format=2)
    geohash_table(heat, cfg).to_csv(out_path(cfg, "geohash_heat"), index=False, float_format="%.3f")
    by_class.to_csv(Path(cfg["paths"]["heat_summary"]).with_name("heat_by_class_hourly.csv"), index=False,
                    float_format="%.3f")
    summary = summarize(heat, by_class, cfg)
    with open(out_path(cfg, "heat_summary"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return heat, by_class, summary
