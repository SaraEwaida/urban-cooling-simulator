"""Milestone 2: hourly sun and shade over one day, and the sky view factor."""
import json

import numpy as np
import pandas as pd
import xarray as xr

from . import classify as C
from . import geohash, shade, solar
from .config import out_path
from .grid import Grid


def day_times(sc: dict) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    local = pd.DatetimeIndex([pd.Timestamp(sc["date"]) + pd.Timedelta(hours=h) for h in sc["hours_local"]])
    return local, local - pd.Timedelta(hours=sc["utc_offset_h"])


def compute(scene: xr.Dataset, cfg: dict, verbose: bool = True, sail_h=None, sail_transmissivity=0.1,
            svf_under_sail=0.3) -> xr.Dataset:
    sc = cfg["sun"]
    res = float(scene.attrs["res_m"])
    lon, lat = cfg["center"]
    local, utc = day_times(sc)
    elev, az = solar.sun_position(utc, lat, lon)
    hb = scene["building_height"].values
    ht = scene["canopy_height"].values

    frames = []
    for t, e, a in zip(local, elev, az):
        f = shade.sunlit_fraction(hb, ht, e, a, res, sc["tree_transmissivity"], sc["max_shadow_m"],
                                  sail_h, sail_transmissivity)
        frames.append(np.round(f * 100).astype("uint8"))
        if verbose:
            print(f"  {t:%H:%M} sun elevation {e:5.1f} deg, azimuth {a:5.1f} deg, sunlit {f.mean():.0%}")
    sunlit = np.stack(frames)
    svf = shade.sky_view_factor(np.fmax(np.nan_to_num(hb), np.nan_to_num(ht)), res,
                                sc["svf_directions"], sc["svf_radius_m"])
    if sail_h is not None:
        under = np.nan_to_num(sail_h) > 0
        svf = np.where(under, np.minimum(svf, svf_under_sail), svf).astype("float32")
    hours = sunlit.astype("float32").sum(axis=0) / 100.0  # one-hour steps

    ds = xr.Dataset(
        {
            "sunlit": (("time", "y", "x"), sunlit),
            "sunlit_hours": (("y", "x"), hours.astype("float32")),
            "sky_view_factor": (("y", "x"), svf),
            "sun_elevation": (("time",), elev),
            "sun_azimuth": (("time",), az),
        },
        coords={"time": local.values, "x": scene.x.values, "y": scene.y.values},
        attrs={**{k: scene.attrs[k] for k in ("name", "crs", "res_m", "transform")},
               "time_zone": f"UTC{sc['utc_offset_h']:+d}",
               "tree_transmissivity": sc["tree_transmissivity"]},
    )
    ds["sunlit"].attrs.update(units="percent", description="share of direct sun reaching the top surface")
    ds["sunlit_hours"].attrs["units"] = "h"
    ds["sun_elevation"].attrs["units"] = "degree"
    ds["sun_azimuth"].attrs["units"] = "degree clockwise from north"
    return ds


def geohash_table(sun: xr.Dataset, scene: xr.Dataset, cfg: dict) -> pd.DataFrame:
    grid = Grid.from_config(cfg)
    lon, lat = grid.lonlat()
    gh = geohash.encode(lon.ravel(), lat.ravel(), cfg["geohash_precision"])
    cls = scene["surface_class"].values.ravel()
    df = pd.DataFrame({"geohash": gh,
                       "svf": sun["sky_view_factor"].values.ravel(),
                       "sunlit_hours": sun["sunlit_hours"].values.ravel()})
    ground = np.isin(cls, [C.CODE[c] for c in ("paved", "bare_soil", "grass")])
    df["ground_sunlit_hours"] = np.where(ground, df["sunlit_hours"], np.nan)
    for h in cfg["sun"]["map_hours"]:
        i = cfg["sun"]["hours_local"].index(h)
        df[f"shade_{h:02d}h"] = 1 - sun["sunlit"].values[i].ravel() / 100.0
    g = df.groupby("geohash")
    out = g.mean(numeric_only=True)
    out.columns = [f"mean_{c}" for c in out.columns]
    return out.reset_index()


def run(cfg: dict, verbose: bool = True) -> xr.Dataset:
    scene = xr.open_zarr(cfg["paths"]["scene"]).load()
    sun = compute(scene, cfg, verbose)
    sun.to_zarr(out_path(cfg, "sun"), mode="w", zarr_format=2)
    table = geohash_table(sun, scene, cfg)
    table.to_csv(out_path(cfg, "geohash_sun"), index=False, float_format="%.4f")
    with open(out_path(cfg, "sun_summary"), "w", encoding="utf-8") as f:
        json.dump(summarize(sun, scene, cfg), f, indent=2)
    return sun


def summarize(sun: xr.Dataset, scene: xr.Dataset, cfg: dict) -> dict:
    cls = scene["surface_class"].values
    out = {"date": cfg["sun"]["date"], "time_zone": sun.attrs["time_zone"],
           "hours": {}, "by_class": {}}
    for i, t in enumerate(pd.DatetimeIndex(sun.time.values)):
        out["hours"][f"{t:%H:%M}"] = {
            "sun_elevation_deg": round(float(sun.sun_elevation[i]), 2),
            "sun_azimuth_deg": round(float(sun.sun_azimuth[i]), 2),
            "sunlit_share": round(float(sun.sunlit.values[i].mean() / 100), 4),
        }
    for name, code in C.CODE.items():
        m = cls == code
        if m.any():
            out["by_class"][name] = {
                "sunlit_hours": round(float(sun.sunlit_hours.values[m].mean()), 2),
                "sky_view_factor": round(float(sun.sky_view_factor.values[m].mean()), 3),
            }
    out["scene_mean"] = {"sunlit_hours": round(float(sun.sunlit_hours.values.mean()), 2),
                         "sky_view_factor": round(float(sun.sky_view_factor.values.mean()), 3)}
    return out
