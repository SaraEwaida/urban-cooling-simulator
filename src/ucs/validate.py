"""Milestone 4: compare simulated surface temperature with Landsat at its overpass time."""
import copy
import json
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from . import classify as C
from . import heat, landsat, sun
from .config import out_path
from .grid import Grid


def block_mean(a: np.ndarray, k: int) -> np.ndarray:
    """Mean over k x k blocks, ignoring NaN (NaN where a block is entirely NaN)."""
    ny, nx = a.shape[0] // k, a.shape[1] // k
    b = a[: ny * k, : nx * k].reshape(ny, k, nx, k)
    if not np.isnan(b).any():
        return b.mean(axis=(1, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN (cloudy) blocks stay NaN
        return np.nanmean(b, axis=(1, 3))


def block_valid_share(a: np.ndarray, k: int) -> np.ndarray:
    ny, nx = a.shape[0] // k, a.shape[1] // k
    return (~np.isnan(a[: ny * k, : nx * k])).reshape(ny, k, nx, k).mean(axis=(1, 3))


def dominant_class(classes: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = classes.shape[0] // k, classes.shape[1] // k
    b = classes[: ny * k, : nx * k].reshape(ny, k, nx, k).transpose(0, 2, 1, 3).reshape(ny, nx, k * k)
    counts = np.stack([(b == code).mean(axis=2) for code in C.CODE.values()], axis=-1)
    return np.array(list(C.CODE.keys()))[counts.argmax(-1)], counts.max(-1)


def metrics(sim: np.ndarray, obs: np.ndarray) -> dict:
    d = sim - obs
    return {"n": int(d.size), "bias": float(d.mean()), "mae": float(np.abs(d).mean()),
            "rmse": float(np.sqrt((d ** 2).mean())), "rmse_bias_removed": float(d.std()),
            "r": float(np.corrcoef(sim, obs)[0, 1]) if d.size > 2 and sim.std() > 0 and obs.std() > 0 else float("nan"),
            "obs_std": float(obs.std()), "sim_std": float(sim.std())}


def compare(sim_2m: np.ndarray, lst_fine: np.ndarray, classes: np.ndarray, res: float, read_res: float,
            res_list, min_valid=0.9) -> tuple[dict, pd.DataFrame]:
    """Metrics at each comparison resolution, and the per-block table at the first one."""
    out, table = {}, None
    for r in res_list:
        ks, ko = int(r / res), int(r / read_res)
        s = block_mean(sim_2m, ks)
        o = block_mean(lst_fine, ko)
        ok = (block_valid_share(lst_fine, ko) >= min_valid) & ~np.isnan(o)
        dom, share = dominant_class(classes, ks)
        m = {"all": metrics(s[ok], o[ok])}
        for name in C.CODE:
            sel = ok & (dom == name)
            if sel.sum() >= 3:
                m[name] = metrics(s[sel], o[sel])
        out[f"{int(r)}m"] = m
        if table is None:
            iy, ix = np.nonzero(ok)
            table = pd.DataFrame({"row": iy, "col": ix, "sim": s[ok], "landsat": o[ok], "diff": s[ok] - o[ok],
                                  "dominant_class": dom[ok], "dominant_share": share[ok]})
    return out, table


def run(cfg: dict, verbose=True):
    grid = Grid.from_config(cfg)
    lc = cfg["landsat"]
    if verbose:
        print("Searching Landsat scenes:")
    item, lst, clear = landsat.best_scene(grid, cfg, verbose)
    off = pd.Timedelta(hours=cfg["sun"]["utc_offset_h"])
    t_utc = pd.Timestamp(item.datetime).tz_convert("UTC")
    local = (t_utc + off).tz_localize(None)

    # Run sun and heat for the Landsat date without touching the main-day outputs
    vcfg = copy.deepcopy(cfg)
    vcfg["sun"]["date"] = f"{local:%Y-%m-%d}"
    vcfg["paths"]["forcing"] = cfg["paths"]["forcing"].replace(".csv", f"_{local:%Y%m%d}.csv")
    scene = xr.open_zarr(cfg["paths"]["scene"]).load()
    if verbose:
        print(f"Simulating {local:%Y-%m-%d} up to the overpass at {local:%H:%M} local:")
    sun_ds = sun.compute(scene, vcfg, verbose=False)
    forcing = heat.load_forcing(vcfg)
    ds, _ = heat.simulate(scene, sun_ds, forcing, vcfg, verbose=False, snapshot_utc=t_utc)
    sim = ds["surface_temperature_snapshot"].values

    res_list = lc["compare_res_m"]
    result, table = compare(sim, lst, scene["surface_class"].values, grid.res, lc["read_res_m"], res_list)
    summary = {
        "landsat_item": item.id, "platform": item.properties.get("platform"),
        "overpass_utc": str(t_utc), "overpass_local": f"{local:%Y-%m-%d %H:%M}",
        "clear_share": round(clear, 4),
        "air_temperature_at_overpass": round(float(ds["surface_temperature_snapshot"].attrs["air_temperature"]), 2),
        "landsat_scene_mean": round(float(np.nanmean(lst)), 2),
        "sim_scene_mean": round(float(sim.mean()), 2),
        "metrics": {r: {k: {m: (round(v, 3) if isinstance(v, float) else v) for m, v in d.items()}
                        for k, d in res.items()} for r, res in result.items()},
    }
    with open(out_path(cfg, "validation"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    table.to_csv(out_path(cfg, "validation_blocks"), index=False, float_format="%.3f")
    return summary, sim, lst, table
