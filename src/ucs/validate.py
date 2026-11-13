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


def gaussian_blur(a: np.ndarray, sigma_px: float) -> np.ndarray:
    """Gaussian blur through the FFT, with mirrored edges so the scene border is not darkened."""
    if sigma_px <= 0:
        return a
    pad = int(np.ceil(3 * sigma_px))
    p = np.pad(a.astype("float64"), pad, mode="reflect")
    fy = np.fft.fftfreq(p.shape[0])[:, None]
    fx = np.fft.rfftfreq(p.shape[1])[None, :]
    h = np.exp(-2 * (np.pi * sigma_px) ** 2 * (fx ** 2 + fy ** 2))
    out = np.fft.irfft2(np.fft.rfft2(p) * h, s=p.shape)
    return out[pad:-pad, pad:-pad].astype("float32")


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
            res_list, min_valid=0.9, psf_sigma_m: float = 0.0) -> tuple[dict, pd.DataFrame]:
    """Metrics at each comparison resolution, and the per-block table at the first one.

    The simulation is first blurred with the sensor footprint (`psf_sigma_m`), then both are block-averaged.
    """
    sim_2m = gaussian_blur(sim_2m, psf_sigma_m / res)
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


def config_for_date(cfg: dict, local_date: str) -> dict:
    """A copy of the config for another day; its forcing is cached separately."""
    vcfg = copy.deepcopy(cfg)
    vcfg["sun"]["date"] = local_date
    vcfg["paths"]["forcing"] = cfg["paths"]["forcing"].replace(".csv", f"_{local_date.replace('-', '')}.csv")
    return vcfg


class Case:
    """One Landsat scene with everything needed to simulate its overpass."""

    def __init__(self, cfg: dict, item, lst: np.ndarray, clear: float, scene: xr.Dataset):
        off = pd.Timedelta(hours=cfg["sun"]["utc_offset_h"])
        self.item, self.lst, self.clear, self.scene = item, lst, clear, scene
        self.t_utc = pd.Timestamp(item.datetime).tz_convert("UTC")
        self.local = (self.t_utc + off).tz_localize(None)
        self.cfg = config_for_date(cfg, f"{self.local:%Y-%m-%d}")
        self.sun = sun.compute(scene, self.cfg, verbose=False)
        self.forcing = heat.load_forcing(self.cfg)
        self.air = float(np.interp(self.t_utc.timestamp(),
                                   (pd.to_datetime(self.forcing["time"], utc=True)
                                    - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds(),
                                   self.forcing["temperature_2m"]))

    def simulate(self, cfg_overrides=None, mask=None, base=None) -> np.ndarray:
        """Overpass surface temperature; with `mask`, only those pixels are rerun and the rest come from `base`."""
        c = copy.deepcopy(self.cfg)
        for path, value in (cfg_overrides or {}).items():
            d = c
            *keys, last = path.split(".")
            for k in keys:
                d = d[k]
            d[last] = value
        ts = heat.snapshot(self.scene, self.sun, self.forcing, c, self.t_utc, mask)
        if mask is None:
            return ts
        out = base.copy()
        out[mask] = ts
        return out

    def score(self, sim: np.ndarray) -> tuple[dict, pd.DataFrame]:
        lc = self.cfg["landsat"]
        return compare(sim, self.lst, self.scene["surface_class"].values, float(self.scene.attrs["res_m"]),
                       lc["read_res_m"], lc["compare_res_m"], psf_sigma_m=lc.get("psf_sigma_m", 0.0))

    def summary(self, result: dict, sim: np.ndarray) -> dict:
        return {
            "landsat_item": self.item.id, "platform": self.item.properties.get("platform"),
            "overpass_utc": str(self.t_utc), "overpass_local": f"{self.local:%Y-%m-%d %H:%M}",
            "clear_share": round(self.clear, 4), "air_temperature_at_overpass": round(self.air, 2),
            "landsat_scene_mean": round(float(np.nanmean(self.lst)), 2),
            "sim_scene_mean": round(float(sim.mean()), 2),
            "metrics": rounded(result),
        }


def rounded(result: dict) -> dict:
    return {r: {k: {m: (round(v, 3) if isinstance(v, float) else v) for m, v in d.items()}
                for k, d in res.items()} for r, res in result.items()}


def run(cfg: dict, verbose=True):
    grid = Grid.from_config(cfg)
    if verbose:
        print("Searching Landsat scenes:")
    item, lst, clear = landsat.clear_scenes(grid, cfg, 1, verbose)[0]
    scene = xr.open_zarr(cfg["paths"]["scene"]).load()
    case = Case(cfg, item, lst, clear, scene)
    if verbose:
        print(f"Simulating {case.local:%Y-%m-%d} up to the overpass at {case.local:%H:%M} local:")
    sim = case.simulate()
    result, table = case.score(sim)
    summary = case.summary(result, sim)
    with open(out_path(cfg, "validation"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    table.to_csv(out_path(cfg, "validation_blocks"), index=False, float_format="%.3f")
    return summary, sim, lst, table
