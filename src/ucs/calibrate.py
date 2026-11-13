"""Calibrate vegetation transpiration on one Landsat scene and test on another.

Only two parameters are tuned: the minimum stomatal resistance of grass and of trees. The
search is a coordinate search (grass first, then trees) minimising the RMSE of 100 m blocks
on the calibration scene. Pixels are independent in the energy balance, so each trial reruns
only the pixels of the class being tuned. The held-out scene is scored once, before and after.
"""
import json

import numpy as np
import pandas as pd
import xarray as xr

from . import classify as C
from . import landsat
from .config import out_path
from .grid import Grid
from .validate import Case, rounded


def search(case: Case, cfg: dict, verbose=True):
    cal = cfg["calibration"]
    cls = case.scene["surface_class"].values
    current = case.simulate()
    baseline = current
    best = {"thermal.grass.r_s_min": cfg["thermal"]["grass"]["r_s_min"],
            "thermal.tree.r_s_min": cfg["thermal"]["tree"]["r_s_min"]}
    rows = []
    for key, values, name in [("thermal.grass.r_s_min", cal["grass_r_s_min"], "grass"),
                              ("thermal.tree.r_s_min", cal["tree_r_s_min"], "tree")]:
        mask = cls == C.CODE[name]
        trials = []
        for v in values:
            sim = case.simulate({**best, key: v}, mask=mask, base=current)
            res, _ = case.score(sim)
            m = res["100m"]["all"]
            cm = res["100m"].get(name, {})
            rows.append({"parameter": key, "value": v, "rmse": m["rmse"], "mae": m["mae"], "bias": m["bias"],
                         f"bias_{name}_blocks": cm.get("bias", np.nan), **{k: val for k, val in best.items() if k != key}})
            trials.append((m["rmse"], v, sim))
            if verbose:
                print(f"  {name} r_s_min {v:5g} s/m: RMSE {m['rmse']:.2f}  bias {m['bias']:+.2f}"
                      f"  {name}-block bias {cm.get('bias', float('nan')):+.2f}")
        _, v_best, current = min(trials, key=lambda x: x[0])
        best[key] = v_best
    return best, pd.DataFrame(rows), baseline, current


def run(cfg: dict, verbose=True):
    grid = Grid.from_config(cfg)
    scene = xr.open_zarr(cfg["paths"]["scene"]).load()
    if verbose:
        print("Searching two clear Landsat scenes:")
    (ci, cl, cc), (ti, tl, tc) = landsat.clear_scenes(grid, cfg, 2, verbose)
    cal = Case(cfg, ci, cl, cc, scene)
    test = Case(cfg, ti, tl, tc, scene)
    if verbose:
        print(f"Calibration scene {cal.local:%Y-%m-%d %H:%M}, held-out test scene {test.local:%Y-%m-%d %H:%M}")
        print("Coordinate search on the calibration scene:")

    best, grid_table, cal_base, cal_tuned = search(cal, cfg, verbose)
    out = {"calibration_scene": None, "test_scene": None, "tuned": best}
    sims = {}
    for label, case in (("calibration", cal), ("test", test)):
        base = cal_base if label == "calibration" else case.simulate()
        tuned = cal_tuned if label == "calibration" else case.simulate(best)
        rb, _ = case.score(base)
        rt, table = case.score(tuned)
        out[f"{label}_scene"] = {"baseline": case.summary(rb, base), "tuned": case.summary(rt, tuned)}
        sims[label] = (case, tuned, table, case.summary(rt, tuned))
        if verbose:
            b, t = rb["100m"]["all"], rt["100m"]["all"]
            print(f"{label:11s} {case.local:%Y-%m-%d}: RMSE {b['rmse']:.2f} -> {t['rmse']:.2f}, "
                  f"MAE {b['mae']:.2f} -> {t['mae']:.2f}, bias {b['bias']:+.2f} -> {t['bias']:+.2f}, "
                  f"r {b['r']:.2f} -> {t['r']:.2f}")
    with open(out_path(cfg, "calibration"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    grid_table.to_csv(out_path(cfg, "calibration_grid"), index=False, float_format="%.3f")
    return out, sims
