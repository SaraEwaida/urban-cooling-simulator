"""Milestone 5: cooling interventions compared with the calibrated baseline.

Each scenario edits a copy of the scene (geometry, classes or surface properties), recomputes
sun and shade when the geometry changes, reruns the energy balance for the same day and
weather, and reports the change in surface temperature. Scenarios:

* cool_roofs: roof albedo raised to a reflective membrane
* street_trees: shade trees along the outer band of the road surface
* shade_sails: fabric canopies over parking-lot interiors (they shade the ground; the ground stays paved)
* cool_pavement: reflective coating on roads
* porous_paving: moist porous paving on lots and sidewalks (paved surfaces off the road)
* combined: all of the above
"""
import json

import numpy as np
import pandas as pd
import xarray as xr

from . import classify as C
from . import geohash, heat, sun
from .config import out_path
from .grid import Grid


def _shift(a, dy, dx, fill=False):
    out = np.full_like(a, fill)
    ny, nx = a.shape
    out[max(0, dy):ny + min(0, dy), max(0, dx):nx + min(0, dx)] = \
        a[max(0, -dy):ny + min(0, -dy), max(0, -dx):nx + min(0, -dx)]
    return out


def erode(mask, r):
    out = mask.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out &= _shift(mask, dy, dx, fill=False)
    return out


def dilate(mask, r, disk=False):
    out = mask.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if disk and dy * dy + dx * dx > r * r:
                continue
            out |= _shift(mask, dy, dx, fill=False)
    return out


def tree_sites(road, classes, res, p):
    """Crown pixels for new street trees: one centre per spacing cell on the road's outer band."""
    band = road & ~erode(road, max(1, int(round(p["strip_m"] / res))))
    cand = band & (classes == C.CODE["paved"])
    s = max(1, int(round(p["spacing_m"] / res)))
    iy, ix = np.nonzero(cand)
    cell = (iy // s) * (classes.shape[1] // s + 1) + ix // s
    _, first = np.unique(cell, return_index=True)
    centres = np.zeros(classes.shape, bool)
    centres[iy[first], ix[first]] = True
    crowns = dilate(centres, max(1, int(round(p["crown_radius_m"] / res))), disk=True)
    # crowns may overhang ground of any kind, but not roofs or water, and existing trees stay as they are
    return crowns & np.isin(classes, [C.CODE[c] for c in ("paved", "bare_soil", "grass")]), int(first.size)


def parking_lots(road, classes, res, p):
    """Interiors of paved areas off the road (parking lots), by morphological opening."""
    off_road = (classes == C.CODE["paved"]) & ~road
    r = max(1, int(round(p["edge_m"] / res)))
    return dilate(erode(off_road, r), r) & off_road


class Edit:
    """A scenario: changed scene layers, property maps, inertia, sails, and the pixels changed."""

    def __init__(self, scene, props):
        self.classes = scene["surface_class"].values.copy()
        self.canopy = scene["canopy_height"].values.copy()
        self.props = {k: v.copy() for k, v in props.items()}
        self.inertia = np.full(self.classes.shape, np.nan, "float32")
        self.sail = None
        self.changed = np.zeros(self.classes.shape, bool)
        self.geometry = False
        self.notes = {}


def apply(name: str, edit: Edit, scene, cfg: dict):
    ic = cfg["interventions"]
    res = float(scene.attrs["res_m"])
    road = scene["road"].values.astype(bool)
    base_cls = scene["surface_class"].values
    if name == "cool_roofs":
        roof = edit.classes == C.CODE["roof"]
        new = np.maximum(edit.props["albedo"], ic["cool_roofs"]["albedo"])
        changed = roof & (new > edit.props["albedo"])
        edit.props["albedo"] = np.where(roof, new, edit.props["albedo"]).astype("float32")
        edit.changed |= changed
    elif name == "street_trees":
        p = ic["street_trees"]
        crowns, n = tree_sites(road, edit.classes, res, p)
        edit.classes[crowns] = C.CODE["tree"]
        edit.canopy[crowns] = p["height_m"]
        for k in ("albedo", "emissivity", "evap"):
            edit.props[k][crowns] = cfg["materials"]["tree"][k]
        edit.changed |= crowns
        edit.geometry = True
        edit.notes["trees_planted"] = n
    elif name == "shade_sails":
        p = ic["shade_sails"]
        lots = parking_lots(road, base_cls, res, p) & (edit.classes == C.CODE["paved"])
        edit.sail = np.where(lots, p["height_m"], 0.0).astype("float32") if edit.sail is None \
            else np.where(lots, p["height_m"], edit.sail).astype("float32")
        edit.changed |= lots
        edit.geometry = True
    elif name == "cool_pavement":
        m = road & (edit.classes == C.CODE["paved"])
        edit.props["albedo"] = np.where(m, np.maximum(edit.props["albedo"], ic["cool_pavement"]["albedo"]),
                                        edit.props["albedo"]).astype("float32")
        edit.changed |= m
    elif name == "porous_paving":
        p = ic["porous_paving"]
        m = ~road & (edit.classes == C.CODE["paved"])
        edit.props["evap"] = np.where(m, p["evap"], edit.props["evap"]).astype("float32")
        edit.inertia = np.where(m, p["thermal_inertia"], edit.inertia).astype("float32")
        edit.changed |= m
    else:
        raise ValueError(f"unknown intervention {name}")


def build(name: str, scene, cfg: dict, base_props) -> Edit:
    edit = Edit(scene, base_props)
    names = [n for n in cfg["interventions"]["scenarios"] if n != "combined"] if name == "combined" else [name]
    for n in names:
        apply(n, edit, scene, cfg)
    return edit


def run_scenario(edit: Edit, scene, base_sun, forcing, cfg, verbose=False):
    sc = scene.copy()
    sc["surface_class"] = (("y", "x"), edit.classes)
    sc["canopy_height"] = (("y", "x"), edit.canopy)
    if edit.geometry:
        p = cfg["interventions"]["shade_sails"]
        sun_ds = sun.compute(sc, cfg, verbose=False, sail_h=edit.sail, sail_transmissivity=p["transmissivity"],
                             svf_under_sail=p["svf_under"])
    else:
        sun_ds = base_sun
    ds, _ = heat.simulate(sc, sun_ds, forcing, cfg, verbose=verbose, props=edit.props,
                          inertia=None if np.isnan(edit.inertia).all() else edit.inertia)
    return ds["surface_temperature"].values, sun_ds


def run(cfg: dict, verbose=True):
    scene = xr.open_zarr(cfg["paths"]["scene"]).load()
    base_sun = xr.open_zarr(cfg["paths"]["sun"]).load()
    forcing = heat.load_forcing(cfg)
    props = heat.surface_properties(scene, cfg)
    if verbose:
        print("Baseline (calibrated):")
    base_ds, _ = heat.simulate(scene, base_sun, forcing, cfg, verbose=False, props=props)
    base = base_ds["surface_temperature"].values
    hours = list(pd.DatetimeIndex(base_ds.time.values).hour)
    ic = cfg["interventions"]
    cls = scene["surface_class"].values
    street = cls == C.CODE["paved"]  # pedestrian-level ground in the baseline: roads, sidewalks, lots
    res = float(scene.attrs["res_m"])

    rows, deltas = [], {}
    for name in ic["scenarios"]:
        edit = build(name, scene, cfg, props)
        if verbose:
            print(f"  {name}: {edit.changed.sum() * res * res / 1e4:.1f} ha changed"
                  + (f", {edit.notes['trees_planted']} trees" if "trees_planted" in edit.notes else ""))
        ts, sun_ds = run_scenario(edit, scene, base_sun, forcing, cfg)
        d = ts - base
        deltas[name] = d
        row = {"scenario": name, "changed_ha": round(float(edit.changed.sum() * res * res / 1e4), 2),
               "changed_share": round(float(edit.changed.mean()), 4), **edit.notes}
        for h in ic["map_hours"]:
            i = hours.index(h)
            row[f"d_scene_{h:02d}h"] = float(d[i].mean())
            row[f"d_street_{h:02d}h"] = float(d[i][street].mean())
            row[f"d_changed_{h:02d}h"] = float(d[i][edit.changed].mean()) if edit.changed.any() else 0.0
            row[f"d_unchanged_{h:02d}h"] = float(d[i][~edit.changed].mean())
        row["d_scene_daily_max"] = float((ts.max(axis=0) - base.max(axis=0)).mean())
        row["d_scene_daily_mean"] = float(d.mean())
        row["sunlit_hours_change"] = float((sun_ds["sunlit_hours"].values - base_sun["sunlit_hours"].values).mean())
        rows.append(row)
        if verbose:
            h0 = ic["map_hours"][0]
            print(f"     scene {row[f'd_scene_{h0:02d}h']:+.2f} C at {h0:02d}:00, streets {row[f'd_street_{h0:02d}h']:+.2f} C,"
                  f" changed pixels {row[f'd_changed_{h0:02d}h']:+.2f} C, daily mean {row['d_scene_daily_mean']:+.2f} C")

    table = pd.DataFrame(rows)
    table.to_csv(out_path(cfg, "interventions"), index=False, float_format="%.3f")
    gh = geohash_table(deltas, hours, cfg)
    gh.to_csv(out_path(cfg, "geohash_interventions"), index=False, float_format="%.3f")
    summary = {"date": cfg["sun"]["date"], "baseline_scene_mean": {f"{h:02d}h": round(float(base[hours.index(h)].mean()), 2)
                                                                   for h in ic["map_hours"]},
               "scenarios": table.round(3).to_dict(orient="records")}
    with open(out_path(cfg, "interventions_summary"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return table, deltas, hours, base_ds


def geohash_table(deltas: dict, hours: list, cfg: dict) -> pd.DataFrame:
    grid = Grid.from_config(cfg)
    lon, lat = grid.lonlat()
    df = pd.DataFrame({"geohash": geohash.encode(lon.ravel(), lat.ravel(), cfg["geohash_precision"])})
    for name, d in deltas.items():
        for h in cfg["interventions"]["map_hours"]:
            df[f"{name}_{h:02d}h"] = d[hours.index(h)].ravel()
    out = df.groupby("geohash").mean()
    out.columns = [f"d_{c}" for c in out.columns]
    return out.reset_index()
