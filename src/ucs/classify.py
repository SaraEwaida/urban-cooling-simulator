"""Surface classes and their radiative properties.

Classes are assigned in a fixed order so that what is visible from above wins:
ground (paved or bare by brightness) -> roads -> grass -> trees -> water -> buildings.
Road buffers only pave unvegetated pixels, since the buffer widths are approximate and
verges or canopy inside them are what the imagery actually shows.
"""
import numpy as np

CLASSES = ["roof", "paved", "bare_soil", "grass", "tree", "water"]
CODE = {name: i + 1 for i, name in enumerate(CLASSES)}  # 0 = no data


def ndvi(naip: np.ndarray) -> np.ndarray:
    r, nir = naip[0], naip[3]
    with np.errstate(invalid="ignore", divide="ignore"):
        return ((nir - r) / (nir + r)).astype("float32")


def brightness(naip: np.ndarray) -> np.ndarray:
    return naip[:3].mean(axis=0).astype("float32")


def classify(naip, hag, building_ids, road_mask, cfg_classify: dict, water_mask=None) -> tuple[np.ndarray, dict]:
    """Return the class raster (uint8, codes in CODE) and notes on how it was made.

    `hag` is vegetation height above ground (lidar or a canopy height map); NaN where unknown.
    """
    c = cfg_classify
    v = ndvi(naip)
    b = brightness(naip)
    valid = ~np.isnan(v)
    have_lidar = ~np.isnan(hag)
    veg = v >= c["veg_min_ndvi"]

    out = np.zeros(v.shape, dtype="uint8")
    out[valid] = CODE["bare_soil"]
    out[valid & (b <= c["paved_max_brightness"])] = CODE["paved"]
    out[valid & road_mask] = CODE["paved"]
    out[valid & veg] = CODE["grass"]
    # Trees: tall vegetation; where no height is known, fall back to the greenest vegetation
    tall = np.where(have_lidar, hag >= c["tree_min_height_m"], v >= c.get("tree_fallback_ndvi", 0.35))
    # Tree crowns are darker than irrigated turf (self-shading); this catches trees a canopy model misses
    if c.get("tree_max_brightness") is not None:
        tall = tall | (b <= c["tree_max_brightness"])
    out[valid & veg & tall] = CODE["tree"]
    water = (v <= c["water_max_ndvi"]) & (b <= c["water_max_brightness"])
    if water_mask is not None:
        water = water | water_mask
    out[valid & water] = CODE["water"]
    out[building_ids > 0] = CODE["roof"]
    notes = {"lidar_fraction": float(have_lidar.mean())}
    return out, notes


def building_heights(building_ids, hag, overture_height, default: float) -> np.ndarray:
    """Per-pixel roof height: median lidar height within each footprint, else Overture, else default."""
    out = np.full(building_ids.shape, np.nan, dtype="float32")
    inside = building_ids > 0
    if not inside.any():
        return out
    ids = building_ids[inside]
    vals = hag[inside]
    ok = ~np.isnan(vals) & (vals > 0.5)
    med = np.full(ids.max() + 1, np.nan, dtype="float32")
    if ok.any():
        order = np.argsort(ids[ok], kind="stable")
        sid, sval = ids[ok][order], vals[ok][order]
        starts = np.r_[0, np.flatnonzero(np.diff(sid)) + 1]
        for s, e in zip(starts, np.r_[starts[1:], len(sid)]):
            med[sid[s]] = np.median(sval[s:e])
    h = med[ids]
    h = np.where(np.isnan(h), overture_height[inside], h)
    h = np.where(np.isnan(h), default, h)
    out[inside] = h
    return out


def canopy_height(classes, hag) -> np.ndarray:
    out = np.full(classes.shape, np.nan, dtype="float32")
    t = classes == CODE["tree"]
    out[t] = np.where(np.isnan(hag[t]), 6.0, hag[t])
    return out


def material_maps(classes, materials: dict) -> dict[str, np.ndarray]:
    maps = {}
    for prop in ("albedo", "emissivity", "evap"):
        lut = np.full(len(CLASSES) + 1, np.nan, dtype="float32")
        for name, code in CODE.items():
            lut[code] = materials[name][prop]
        maps[prop] = lut[classes]
    return maps
