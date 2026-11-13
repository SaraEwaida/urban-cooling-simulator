"""Scene preview figure."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from . import classify as C
from . import geohash
from .grid import Grid

# Land-cover colours that read naturally; neighbours differ strongly in lightness
CLASS_COLORS = {
    "roof": "#9c3d2e",
    "paved": "#4d4d4d",
    "bare_soil": "#d9c49a",
    "grass": "#a6d96a",
    "tree": "#1a6b30",
    "water": "#2c7fb8",
}


def _save(fig, path):
    """Save, or save next to it if Windows has the file locked (e.g. open in a viewer)."""
    from pathlib import Path

    path = Path(path)
    try:
        fig.savefig(path, dpi=110)
    except OSError:
        alt = path.with_name(f"{path.stem}_new{path.suffix}")
        fig.savefig(alt, dpi=110)
        print(f"{path} is open in another program; saved {alt} instead")
        path = alt
    plt.close(fig)
    return path


def _extent(grid: Grid):
    x0, y0, x1, y1 = grid.bounds
    return [(x0 - x0) / 1000, (x1 - x0) / 1000, 0, (y1 - y0) / 1000]


def scene_figure(ds, cfg: dict, path, table=None):
    grid = Grid.from_config(cfg)
    ext = _extent(grid)
    fig, axes = plt.subplots(2, 2, figsize=(12, 11.5), constrained_layout=True)

    # 1. NAIP true colour (2 % - 98 % stretch)
    ax = axes[0, 0]
    rgb = np.moveaxis(ds["naip"].values[:3], 0, -1)
    lo, hi = np.nanpercentile(rgb, [2, 98])
    ax.imshow(np.clip((rgb - lo) / (hi - lo), 0, 1), extent=ext)
    meta = ds.attrs.get("sources", "{}")
    import json
    naip_year = json.loads(meta).get("naip", {}).get("year", "")
    ax.set_title(f"NAIP {naip_year} aerial image (resampled to {grid.res:g} m)")

    # 2. Surface classes
    ax = axes[0, 1]
    cmap = ListedColormap(["white"] + [CLASS_COLORS[c] for c in C.CLASSES])
    norm = BoundaryNorm(np.arange(-0.5, len(C.CLASSES) + 1.5), cmap.N)
    ax.imshow(ds["surface_class"].values, cmap=cmap, norm=norm, extent=ext, interpolation="nearest")
    cls = ds["surface_class"].values
    valid = cls > 0
    handles = [Patch(color=CLASS_COLORS[c], label=f"{c.replace('_', ' ')} {np.mean(cls[valid] == C.CODE[c]):.0%}")
               for c in C.CLASSES]
    ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.9)
    ax.set_title("Surface class")

    # 3. Heights: buildings and tree canopy
    ax = axes[1, 0]
    h = np.fmax(np.nan_to_num(ds["building_height"].values, nan=0), np.nan_to_num(ds["canopy_height"].values, nan=0))
    h = np.where(h > 0, h, np.nan)
    ax.set_facecolor("#f2f2f2")
    im = ax.imshow(h, cmap="viridis", vmin=0, vmax=np.nanpercentile(h, 99), extent=ext, interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.8, label="height above ground (m)")
    roof_src = "lidar" if ds.attrs.get("lidar_fraction", 0) > 0.5 else "Overture"
    ax.set_title(f"Height: roofs from {roof_src}, trees from {ds.attrs.get('canopy_source', 'none')}", fontsize=10)

    # 4. Tree cover per geohash cell
    ax = axes[1, 1]
    lon, lat = grid.lonlat()
    gh = geohash.encode(lon.ravel(), lat.ravel(), cfg["geohash_precision"]).reshape(grid.shape)
    if table is None:
        import pandas as pd
        table = pd.read_csv(cfg["paths"]["geohash_table"])
    lut = dict(zip(table["geohash"], table["frac_tree"]))
    tree = np.vectorize(lambda g: lut.get(g, np.nan), otypes=[float])(gh)
    im = ax.imshow(tree * 100, cmap="Greens", vmin=0, vmax=max(5, np.nanpercentile(tree * 100, 99)),
                   extent=ext, interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.8, label="tree canopy cover (%)")
    ax.set_title(f"Tree cover per geohash cell (precision {cfg['geohash_precision']}, {len(table)} cells)")

    for ax in axes.flat:
        ax.set_xlabel("km east")
        ax.set_ylabel("km north")
    fig.suptitle(f"{cfg['name'].title()} scene, {grid.nx * grid.res / 1000:g} x {grid.ny * grid.res / 1000:g} km, "
                 f"{grid.crs}", fontsize=13)
    return _save(fig, path)


def sun_figure(sun, cfg: dict, path):
    import pandas as pd

    grid = Grid.from_config(cfg)
    ext = _extent(grid)
    fig, axes = plt.subplots(2, 2, figsize=(12, 11.5), constrained_layout=True)
    times = pd.DatetimeIndex(sun.time.values)
    shade_cmap = ListedColormap(["#1f1f1f", "#7a8f6a", "#f4e3b5"])
    shade_norm = BoundaryNorm([-1, 1, 99, 101], 3)
    for ax, h in zip(axes[0], cfg["sun"]["map_hours"]):
        i = cfg["sun"]["hours_local"].index(h)
        s = sun["sunlit"].values[i]
        ax.imshow(s, cmap=shade_cmap, norm=shade_norm, extent=ext, interpolation="nearest")
        el, az = float(sun.sun_elevation[i]), float(sun.sun_azimuth[i])
        ax.set_title(f"{times[i]:%H:%M} local: sun {el:.0f} deg up, from {az:.0f} deg")
        shares = [np.mean(s == 0), np.mean((s > 0) & (s < 100)), np.mean(s == 100)]
        ax.legend(handles=[Patch(color=c, label=f"{l} {v:.0%}") for c, l, v in
                           zip(shade_cmap.colors, ["building shadow", "tree shade", "full sun"], shares)],
                  loc="lower left", fontsize=8, framealpha=0.9)

    ax = axes[1, 0]
    im = ax.imshow(sun["sunlit_hours"].values, cmap="YlOrRd", vmin=0, extent=ext, interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.8, label="hours of direct sun")
    ax.set_title(f"Direct sun over the day, {cfg['sun']['date']}")

    ax = axes[1, 1]
    im = ax.imshow(sun["sky_view_factor"].values, cmap="cividis", vmin=0, vmax=1, extent=ext,
                   interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.8, label="sky view factor")
    ax.set_title("Sky view factor (1 = open sky)")

    for ax in axes.flat:
        ax.set_xlabel("km east")
        ax.set_ylabel("km north")
    fig.suptitle(f"{cfg['name'].title()} sun and shade, {cfg['sun']['date']} ({sun.attrs['time_zone']})",
                 fontsize=13)
    return _save(fig, path)


def heat_figure(heat, by_class, cfg: dict, path):
    import pandas as pd

    grid = Grid.from_config(cfg)
    ext = _extent(grid)
    fig, axes = plt.subplots(2, 2, figsize=(12, 11.5), constrained_layout=True)
    hours = list(pd.DatetimeIndex(heat.time.values).hour)
    ts = heat["surface_temperature"].values
    day_h, night_h = [h for h in cfg["heat"]["map_hours"] if 8 <= h <= 18], \
                     [h for h in cfg["heat"]["map_hours"] if not 8 <= h <= 18]
    day = np.stack([ts[hours.index(h)] for h in day_h])
    vmin, vmax = np.percentile(day, [1, 99])
    for ax, h in zip([axes[0, 0], axes[0, 1]], day_h[:2]):
        i = hours.index(h)
        im = ax.imshow(ts[i], cmap="inferno", vmin=vmin, vmax=vmax, extent=ext, interpolation="nearest")
        ax.set_title(f"Surface temperature {h:02d}:00 (air {float(heat.air_temperature[i]):.1f} C)")
    fig.colorbar(im, ax=axes[0, :], shrink=0.8, label="surface temperature (C), same scale")
    if night_h:
        ax = axes[1, 0]
        i = hours.index(night_h[0])
        lo, hi = np.percentile(ts[i], [1, 99])
        im = ax.imshow(ts[i], cmap="inferno", vmin=lo, vmax=hi, extent=ext, interpolation="nearest")
        fig.colorbar(im, ax=ax, shrink=0.8, label="surface temperature (C), night scale")
        ax.set_title(f"Surface temperature {night_h[0]:02d}:00 (air {float(heat.air_temperature[i]):.1f} C)")
    for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
        ax.set_xlabel("km east")
        ax.set_ylabel("km north")

    ax = axes[1, 1]
    bc = by_class.copy()
    bc["hour"] = pd.DatetimeIndex(bc["time"]).hour
    bc = bc.drop_duplicates(["class", "hour"])
    for name in C.CLASSES:
        g = bc[bc["class"] == name].sort_values("hour")
        if len(g):
            ax.plot(g["hour"], g["ts"], color=CLASS_COLORS[name], lw=2, label=name.replace("_", " "))
    ax.plot(hours, heat.air_temperature.values, color="black", lw=2, ls="--", label="air (ERA5)")
    ax.set_xlim(0, 23)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel(f"hour ({heat.attrs['time_zone']})")
    ax.set_ylabel("mean surface temperature (C)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("Daily cycle by surface class")
    fig.suptitle(f"{cfg['name'].title()} surface temperature, {cfg['sun']['date']}", fontsize=13)
    return _save(fig, path)


def validation_figure(summary: dict, sim, lst, table, cfg: dict, path):
    from .validate import block_mean, block_valid_share

    grid = Grid.from_config(cfg)
    ext = _extent(grid)
    r = cfg["landsat"]["compare_res_m"][0]
    ks, ko = int(r / grid.res), int(r / cfg["landsat"]["read_res_m"])
    s = block_mean(sim, ks)
    o = block_mean(lst, ko)
    o = np.where(block_valid_share(lst, ko) >= 0.9, o, np.nan)
    s = np.where(np.isnan(o), np.nan, s)
    lo, hi = np.nanpercentile(np.r_[s.ravel(), o.ravel()], [1, 99])

    fig, axes = plt.subplots(2, 2, figsize=(12, 11.5), constrained_layout=True)
    for ax, a, title in [(axes[0, 0], o, f"Landsat surface temperature ({summary['platform']})"),
                         (axes[0, 1], s, "Simulated surface temperature")]:
        im = ax.imshow(a, cmap="inferno", vmin=lo, vmax=hi, extent=ext, interpolation="nearest")
        ax.set_title(f"{title}, {r} m blocks")
    fig.colorbar(im, ax=axes[0, :], shrink=0.8, label="surface temperature (C), same scale")

    ax = axes[1, 0]
    d = s - o
    lim = float(np.nanpercentile(np.abs(d), 98))
    im = ax.imshow(d, cmap="RdBu_r", vmin=-lim, vmax=lim, extent=ext, interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.8, label="simulated - Landsat (C)")
    m = summary["metrics"][f"{r}m"]["all"]
    ax.set_title(f"Difference: bias {m['bias']:+.1f} C, MAE {m['mae']:.1f} C, r {m['r']:.2f}")
    for a in (axes[0, 0], axes[0, 1], axes[1, 0]):
        a.set_xlabel("km east")
        a.set_ylabel("km north")

    ax = axes[1, 1]
    for name in C.CLASSES:
        g = table[table["dominant_class"] == name]
        if len(g):
            ax.scatter(g["landsat"], g["sim"], s=14, color=CLASS_COLORS[name], edgecolor="white", linewidth=0.4,
                       label=f"{name.replace('_', ' ')} ({len(g)})")
    both = np.r_[table["landsat"], table["sim"]]
    a0, a1 = both.min() - 1, both.max() + 1
    ax.plot([a0, a1], [a0, a1], color="black", lw=1, ls="--", label="1:1")
    ax.set_xlim(a0, a1)
    ax.set_ylim(a0, a1)
    ax.set_aspect("equal")
    ax.set_xlabel("Landsat (C)")
    ax.set_ylabel("simulated (C)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, title="dominant class of block", title_fontsize=8)
    ax.set_title(f"{r} m blocks by dominant surface class")
    fig.suptitle(f"{cfg['name'].title()}: simulation vs Landsat at the overpass, {summary['overpass_local']} "
                 f"local (air {summary['air_temperature_at_overpass']:.1f} C)", fontsize=13)
    return _save(fig, path)
