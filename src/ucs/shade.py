"""Shadows and sky view factor on a height raster, by marching along rays.

Heights are above local ground; the scene is treated as flat (the 30 m terrain model varies
by a few metres across 2 km of central Phoenix, far less than the buildings and trees).
Directions use azimuth clockwise from north; grid rows run north to south.
"""
import numpy as np


def _shifted(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """out[y, x] = a[y + dy, x + dx], zero outside the grid (open ground beyond the scene)."""
    out = np.zeros_like(a)
    ny, nx = a.shape
    ys, ye = max(0, -dy), min(ny, ny - dy)
    xs, xe = max(0, -dx), min(nx, nx - dx)
    if ys < ye and xs < xe:
        out[ys:ye, xs:xe] = a[ys + dy:ye + dy, xs + dx:xe + dx]
    return out


def _ray_offsets(azimuth_deg: float, max_dist_m: float, res: float):
    """Unique integer pixel offsets along a ray towards `azimuth`, with their distances in metres."""
    a = np.radians(azimuth_deg)
    ux, uy = np.sin(a), -np.cos(a)  # east, and rows (south positive)
    n = int(np.ceil(max_dist_m / res))
    seen, out = set(), []
    for k in np.arange(1, n + 1, 0.5):  # half-pixel steps so diagonal rays skip nothing
        dx, dy = int(round(k * ux)), int(round(k * uy))
        if (dx, dy) in seen or (dx, dy) == (0, 0):
            continue
        seen.add((dx, dy))
        out.append((dy, dx, res * float(np.hypot(dx, dy))))
    return out


def sunlit_fraction(building_h, tree_h, elevation_deg, azimuth_deg, res, tree_transmissivity=0.25,
                    max_dist_m=300.0, sail_h=None, sail_transmissivity=0.1) -> np.ndarray:
    """Fraction of direct sunlight reaching each pixel's top surface (roof, canopy or ground).

    1 = full sun, `tree_transmissivity` = under or behind a tree crown, 0 = behind a building
    (or the sun is below the horizon). Optional shade sails (`sail_h`, height of a fabric canopy
    over the ground) shade the ground under them and cast shadows like a thin roof that lets
    `sail_transmissivity` through; the surface under a sail is the ground.
    """
    hb = np.nan_to_num(building_h, nan=0.0).astype("float32")
    ht = np.nan_to_num(tree_h, nan=0.0).astype("float32")
    hs = np.zeros_like(hb) if sail_h is None else np.nan_to_num(sail_h, nan=0.0).astype("float32")
    surface = np.maximum(hb, ht)
    if elevation_deg <= 0:
        return np.zeros(surface.shape, "float32")
    tan_el = np.tan(np.radians(elevation_deg))
    reach = min(max_dist_m, float(max(surface.max(), hs.max())) / tan_el + res)
    by_building = np.zeros(surface.shape, bool)
    by_tree = np.zeros(surface.shape, bool)
    by_sail = hs > surface  # directly under a sail
    for dy, dx, d in _ray_offsets(azimuth_deg, reach, res):
        rise = d * tan_el
        by_building |= _shifted(hb, dy, dx) - surface > rise
        by_tree |= _shifted(ht, dy, dx) - surface > rise
        if sail_h is not None:
            by_sail |= _shifted(hs, dy, dx) - surface > rise
    out = np.ones(surface.shape, "float32")
    out[by_tree] = tree_transmissivity
    out[by_sail] = np.minimum(out[by_sail], sail_transmissivity)
    out[by_building] = 0.0
    return out


def sky_view_factor(height, res, n_directions=32, max_dist_m=100.0) -> np.ndarray:
    """Sky view factor of each pixel's top surface: 1/(2 pi) * integral of cos^2(horizon angle).

    Horizon angles come from the tallest obstruction within `max_dist_m` in each direction;
    tree crowns count as solid here.
    """
    h = np.nan_to_num(height, nan=0.0).astype("float32")
    acc = np.zeros(h.shape, "float64")
    for az in np.arange(n_directions) * 360.0 / n_directions:
        tan_max = np.zeros(h.shape, "float32")
        for dy, dx, d in _ray_offsets(az, max_dist_m, res):
            # Distance to the near edge of the obstructing pixel, not its centre
            np.maximum(tan_max, (_shifted(h, dy, dx) - h) / (d - 0.5 * res), out=tan_max)
        acc += 1.0 / (1.0 + tan_max.astype("float64") ** 2)  # cos^2(atan(t))
    return (acc / n_directions).astype("float32")
