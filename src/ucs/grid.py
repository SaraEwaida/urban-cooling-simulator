"""The scene grid: a square, north-up raster in a projected CRS, centred on a lon/lat point."""
from dataclasses import dataclass

import numpy as np
from pyproj import Transformer
from rasterio.transform import from_origin


@dataclass(frozen=True)
class Grid:
    x0: float  # left edge (m)
    y1: float  # top edge (m)
    res: float
    nx: int
    ny: int
    crs: str

    @classmethod
    def from_config(cls, cfg: dict) -> "Grid":
        lon, lat = cfg["center"]
        res, size = float(cfg["res_m"]), float(cfg["size_m"])
        x, y = Transformer.from_crs("EPSG:4326", cfg["crs"], always_xy=True).transform(lon, lat)
        n = int(round(size / res))
        # Snap the corner to the resolution so grids built from the same config always line up
        x0 = np.floor((x - size / 2) / res) * res
        y1 = np.floor((y + size / 2) / res) * res
        return cls(float(x0), float(y1), res, n, n, cfg["crs"])

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ny, self.nx)

    @property
    def transform(self):
        return from_origin(self.x0, self.y1, self.res, self.res)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y1 - self.ny * self.res, self.x0 + self.nx * self.res, self.y1)

    def coords(self) -> tuple[np.ndarray, np.ndarray]:
        """Pixel-centre x (columns) and y (rows, north to south)."""
        x = self.x0 + (np.arange(self.nx) + 0.5) * self.res
        y = self.y1 - (np.arange(self.ny) + 0.5) * self.res
        return x, y

    def lonlat(self) -> tuple[np.ndarray, np.ndarray]:
        """Pixel-centre lon/lat, both shaped like the grid."""
        x, y = self.coords()
        xx, yy = np.meshgrid(x, y)
        t = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        lon, lat = t.transform(xx, yy)
        return np.asarray(lon), np.asarray(lat)

    def bbox_wgs84(self, margin_m: float = 50.0) -> tuple[float, float, float, float]:
        """Lon/lat box enclosing the grid plus a margin (for querying data sources)."""
        x0, y0, x1, y1 = self.bounds
        xs = np.array([x0 - margin_m, x1 + margin_m, x1 + margin_m, x0 - margin_m])
        ys = np.array([y0 - margin_m, y0 - margin_m, y1 + margin_m, y1 + margin_m])
        lon, lat = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True).transform(xs, ys)
        return (float(min(lon)), float(min(lat)), float(max(lon)), float(max(lat)))
