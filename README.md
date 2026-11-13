# urban-cooling-simulator

Physics-based simulation of street-level heat in a Phoenix neighbourhood: building and tree shading, surface energy balance and wind sheltering, and the cooling effect of trees, shade structures, cool roofs and porous paving, validated against Landsat surface temperature.

The study area is a 2 x 2 km square around Encanto Park in central Phoenix: a park with a lagoon, tree-lined historic residential streets, commercial corridors and large parking lots, so every surface type the simulator handles appears in one scene.

## Status

| Milestone | Content | State |
|---|---|---|
| 1. Scene builder | Buildings, roads, aerial imagery and lidar on a 2 m grid; surface classes, heights and materials; geohash summary; Zarr | done |
| 2. Sun and shade | Sun position, building and tree shadows, sky view factor | next |
| 3. Energy balance | Surface temperature from radiation, convection, storage and evaporation, with ERA5 forcing and wind sheltering | planned |
| 4. Validation | Simulated surface temperature against Landsat Collection 2 surface temperature | planned |
| 5. Interventions | Trees, shade sails, cool roofs, porous paving; cooling maps and per-geohash tables | planned |

## Data (all free, no account)

| Layer | Source | Use |
|---|---|---|
| Building footprints and heights | Overture Maps `building` (via the `overturemaps` package) | roofs, fallback heights (height, else floors x 3.5 m) |
| Road centre lines | Overture Maps `segment`, subtype `road` | paved surface, buffered by road class |
| Aerial imagery, 0.6 m RGB + NIR | USDA NAIP on Microsoft Planetary Computer | NDVI, brightness, surface class |
| Height above ground, 2 m | USGS 3DEP lidar `3dep-lidar-hag` on Planetary Computer | tree vs grass, roof heights (where available) |
| Canopy height, about 1 m | Meta/WRI High Resolution Canopy Height Maps v1 on AWS Open Data (no account) | tree vs grass and tree height when there is no lidar |
| Terrain | 3DEP lidar `3dep-lidar-dtm`, else Copernicus DEM GLO-30 (`cop-dem-glo-30`) on Planetary Computer | elevation |

Planetary Computer is read anonymously; only the part of each cloud-optimised GeoTIFF inside the scene is downloaded, warped straight onto the grid (NAIP averaged from 0.6 m to 2 m, lidar bilinear). Planetary Computer's 3DEP lidar collection has no coverage over Phoenix (checked for the whole metro area), so for Encanto the scene uses the fallbacks: tree height from the Meta/WRI canopy height map, roof height from Overture, and terrain from the 30 m Copernicus DEM. The canopy map is a model estimate from older Maxar imagery, not a measurement, and NAIP is from 2023, so trees planted or removed in between are misclassified. The code still prefers lidar for any scene where it exists. The sources and dates used are recorded in `outputs/scene_summary.json`.

## Method (milestone 1)

1. **Grid.** A square, north-up 2 m grid in UTM 12N (EPSG:32612), corner snapped to the resolution, 1000 x 1000 pixels.
2. **Surface classes**, applied in order so that what is visible from above wins:
   - ground: bare soil, or paved if dark (asphalt, concrete, parking lots);
   - road buffers pave unvegetated pixels (verges and canopy inside the buffer keep their class);
   - vegetation (NDVI >= 0.20) is grass, or tree where its height (lidar, else canopy height map) is >= 2 m;
   - water: negative NDVI and dark;
   - building footprints are roof.
   With no height source at all, trees fall back to the greenest vegetation (NDVI >= 0.35) at a nominal 6 m.
3. **Heights.** Roof height is the median lidar height inside each footprint, else the Overture height (or floors x 3.5 m), else 4 m; the canopy height map is never used for roofs. Canopy height is the vegetation height on tree pixels.
4. **Materials.** Each class gets an albedo, a thermal emissivity and a relative evaporative fraction (`configs/encanto.yaml`), used by the energy balance in milestone 3.
5. **Geohash summary.** Every pixel is tagged with its precision-8 geohash (about 38 x 19 m); the table gives class fractions, mean albedo and mean building and canopy height per cell.

## Setup

```powershell
git clone https://github.com/SaraEwaida/urban-cooling-simulator.git
cd urban-cooling-simulator
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest -q
```

## Run

```powershell
ucs fetch    # Overture vectors, NAIP and lidar onto the grid (a few minutes, needs internet)
ucs scene    # classes, heights, materials -> data/encanto/scene.zarr, geohash table, summary, figure
```

Outputs:

- `data/encanto/scene.zarr`: all layers on the grid (`surface_class`, `building_height`, `canopy_height`, `albedo`, `emissivity`, `evap`, `ndvi`, `naip`, `terrain`, ...), with CRS, transform, class codes and source metadata in the attributes.
- `outputs/geohash_cells.csv`: per-geohash class fractions, albedo and heights.
- `outputs/scene_summary.json`: class fractions, building count, heights, source items and dates.
- `outputs/figures/encanto_scene.png`: imagery, classes, heights and tree cover per geohash.

## Tests

Tests run offline on a synthetic 200 m scene (grass with a tree clump, bare ground, a building, a road and a pond) and check the geohash encoder against published reference values, grid snapping, every class rule, the roof height fallbacks, material lookup and the written Zarr, table and figure.

## Layout

```
configs/encanto.yaml      scene, thresholds, road widths, materials, paths
src/ucs/grid.py           projected grid, lon/lat of pixel centres
src/ucs/geohash.py        vectorised geohash encoder
src/ucs/overture.py       buildings and roads from Overture Maps
src/ucs/imagery.py        NAIP and 3DEP lidar from Planetary Computer, warped onto the grid
src/ucs/rasterize.py      footprints and buffered roads onto the grid
src/ucs/classify.py       surface classes, heights, material maps
src/ucs/scene.py          fetch and build steps, Zarr, geohash table, summary
src/ucs/plots.py          scene figure
src/ucs/cli.py            `ucs fetch | scene | all`
```
