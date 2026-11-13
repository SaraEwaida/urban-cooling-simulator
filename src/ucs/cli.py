import argparse
import json

from . import heat, plots, scene, sun
from .config import load_config, out_path


def main(argv=None):
    p = argparse.ArgumentParser(prog="ucs", description="Urban cooling simulator")
    p.add_argument("command", choices=["fetch", "scene", "sun", "heat", "all"],
                   help="fetch: download inputs; scene: classify, write Zarr, table and figure; "
                        "sun: hourly shade and sky view factor; heat: surface energy balance; all: every step")
    p.add_argument("--config", default="configs/encanto.yaml")
    a = p.parse_args(argv)
    cfg = load_config(a.config)

    if a.command in ("fetch", "all"):
        scene.fetch(cfg)
    if a.command in ("scene", "all"):
        ds = scene.build(cfg)
        fig = plots.scene_figure(ds, cfg, out_path(cfg, "scene_figure"))
        with open(cfg["paths"]["scene_summary"], encoding="utf-8") as f:
            s = json.load(f)
        print(json.dumps({k: s[k] for k in ("valid_fraction", "lidar_fraction", "canopy_source", "class_fractions",
                                             "buildings", "building_height_m", "geohash_cells")}, indent=2))
        print(f"Scene: {cfg['paths']['scene']}\nFigure: {fig}")
    if a.command in ("sun", "all"):
        print(f"Sun and shade on {cfg['sun']['date']}:")
        ds = sun.run(cfg)
        fig = plots.sun_figure(ds, cfg, out_path(cfg, "sun_figure"))
        with open(cfg["paths"]["sun_summary"], encoding="utf-8") as f:
            s = json.load(f)
        print(json.dumps({"by_class": s["by_class"], "scene_mean": s["scene_mean"]}, indent=2))
        print(f"Sun: {cfg['paths']['sun']}\nFigure: {fig}")
    if a.command in ("heat", "all"):
        print(f"Surface energy balance for {cfg['sun']['date']} (after {cfg['heat']['spinup_days']} spin-up days):")
        ds, by_class, s = heat.run(cfg)
        fig = plots.heat_figure(ds, by_class, cfg, out_path(cfg, "heat_figure"))
        print(json.dumps({k: s[k] for k in ("air_temperature", "by_class", "scene_mean")}, indent=2))
        print(f"Heat: {cfg['paths']['heat']}\nFigure: {fig}")


if __name__ == "__main__":
    main()
