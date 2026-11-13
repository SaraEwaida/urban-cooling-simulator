from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def out_path(cfg: dict, key: str) -> Path:
    p = Path(cfg["paths"][key])
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
