import os
import yaml
import logging
from pathlib import Path
from datetime import datetime
import shutil
import json

DEFAULT_CONFIG_PATH = os.environ.get("CONFIG_PATH", "/opt/airflow/config.yaml")

def load_config(path: str = DEFAULT_CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def setup_logger(name: str = "app", level: str | None = None) -> logging.Logger:
    cfg = load_config()
    lvl = level or cfg.get("logging", {}).get("level", "INFO")
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(getattr(logging, lvl))
        ch = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def timestamp(fmt: str | None = None) -> str:
    cfg = load_config()
    fmt = fmt or cfg.get("versioning", {}).get("format", "%Y%m%d_%H%M%S")
    return datetime.now().strftime(fmt)


def make_versioned_copy(p: str) -> str:

    cfg = load_config()
    
    if not cfg['versioning']['enable']:
        return p

    timestamp = datetime.now().strftime(cfg['versioning']['format'])
    orig = Path(p)
    versioned = orig.with_name(f"{orig.stem}_{timestamp}{orig.suffix}")

    with open(orig, "rb") as f_src, open(versioned, "wb") as f_dst:
        f_dst.write(f_src.read())

    return str(versioned)


def write_json(obj: dict, path: str | Path) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
