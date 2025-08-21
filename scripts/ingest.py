from pathlib import Path
from utils import load_config, setup_logger, ensure_parent_dir

def main():
    cfg = load_config()
    log = setup_logger("ingest")
    raw_path = Path(cfg["paths"]["raw_data"])
    if not raw_path.exists():
        ensure_parent_dir(raw_path)
        raise FileNotFoundError(f"Raw dataset not found at {raw_path}")
    log.info(f"Raw dataset available: {raw_path.resolve()}")

if __name__ == "__main__":
    main()
