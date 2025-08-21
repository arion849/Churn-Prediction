import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from utils import load_config, setup_logger, ensure_parent_dir, make_versioned_copy

def main():
    cfg = load_config()
    log = setup_logger("train")

    proc_path = Path(cfg["paths"]["processed_data"])
    model_path = Path(cfg["paths"]["model"])
    Xtest_path = Path(cfg["paths"]["X_test"])
    ytest_path = Path(cfg["paths"]["y_test"])
    target = cfg["ml"]["target"]

    if not proc_path.exists():
        raise FileNotFoundError(f"Missing processed file: {proc_path}")

    df = pd.read_csv(proc_path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' missing from processed data.")

    X = df.drop(columns=[target]).values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["ml"]["test_size"],
        random_state=cfg["ml"]["random_state"],
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    model = LogisticRegression(
        max_iter=cfg["ml"]["max_iter"],
        solver=cfg["ml"].get("solver", "lbfgs"),
    )
    model.fit(X_train, y_train)

    ensure_parent_dir(model_path)
    joblib.dump(model, model_path)
    vpath = make_versioned_copy(model_path)

    ensure_parent_dir(Xtest_path)
    ensure_parent_dir(ytest_path)
    np.save(Xtest_path, X_test)
    np.save(ytest_path, y_test)

    log.info(
    f"Model saved -> {model_path} "
    f"(versioned copy: {Path(vpath).name if vpath else 'n/a'})"
    )
    log.info(f"Test split saved -> {Xtest_path}, {ytest_path}")

if __name__ == "__main__":
    main()
