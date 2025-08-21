import csv
import joblib
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from utils import load_config, setup_logger, ensure_parent_dir, timestamp
from sklearn.preprocessing import LabelEncoder


def _init_metrics_csv(metrics_path: Path):
    header = [
        "timestamp",
        "model_path",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "tn","fp","fn","tp"
    ]
    if not metrics_path.exists():
        ensure_parent_dir(metrics_path)
        with open(metrics_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

def main():
    cfg = load_config()
    log = setup_logger("evaluate")

    model_path = Path(cfg["paths"]["model"])
    Xtest_path = Path(cfg["paths"]["X_test"])
    ytest_path = Path(cfg["paths"]["y_test"])
    metrics_path = Path(cfg["paths"]["metrics"])

    if not all(p.exists() for p in [model_path, Xtest_path, ytest_path]):
        missing = [str(p) for p in [model_path, Xtest_path, ytest_path] if not p.exists()]
        raise FileNotFoundError(f"Missing artifacts for evaluation: {missing}")

    model = joblib.load(model_path)
    X_test = np.load(Xtest_path, allow_pickle = True)
    y_test = np.load(ytest_path, allow_pickle = True)

    y_pred = model.predict(X_test)


    le = LabelEncoder()
    y_test_enc = le.fit_transform(y_test)
    y_pred_enc = le.transform(y_pred)


    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test_enc, y_pred_enc, zero_division=0)
    rec = recall_score(y_test_enc, y_pred_enc, zero_division=0)
    f1 = f1_score(y_test_enc, y_pred_enc, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel() if len(np.unique(y_test)) == 2 else (None,)*4

    _init_metrics_csv(metrics_path)
    with open(metrics_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp(),
            str(model_path),
            round(acc, 6),
            round(prec, 6),
            round(rec, 6),
            round(f1, 6),
            tn, fp, fn, tp
        ])

    log.info(f"Evaluation complete | acc={acc:.4f} f1={f1:.4f} (metrics -> {metrics_path})")

if __name__ == "__main__":
    main()
