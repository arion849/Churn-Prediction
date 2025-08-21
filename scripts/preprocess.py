import joblib
import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from utils import load_config, setup_logger, ensure_parent_dir

def main():
    cfg = load_config()
    log = setup_logger("preprocess")

    raw_path = Path(cfg["paths"]["raw_data"])
    out_path = Path(cfg["paths"]["processed_data"])
    preproc_path = Path(cfg["paths"]["preprocessor"])
    target = cfg["ml"]["target"]

    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw file: {raw_path}")

    df = pd.read_csv(raw_path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not in dataset. Columns: {list(df.columns)}")

    X = df.drop(columns=[target])
    y = df[target]

    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    X_num = pd.DataFrame(num_imputer.fit_transform(X[num_cols]), columns=num_cols, index=X.index) if num_cols else pd.DataFrame(index=X.index)
    X_cat_imp = pd.DataFrame(cat_imputer.fit_transform(X[cat_cols]), columns=cat_cols, index=X.index) if cat_cols else pd.DataFrame(index=X.index)

    if cat_cols:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_cat = pd.DataFrame(encoder.fit_transform(X_cat_imp), columns=cat_cols, index=X.index)
    else:
        encoder = None
        X_cat = pd.DataFrame(index=X.index)

    X_proc = pd.concat([X_num, X_cat], axis=1)
    df_out = X_proc.copy()
    df_out[target] = y

    ensure_parent_dir(out_path)
    df_out.to_csv(out_path, index=False)

    preprocessor = {
        "num_imputer": num_imputer,
        "cat_imputer": cat_imputer,
        "ordinal_encoder": encoder,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "target": target,
    }
    ensure_parent_dir(preproc_path)
    joblib.dump(preprocessor, preproc_path)

    log.info(f"Preprocessed data -> {out_path}")
    log.info(f"Preprocessor saved -> {preproc_path}")

if __name__ == "__main__":
    main()
