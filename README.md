# Churn Prediction Pipeline

An Apache Airflow orchestrated ETL and machine learning pipeline that predicts customer churn from bank account data. A single DAG runs ingestion, preprocessing, model training and evaluation, with all paths and hyperparameters driven by `config.yaml`. The whole stack runs locally under Docker Compose with a PostgreSQL metadata database.

## Pipeline overview

```mermaid
flowchart LR
    A[ingest] --> B[preprocess] --> C[train] --> D[evaluate]
```

| Task | Script | What it does |
| --- | --- | --- |
| `ingest` | `scripts/ingest.py` | Checks the raw CSV exists at the configured path, creating the parent directory and failing with a clear error if it does not. |
| `preprocess` | `scripts/preprocess.py` | Loads the CSV, validates the target column, imputes and encodes the features, writes the processed CSV and saves the fitted preprocessor. |
| `train` | `scripts/train.py` | Stratified train/test split, fits a logistic regression, saves the model plus a timestamped copy, and writes the test split to `.npy` files. |
| `evaluate` | `scripts/evaluate.py` | Loads the saved model and test split, scores it, and appends one row of metrics to `models/metrics.csv`. |

Tasks run strictly in sequence (`ingest >> preprocess >> train >> evaluate`), and each one fails loudly on a missing input rather than producing a partial artefact.

### Data quality steps in `preprocess`

- **Validation**: raises if the raw file is missing, or if the configured target column is absent from the CSV (the error lists the columns actually found).
- **Column typing**: numeric and categorical columns are detected automatically from dtypes, so no column list is hard coded.
- **Imputation**: median for numeric columns, most frequent for categorical columns.
- **Encoding**: ordinal encoding of categorical columns, with unseen categories at inference mapped to `-1` instead of raising.
- **Saved preprocessor**: the fitted imputers, the encoder, the column lists and the target name are pickled together to `models/ordinal_encoder.pkl`, so the same transformations can be reapplied to new data.

## Design choices

- **YAML driven config**: every path, hyperparameter and the target column live in `config.yaml`. No script hard codes a filename, and the config location itself is overridable via the `CONFIG_PATH` environment variable.
- **Stratified split**: the train/test split preserves the class balance, which matters on an imbalanced target like churn. It falls back to an unstratified split if only one class is present.
- **Timestamped model versioning**: alongside `logistic_model.pkl`, each run writes `logistic_model_<YYYYmmdd_HHMMSS>.pkl`, so earlier models are not overwritten. Controlled by the `versioning` block in the config.
- **Per run metrics log**: `evaluate` appends to `models/metrics.csv` rather than overwriting it, giving a run history of timestamp, model path, accuracy, precision, recall, F1 and the four confusion matrix cells (`tn`, `fp`, `fn`, `tp`). The header is written on first run.
- **Dockerised Airflow**: a custom image on top of `apache/airflow:2.10.2-python3.10`, the `LocalExecutor`, and PostgreSQL 13 for the Airflow metadata database on a named volume. Code, config, data and models are bind mounted, so edits appear in the container without a rebuild.

## Tech stack

Python 3.10, Apache Airflow 2.10.2, scikit-learn 1.5.1, pandas 2.2.2, NumPy 1.26.4, XGBoost 2.1.1, PostgreSQL 13, Docker and Docker Compose. Matplotlib, seaborn and JupyterLab are pinned for exploratory work.

## Project structure

```
.
├── dags/
│   └── churn_dag.py       # DAG definition: four PythonOperators in sequence
├── scripts/
│   ├── ingest.py          # raw file presence check
│   ├── preprocess.py      # validation, imputation, encoding, saved preprocessor
│   ├── train.py           # stratified split, logistic regression, versioned model
│   ├── evaluate.py        # metrics appended to models/metrics.csv
│   └── utils.py           # config loading, logging, timestamps, versioned copies
├── config.yaml            # paths, ML hyperparameters, versioning, log level
├── Dockerfile             # Airflow image plus requirements.txt
├── docker-compose.yaml    # postgres, airflow-webserver, airflow-scheduler
└── requirements.txt
```

`data/` and `models/` are created at runtime and are gitignored, so no data or artefacts are committed.

## Getting started

Requires Docker and Docker Compose.

1. **Set the Airflow user id** so files written in the container are owned by you rather than root. Without this, Compose defaults to uid `50000`.

   ```bash
   echo "AIRFLOW_UID=$(id -u)" > .env
   ```

2. **Add the dataset** at `data/raw/churn_data.csv`. It must contain a `Churn` target column; see [Dataset](#dataset).

   ```bash
   mkdir -p data/raw && cp /path/to/churn_data.csv data/raw/
   ```

3. **Build the image and start PostgreSQL.**

   ```bash
   docker compose build
   docker compose up -d postgres
   ```

4. **Initialise the metadata database and create a UI user.** The Compose file has no init service, so this is a one off manual step.

   ```bash
   docker compose run --rm airflow-webserver airflow db migrate
   docker compose run --rm airflow-webserver airflow users create \
     --username admin --password admin --role Admin \
     --firstname Admin --lastname User --email admin@example.com
   ```

5. **Start the webserver and scheduler.**

   ```bash
   docker compose up -d
   ```

6. **Open the Airflow UI** at http://localhost:8080 and log in with the credentials above. The DAG `churn_prediction_pipeline` has `schedule_interval=None` and is paused on creation, so it never runs on its own: unpause it and trigger a run manually with the play button. The equivalent from the command line is:

   ```bash
   docker compose run --rm airflow-scheduler airflow dags trigger churn_prediction_pipeline
   ```

**Where outputs land** (on the host, via bind mounts):

- `data/processed/churn_preprocessed.csv`, `X_test.npy`, `y_test.npy`
- `models/logistic_model.pkl` plus a timestamped copy per run
- `models/ordinal_encoder.pkl`, the fitted preprocessor
- `models/metrics.csv`, one row appended per evaluation
- `logs/`, Airflow task logs

## Dataset

The CSV is not included in this repository, and `data/` is gitignored. The pipeline expects a file at `data/raw/churn_data.csv` with:

- a target column named `Churn` (configurable via `ml.target` in `config.yaml`)
- any mix of numeric and categorical feature columns, which are detected from their dtypes rather than listed anywhere

Any bank churn dataset in that shape will run. The original source of the file used during development is not recorded, so it is deliberately not cited here rather than guessed at.

## Limitations and next steps

Stated plainly, since this is a pipeline engineering project rather than a modelling result:

- The model is a **logistic regression baseline** with default regularisation. No hyperparameter search, no cross validation, and no benchmark against other models.
- **`ingest` only checks that the file exists.** There is no download step, no schema check and no row count or freshness assertion at that stage.
- The **EDA notebook has not been written**, so the feature choices rest on automatic dtype detection rather than on analysis.
- `evaluate` reports **binary classification metrics only**; the confusion matrix columns are left empty for a non binary target.

Next steps, in the order worth doing them:

1. Try **XGBoost**, already pinned in `requirements.txt`, and compare it against the baseline using the existing `metrics.csv` history.
2. Add **tests** for the preprocessing and training steps, with a small synthetic frame as a fixture.
3. Add **data validation rules** to `ingest`, covering expected columns, dtypes, null thresholds and row counts, so a bad input fails before it reaches the model.
