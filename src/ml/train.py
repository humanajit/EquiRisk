"""
src/ml/train.py

Trains and compares the candidate models defined in config.yaml
(ml.candidate_models) on the feature table produced by the ETL stage,
picks the best one by macro F1, and saves it to S3
(models/risk_model_{version}/model.pkl).

The actual "which model is best and why" exploration/plotting belongs in
notebooks/04_model_training_comparison.ipynb -- that notebook should
import train_all_candidates() and compare_models() from here/evaluate.py
rather than reimplementing training, so the notebook's numbers and this
script's chosen model are guaranteed to match.
"""

import io
import logging

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import lightgbm as lgb

from src.ml.evaluate import compare_models
from src.utils.s3_io import read_hive_partitioned_parquet_s3, put_bytes, model_key

logger = logging.getLogger("equirisk.ml.train")
logging.basicConfig(level=logging.INFO)

FEATURE_COLUMNS = [
    "daily_return", "volatility_20d", "volatility_60d", "volatility_90d",
    "ma_20d", "ma_60d", "ma_90d", "rsi_14", "macd_line", "macd_signal",
    "daily_sentiment", "sentiment_3d_avg", "article_count",
]
LABEL_COLUMN = "risk_label"

MODEL_REGISTRY = {
    "logistic_regression": lambda: LogisticRegression(max_iter=1000, multi_class="multinomial"),
    "random_forest": lambda: RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42),
    "xgboost": lambda: xgb.XGBClassifier(n_estimators=300, max_depth=6, eval_metric="mlogloss", random_state=42),
    "lightgbm": lambda: lgb.LGBMClassifier(n_estimators=300, max_depth=6, random_state=42),
}


def _load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_feature_table(bucket: str, processed_prefix: str) -> pd.DataFrame:
    """Reads all per-ticker parquet partitions (reconstructing the
    'ticker' column Spark strips out into the folder name) and
    concatenates into one pandas DataFrame for scikit-learn/XGBoost/
    LightGBM training. This is the one place a full Spark->pandas
    handoff happens -- fine at this data scale (150 tickers x ~5 years
    of daily rows), but if the dataset grows much larger, train
    directly against Spark ML instead."""
    df = read_hive_partitioned_parquet_s3(processed_prefix, bucket, partition_col="ticker")
    logger.info(f"Loaded feature table: {df.shape[0]} rows")
    return df


def prepare_train_test(df: pd.DataFrame, test_size: float, random_state: int):
    """Drops rows with a null label (the most recent horizon_days rows
    per ticker -- see feature_engineering.add_forward_volatility_label)
    since those are for live inference, not training. Also drops any
    row with a null feature value rather than imputing -- for a course
    project, being explicit about "we only train on complete rows" is
    easier to defend than a silent imputation choice.

    Labels are encoded to integers via LabelEncoder before the split --
    XGBoost's sklearn wrapper requires integer class labels (0,1,2,...),
    while LogisticRegression/RandomForest/LightGBM accept strings
    directly. Encoding once here (rather than per-model) keeps every
    model trained on identical labels, and the encoder is returned so
    predictions can be decoded back to "Low"/"Medium"/"High" afterward."""
    df = df.dropna(subset=[LABEL_COLUMN] + FEATURE_COLUMNS)

    X = df[FEATURE_COLUMNS]
    y_raw = df[LABEL_COLUMN]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, encoder


def train_all_candidates(X_train, y_train, X_test, candidate_names: list, encoder: LabelEncoder) -> dict:
    """Trains every candidate model listed in config.yaml. Returns
    {"model_name": (fitted_model, y_pred_labels)} -- y_pred_labels is
    already decoded back to the original string labels via `encoder`,
    so callers (evaluate.compare_models, etc.) never need to know
    encoding happened."""
    results = {}
    for name in candidate_names:
        if name not in MODEL_REGISTRY:
            logger.warning(f"Unknown model '{name}' in candidate_models -- skipping")
            continue
        logger.info(f"Training {name}...")
        model = MODEL_REGISTRY[name]()
        model.fit(X_train, y_train)
        y_pred_encoded = model.predict(X_test)
        y_pred_labels = encoder.inverse_transform(y_pred_encoded)
        results[name] = (model, y_pred_labels)
    return results


def run_training(config_path: str = "config/config.yaml") -> str:
    """Main training entrypoint. Trains all candidates, picks the best
    by macro F1, saves it + the scaler to S3. Returns the winning model
    name (mainly useful for logging/notebook display)."""
    config = _load_config(config_path)
    bucket = config["s3"]["bucket"]
    processed_prefix = config["s3"]["paths"]["processed_features"]
    ml_config = config["ml"]
    labels = ml_config["risk_label"]["buckets"]

    df = load_feature_table(bucket, processed_prefix)
    X_train, X_test, y_train, y_test, scaler, encoder = prepare_train_test(
        df, ml_config["train_test_split"], ml_config["random_state"]
    )

    trained = train_all_candidates(X_train, y_train, X_test, ml_config["candidate_models"], encoder)
    if not trained:
        raise RuntimeError("No candidate models were successfully trained")

    y_test_labels = encoder.inverse_transform(y_test)
    comparison_input = {name: (y_test_labels, y_pred) for name, (model, y_pred) in trained.items()}
    comparison_table = compare_models(comparison_input, labels)
    logger.info(f"Model comparison:\n{comparison_table}")

    best_model_name = comparison_table.index[0]
    best_model, _ = trained[best_model_name]
    logger.info(f"Best model: {best_model_name} (f1_macro={comparison_table.loc[best_model_name, 'f1_macro']:.4f})")

    version = ml_config["model_version"]
    model_buf = io.BytesIO()
    joblib.dump(
        {"model": best_model, "scaler": scaler, "encoder": encoder, "features": FEATURE_COLUMNS, "labels": labels},
        model_buf,
    )
    put_bytes(model_key(version, "model.pkl"), model_buf.getvalue(), bucket)

    logger.info(f"Saved {best_model_name} -> s3://{bucket}/{model_key(version)}")
    return best_model_name


if __name__ == "__main__":
    run_training()