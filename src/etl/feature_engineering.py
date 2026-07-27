"""
src/etl/feature_engineering.py

Reads the joined base table (processed/features/ written by
clean_transform.py), computes returns, rolling volatility, moving
averages, RSI, and MACD, and writes the final feature table back to S3.
This is the direct input to ML training/inference.

Kept separate from clean_transform.py so you can re-run just the feature
math (e.g. to try a different rolling window) without re-doing the raw
price/news join every time.
"""

import logging

import yaml
from pyspark.sql import DataFrame, Window, functions as F

from src.etl.spark_session import get_spark_session, s3a_path

logger = logging.getLogger("equirisk.etl.feature_engineering")
logging.basicConfig(level=logging.INFO)


def _load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def add_returns(df: DataFrame) -> DataFrame:
    """Daily simple return = (close_t - close_t-1) / close_t-1, computed
    per ticker via a lag window ordered by date."""
    w = Window.partitionBy("ticker").orderBy("date")
    return df.withColumn("prev_close", F.lag("close", 1).over(w)).withColumn(
        "daily_return", (F.col("close") - F.col("prev_close")) / F.col("prev_close")
    )


def add_rolling_volatility(df: DataFrame, windows: list) -> DataFrame:
    """Rolling stddev of daily returns over each window (in trading
    days) -- the core "risk" signal. rowsBetween uses trading-day counts,
    not calendar days, since that's what the raw data actually has."""
    for w_days in windows:
        w = Window.partitionBy("ticker").orderBy("date").rowsBetween(-(w_days - 1), 0)
        df = df.withColumn(f"volatility_{w_days}d", F.stddev("daily_return").over(w))
    return df


def add_moving_averages(df: DataFrame, windows: list) -> DataFrame:
    for w_days in windows:
        w = Window.partitionBy("ticker").orderBy("date").rowsBetween(-(w_days - 1), 0)
        df = df.withColumn(f"ma_{w_days}d", F.avg("close").over(w))
    return df


def add_rsi(df: DataFrame, period: int = 14) -> DataFrame:
    """Standard 14-day RSI. Computed from average gain/loss over the
    window using the daily_return column already added by add_returns."""
    w_order = Window.partitionBy("ticker").orderBy("date")
    df = df.withColumn("gain", F.when(F.col("daily_return") > 0, F.col("daily_return")).otherwise(0.0))
    df = df.withColumn("loss", F.when(F.col("daily_return") < 0, -F.col("daily_return")).otherwise(0.0))

    w_roll = Window.partitionBy("ticker").orderBy("date").rowsBetween(-(period - 1), 0)
    df = df.withColumn("avg_gain", F.avg("gain").over(w_roll))
    df = df.withColumn("avg_loss", F.avg("loss").over(w_roll))

    df = df.withColumn(
        "rsi_14",
        F.when(F.col("avg_loss") == 0, 100.0).otherwise(
            100.0 - (100.0 / (1.0 + (F.col("avg_gain") / F.col("avg_loss"))))
        ),
    )
    return df.drop("gain", "loss", "avg_gain", "avg_loss")


def add_macd(df: DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> DataFrame:
    """MACD via EMA approximated over a rolling window (Spark has no
    native EMA/recursive window function, so this uses a simple-average
    approximation over the fast/slow windows rather than a true
    exponential decay). Good enough as a technical-indicator feature for
    this project; note the approximation in your report if asked."""
    w_fast = Window.partitionBy("ticker").orderBy("date").rowsBetween(-(fast - 1), 0)
    w_slow = Window.partitionBy("ticker").orderBy("date").rowsBetween(-(slow - 1), 0)
    w_signal = Window.partitionBy("ticker").orderBy("date").rowsBetween(-(signal - 1), 0)

    df = df.withColumn("ema_fast_approx", F.avg("close").over(w_fast))
    df = df.withColumn("ema_slow_approx", F.avg("close").over(w_slow))
    df = df.withColumn("macd_line", F.col("ema_fast_approx") - F.col("ema_slow_approx"))
    df = df.withColumn("macd_signal", F.avg("macd_line").over(w_signal))
    return df.drop("ema_fast_approx", "ema_slow_approx")


def add_forward_volatility_label(df: DataFrame, horizon_days: int, buckets: list) -> DataFrame:
    """Builds the ML risk label: forward-looking volatility over the
    next `horizon_days`, bucketed into risk categories. This looks
    FORWARD (uses F.lead over future rows), so the last `horizon_days`
    rows per ticker will have null labels -- filter those out before
    training, they're fine to keep for live inference (that's the row
    you're actually trying to predict).
    """
    w_forward = Window.partitionBy("ticker").orderBy("date").rowsBetween(1, horizon_days)
    df = df.withColumn("forward_volatility", F.stddev("daily_return").over(w_forward))

    quantiles = df.approxQuantile("forward_volatility", [1 / len(buckets) * i for i in range(1, len(buckets))], 0.01)

    bucket_expr = F.when(F.col("forward_volatility").isNull(), None)
    for i, q in enumerate(quantiles):
        bucket_expr = bucket_expr.when(F.col("forward_volatility") <= q, buckets[i])
    bucket_expr = bucket_expr.otherwise(buckets[-1])

    return df.withColumn("risk_label", bucket_expr)


def run_feature_engineering(config_path: str = "config/config.yaml") -> None:
    """Main feature engineering entrypoint. Reads the joined base table,
    adds all feature columns, writes the final feature table back to S3
    (overwriting processed/features/ in place)."""
    config = _load_config(config_path)
    bucket = config["s3"]["bucket"]
    processed_prefix = config["s3"]["paths"]["processed_features"]
    fe_config = config["feature_engineering"]
    ml_config = config["ml"]["risk_label"]

    spark = get_spark_session(
        app_name=config["spark"]["app_name"] + "-FeatureEng",
        master=config["spark"]["master"],
    )

    try:
        in_path = s3a_path(bucket, processed_prefix)
        df = spark.read.parquet(in_path)

        df = add_returns(df)
        df = add_rolling_volatility(df, fe_config["rolling_windows_days"])
        df = add_moving_averages(df, fe_config["rolling_windows_days"])
        df = add_rsi(df)
        df = add_macd(df)
        df = add_forward_volatility_label(df, ml_config["horizon_days"], ml_config["buckets"])

        (
            df.write.mode("overwrite")
            .partitionBy("ticker")
            .parquet(in_path)
        )
        logger.info(f"Wrote feature table -> {in_path} (partitioned by ticker)")
    finally:
        spark.stop()


if __name__ == "__main__":
    run_feature_engineering()