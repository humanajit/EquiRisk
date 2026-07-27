"""
src/etl/clean_transform.py

Reads raw prices + raw news from S3, cleans/standardizes both, joins them
on ticker + date, and writes the combined base table to
processed/features/ (partitioned by ticker) as the input for
feature_engineering.py.

This module owns "get raw data into one clean, joined Spark DataFrame."
Rolling-window stats, technical indicators, and sentiment scoring live in
feature_engineering.py / sentiment.py -- kept separate so each stage is
independently testable/inspectable in the notebooks.
"""

import logging

import yaml
from pyspark.sql import DataFrame, functions as F

from src.etl.spark_session import get_spark_session, s3a_path

logger = logging.getLogger("equirisk.etl.clean_transform")
logging.basicConfig(level=logging.INFO)


def _load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def read_raw_prices(spark, bucket: str, raw_prefix: str) -> DataFrame:
    """Reads all partitions under raw/prices/ -- each file already has a
    'ticker' column written during ingestion, so no path-parsing needed."""
    path = s3a_path(bucket, raw_prefix) + "/*/*.parquet"
    df = spark.read.parquet(path)
    logger.info(f"Read raw prices: {df.count()} rows")
    return df


def read_raw_news(spark, bucket: str, raw_prefix: str) -> DataFrame:
    """Raw news lands as one JSON blob per ticker/day (marketaux's
    response shape, with an outer 'data' array of articles). Explode
    that array into one row per article before anything downstream can
    use it."""
    path = s3a_path(bucket, raw_prefix) + "/*/*.json"
    raw = spark.read.option("multiLine", "true").json(path)

    exploded = raw.select(F.explode("data").alias("article"))
    news = exploded.select(
        F.col("article.uuid").alias("article_id"),
        F.col("article.title").alias("title"),
        F.col("article.description").alias("description"),
        F.col("article.published_at").alias("published_at"),
        F.explode("article.entities").alias("entity"),
    ).select(
        "article_id", "title", "description", "published_at",
        F.col("entity.symbol").alias("ticker"),
    )
    logger.info(f"Read raw news: {news.count()} article rows")
    return news


def clean_prices(df: DataFrame) -> DataFrame:
    """Standardize column names/types, drop duplicate ticker+date rows,
    drop rows with null close price (can't compute returns without it)."""
    df = (
        df.withColumnRenamed("Date", "date")
        .withColumnRenamed("Open", "open")
        .withColumnRenamed("High", "high")
        .withColumnRenamed("Low", "low")
        .withColumnRenamed("Close", "close")
        .withColumnRenamed("Volume", "volume")
        .withColumn("ticker", F.regexp_replace("ticker", "\\.NS$", ""))
        .withColumn("date", F.to_date("date"))
        .dropDuplicates(["ticker", "date"])
        .filter(F.col("close").isNotNull())
    )
    return df.select("ticker", "date", "open", "high", "low", "close", "volume")


def clean_news(df: DataFrame) -> DataFrame:
    """Standardize the news date to a plain date (for joining against
    daily price rows) and drop rows missing the fields RAG/sentiment
    need downstream."""
    df = (
        df.withColumn("news_date", F.to_date("published_at"))
        .filter(F.col("title").isNotNull())
        .dropDuplicates(["article_id"])
    )
    return df.select("ticker", "news_date", "article_id", "title", "description", "published_at")


def aggregate_news_daily(news_df: DataFrame) -> DataFrame:
    """Collapse multiple articles/day into one row per ticker+date, since
    price rows are daily. Keeps a concatenated headline list (used later
    by sentiment.py) and an article count (itself a mild signal --
    news volume spikes often coincide with volatility)."""
    return news_df.groupBy("ticker", "news_date").agg(
        F.collect_list("title").alias("headlines"),
        F.collect_list("description").alias("descriptions"),
        F.count("article_id").alias("article_count"),
    )


def join_prices_news(prices_df: DataFrame, news_daily_df: DataFrame) -> DataFrame:
    """Left join -- keep every price row even on days with no news
    (most days, for most midcap tickers). Downstream sentiment scoring
    should treat null headlines as neutral/no-signal, not missing data."""
    joined = prices_df.join(
        news_daily_df,
        (prices_df.ticker == news_daily_df.ticker) & (prices_df.date == news_daily_df.news_date),
        how="left",
    ).drop(news_daily_df.ticker).drop(news_daily_df.news_date)

    joined = joined.fillna({"article_count": 0})
    return joined


def run_etl(config_path: str = "config/config.yaml") -> None:
    """Main ETL entrypoint, called by the orchestrator."""
    config = _load_config(config_path)
    bucket = config["s3"]["bucket"]
    raw_prices_prefix = config["s3"]["paths"]["raw_prices"]
    raw_news_prefix = config["s3"]["paths"]["raw_news"]
    processed_prefix = config["s3"]["paths"]["processed_features"]

    spark = get_spark_session(
        app_name=config["spark"]["app_name"],
        master=config["spark"]["master"],
    )

    try:
        prices = clean_prices(read_raw_prices(spark, bucket, raw_prices_prefix))
        news = clean_news(read_raw_news(spark, bucket, raw_news_prefix))
        news_daily = aggregate_news_daily(news)
        base_table = join_prices_news(prices, news_daily)

        out_path = s3a_path(bucket, processed_prefix)
        (
            base_table.write.mode("overwrite")
            .partitionBy("ticker")
            .parquet(out_path)
        )
        logger.info(f"Wrote joined base table -> {out_path} (partitioned by ticker)")
    finally:
        spark.stop()


if __name__ == "__main__":
    run_etl()