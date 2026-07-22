import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat_ws, coalesce, lit, udf
from pyspark.sql.types import DoubleType, StructField, StructType, StringType
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize VADER global instance for UDF serialization
analyzer = SentimentIntensityAnalyzer()


def compute_sentiment_risk(text):
    """PySpark UDF function to calculate sentiment and map it to risk impact."""
    if not text or not isinstance(text, str):
        return (0.0, "NEUTRAL", 10.0)

    scores = analyzer.polarity_scores(text)
    compound = float(scores["compound"])

    if compound <= -0.05:
        label = "HIGH_RISK_SENTIMENT"
        risk_impact = float(abs(compound) * 100)
    elif compound >= 0.05:
        label = "LOW_RISK_SENTIMENT"
        risk_impact = 0.0
    else:
        label = "NEUTRAL"
        risk_impact = 10.0

    return (compound, label, risk_impact)


# Define PySpark Return Schema for the UDF
sentiment_schema = StructType([
    StructField("compound_score", DoubleType(), True),
    StructField("risk_sentiment_label", StringType(), True),
    StructField("sentiment_risk_impact", DoubleType(), True)
])


def get_spark_session(app_name="EquiRisk-ETL"):
    """Creates and configures a PySpark Session with S3A Hadoop connector support."""
    access_key = os.getenv("AWS_ACCESS_KEY_ID", "")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "")

    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4")
        .config("spark.hadoop.fs.s3a.access.key", access_key)
        .config("spark.hadoop.fs.s3a.secret.key", secret_key)
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .getOrCreate()
    )


def run_spark_sentiment_etl(ticker_symbol: str, date_str: str):
    """Distributed PySpark ETL pipeline for news risk sentiment analysis."""
    spark = get_spark_session("News-Sentiment-ETL")
    clean_ticker = ticker_symbol.replace(".NS", "")
    bucket_name = os.getenv("S3_BUCKET_NAME", "equirisk-data-lake")

    raw_s3_path = f"s3a://{bucket_name}/raw/gnews_articles/dt={date_str}/ticker={clean_ticker}/news.parquet"
    processed_s3_path = f"s3a://{bucket_name}/processed/sentiment/dt={date_str}/ticker={clean_ticker}/"

    try:
        print(f"Reading raw news from {raw_s3_path}...")
        df = spark.read.parquet(raw_s3_path)

        # Concatenate title and description
        df_combined = df.withColumn(
            "full_text",
            concat_ws(" ", coalesce(col("title"), lit("")), coalesce(col("description"), lit("")))
        )

        # Register PySpark UDF
        risk_sentiment_udf = udf(compute_sentiment_risk, sentiment_schema)

        # Extract risk sentiment struct into explicit columns
        processed_df = df_combined.withColumn("sentiment_metrics", risk_sentiment_udf(col("full_text"))) \
            .withColumn("compound_score", col("sentiment_metrics.compound_score")) \
            .withColumn("risk_sentiment_label", col("sentiment_metrics.risk_sentiment_label")) \
            .withColumn("sentiment_risk_impact", col("sentiment_metrics.sentiment_risk_impact")) \
            .drop("sentiment_metrics", "full_text")

        # Write partitioned Parquet output back to S3
        processed_df.write.mode("overwrite").parquet(processed_s3_path)
        print(f"Successfully processed PySpark sentiment ETL for {ticker_symbol}")

    except Exception as e:
        print(f"Error during PySpark Sentiment ETL: {str(e)}")
    finally:
        spark.stop()


if __name__ == "__main__":
    run_spark_sentiment_etl("RELIANCE.NS", "2026-03-30")