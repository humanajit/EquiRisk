import os
import math
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, stddev, min as spark_min, max as spark_max, 
    last, current_timestamp, when, lit
)
from pyspark.sql.window import Window


def get_spark_session(app_name="EquiRisk-Financials"):
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


def run_spark_financial_risk_etl(ticker_symbol: str):
    """Distributed PySpark ETL for quantitative price & leverage risk metrics."""
    spark = get_spark_session("Financial-Risk-ETL")
    clean_ticker = ticker_symbol.replace(".NS", "")
    bucket_name = os.getenv("S3_BUCKET_NAME", "equirisk-data-lake")

    raw_price_path = f"s3a://{bucket_name}/raw/yahoo/history/ticker={clean_ticker}/price_history.parquet"
    processed_risk_path = f"s3a://{bucket_name}/processed/financial_risk/ticker={clean_ticker}/"

    try:
        print(f"Reading raw price history from {raw_price_path}...")
        df = spark.read.parquet(raw_price_path)

        # 1. Compute daily returns using Window function
        window_spec = Window.orderBy("Date")
        df_returns = df.withColumn("prev_close", col("Close").over(window_spec)) \
                       .withColumn("daily_return", (col("Close") - col("prev_close")) / col("prev_close"))

        # 2. Compute Peak and Drawdown per record
        window_cumulative = Window.orderBy("Date").rowsBetween(Window.unboundedPreceding, Window.currentRow)
        df_drawdown = df_returns.withColumn("rolling_peak", spark_max("Close").over(window_cumulative)) \
                                .withColumn("drawdown", (col("Close") - col("rolling_peak")) / col("rolling_peak"))

        # 3. Aggregate Distributed Risk Indicators
        # Annualized Volatility (252 trading days factor ~ sqrt(252))
        sq_252 = math.sqrt(252)

        summary_df = df_drawdown.agg(
            (stddev("daily_return") * sq_252).alias("annualized_volatility"),
            spark_min("drawdown").alias("max_drawdown"),
            last("Close").alias("latest_close")
        ).withColumn("symbol", lit(clean_ticker)) \
         .withColumn("calculated_at", current_timestamp())

        # Write processed risk metrics to S3
        summary_df.write.mode("overwrite").parquet(processed_risk_path)
        print(f"Successfully calculated financial risk features with PySpark for {ticker_symbol}")

    except Exception as e:
        print(f"Error executing PySpark financial ETL: {str(e)}")
    finally:
        spark.stop()


if __name__ == "__main__":
    run_spark_financial_risk_etl("RELIANCE.NS")