import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from data_pipeline.s3_uploader import upload_dataframe_to_s3
import boto3
import io
import os

class NewsRiskSentimentAnalyzer:
    def __init__(self):
        # Initialize VADER sentiment analyzer
        self.analyzer = SentimentIntensityAnalyzer()

    def calculate_risk_sentiment(self, text: str) -> dict:
        """
        Analyzes headline/description text and returns sentiment scores
        and a risk contribution factor.
        """
        if not text or not isinstance(text, str):
            return {"compound": 0.0, "risk_sentiment_label": "NEUTRAL", "risk_impact": 0.0}
        
        scores = self.analyzer.polarity_scores(text)
        compound = scores["compound"]

        # Map sentiment compound score to Risk Perception:
        # Negative sentiment increases perceived risk score.
        if compound <= -0.05:
            label = "HIGH_RISK_SENTIMENT"
            risk_impact = abs(compound) * 100  # Converts negative sentiment to risk scale
        elif compound >= 0.05:
            label = "LOW_RISK_SENTIMENT"
            risk_impact = 0.0
        else:
            label = "NEUTRAL"
            risk_impact = 10.0

        return {
            "compound_score": compound,
            "risk_sentiment_label": label,
            "sentiment_risk_impact": risk_impact
        }

    def process_news_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Processes news DataFrame and appends risk sentiment metrics."""
        if df.empty:
            return df

        # Combine title and description for holistic text evaluation
        df["full_text"] = df["title"].fillna("") + " " + df["description"].fillna("")
        
        results = df["full_text"].apply(self.calculate_risk_sentiment)
        results_df = pd.DataFrame(list(results))

        # Concatenate results back to main dataframe
        processed_df = pd.concat([df.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)
        return processed_df


def run_sentiment_etl_pipeline(ticker_symbol: str, date_str: str):
    """
    Reads raw news parquet from S3, calculates news risk metrics,
    and streams processed features back to S3.
    """
    clean_ticker = ticker_symbol.replace(".NS", "")
    raw_s3_key = f"raw/gnews_articles/dt={date_str}/ticker={clean_ticker}/news.parquet"
    processed_s3_key = f"processed/sentiment/dt={date_str}/ticker={clean_ticker}/news_risk_features.parquet"

    bucket_name = os.getenv("S3_BUCKET_NAME", "equirisk-data-lake")
    s3_client = boto3.client("s3")

    try:
        # Fetch raw parquet directly from S3 into memory
        response = s3_client.get_object(Bucket=bucket_name, Key=raw_s3_key)
        raw_df = pd.read_parquet(io.BytesIO(response["Body"].read()))

        # Run NLP risk transformation
        nlp_engine = NewsRiskSentimentAnalyzer()
        processed_df = nlp_engine.process_news_dataframe(raw_df)

        # Upload processed risk dataset back to S3
        upload_dataframe_to_s3(processed_df, processed_s3_key, file_format="parquet")
        print(f"Sentiment risk ETL complete for {ticker_symbol}")

    except Exception as e:
        print(f"Error executing sentiment ETL pipeline: {str(e)}")


if __name__ == "__main__":
    # Test execution for a given date
    today_str = pd.Timestamp.now().strftime("%Y-%m-%d")
    run_sentiment_etl_pipeline("RELIANCE.NS", today_str)