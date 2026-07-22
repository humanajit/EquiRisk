import os
import requests
import pandas as pd
from dotenv import load_dotenv
from data_pipeline.s3_uploader import upload_dataframe_to_s3

# Load environment variables
load_dotenv()

GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")

class GNewsScraper:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or GNEWS_API_KEY
        self.base_url = "https://gnews.io/api/v4/search"

    def fetch_company_news(self, company_query: str, max_articles: int = 10) -> pd.DataFrame:
        """Fetches live news articles for a company/ticker using GNews API."""
        if not self.api_key:
            print("Error: GNEWS_API_KEY environment variable not set.")
            return pd.DataFrame()

        params = {
            "q": company_query,
            "lang": "en",
            "country": "in",
            "max": max_articles,
            "apikey": self.api_key
        }

        try:
            print(f"Fetching GNews articles for query: '{company_query}'...")
            response = requests.get(self.base_url, params=params, timeout=10)

            if response.status_code == 200:
                articles = response.json().get("articles", [])
                if not articles:
                    print(f"No news articles found for query '{company_query}'.")
                    return pd.DataFrame()

                df = pd.DataFrame(articles)
                # Keep relevant metadata fields
                df["search_query"] = company_query
                df["ingested_at"] = pd.Timestamp.now().isoformat()

                # Extract nested source name if present
                if "source" in df.columns:
                    df["source_name"] = df["source"].apply(lambda x: x.get("name") if isinstance(x, dict) else x)

                return df
            else:
                print(f"Failed to fetch GNews data. Status Code: {response.status_code}, Response: {response.text}")
                return pd.DataFrame()

        except Exception as e:
            print(f"Exception during GNews API request: {str(e)}")
            return pd.DataFrame()


def stream_gnews_to_s3(company_query: str, ticker_symbol: str):
    """Fetches news for a company and streams the resulting DataFrame to S3."""
    scraper = GNewsScraper()
    df_news = scraper.fetch_company_news(company_query)

    if not df_news.empty:
        today_str = pd.Timestamp.now().strftime("%Y-%m-%d")
        clean_ticker = ticker_symbol.replace(".NS", "")
        s3_key = f"raw/gnews_articles/dt={today_str}/ticker={clean_ticker}/news.parquet"

        upload_dataframe_to_s3(df_news, s3_key, file_format="parquet")
    else:
        print(f"Skipping S3 upload for {ticker_symbol} due to empty data.")


if __name__ == "__main__":
    # Test run for Reliance Industries
    stream_gnews_to_s3(company_query="Reliance Industries stock", ticker_symbol="RELIANCE.NS")