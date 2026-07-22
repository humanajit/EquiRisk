import os
import requests
import pandas as pd
from dotenv import load_dotenv
from data_pipeline.s3_uploader import upload_dataframe_to_s3

# Load environment variables
load_dotenv()

class NSEIndiaScraper:
    def __init__(self):
        self.base_url = "https://www.nseindia.com"
        self.nifty50_api = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"
        self.session = requests.Session()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/",
        }
        self.session.headers.update(self.headers)
        self._initialize_session()

    def _initialize_session(self):
        """Performs initial handshake to grab required Akamai session cookies."""
        try:
            print("Initializing NSE session & cookies...")
            response = self.session.get(self.base_url, timeout=10)
            if response.status_code == 200:
                print("NSE Session Handshake successful.")
            else:
                print(f"Warning: Handshake returned status code {response.status_code}")
        except Exception as e:
            print(f"Error initializing NSE session: {str(e)}")

    def fetch_nifty50_live(self) -> pd.DataFrame:
        """Fetches NIFTY 50 live stock indicators, prices, and depth directly from NSE."""
        try:
            print("Fetching live NIFTY 50 data from NSE...")
            response = self.session.get(self.nifty50_api, timeout=10)
            
            if response.status_code == 200:
                json_data = response.json()
                raw_stocks = json_data.get("data", [])
                
                df = pd.DataFrame(raw_stocks)
                
                # Keep core financial & market depth features
                selected_cols = [
                    "symbol", "open", "dayHigh", "dayLow", "lastPrice", 
                    "previousClose", "change", "pChange", "totalTradedVolume", 
                    "totalTradedValue", "yearHigh", "yearLow"
                ]
                
                # Filter columns present in response
                available_cols = [col for col in selected_cols if col in df.columns]
                cleaned_df = df[available_cols].copy()
                
                # Add ingestion timestamp
                cleaned_df["timestamp"] = pd.Timestamp.now().isoformat()
                return cleaned_df
            else:
                print(f"Failed to fetch NSE data. Status Code: {response.status_code}")
                return pd.DataFrame()

        except Exception as e:
            print(f"Exception during NSE scraping: {str(e)}")
            return pd.DataFrame()


def stream_nse_live_to_s3():
    """Instantiates scraper and uploads live data directly to S3."""
    scraper = NSEIndiaScraper()
    df_live = scraper.fetch_nifty50_live()

    if not df_live.empty:
        today_str = pd.Timestamp.now().strftime("%Y-%m-%d")
        s3_key = f"raw/nse_live/dt={today_str}/nifty50_live.parquet"
        
        # Directly upload into memory to S3
        upload_dataframe_to_s3(df_live, s3_key, file_format="parquet")
    else:
        print("No data extracted from NSE. S3 upload skipped.")


if __name__ == "__main__":
    stream_nse_live_to_s3()