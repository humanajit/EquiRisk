import pandas as pd
import yfinance as yf
from data_pipeline.s3_uploader import upload_dataframe_to_s3


def fetch_yahoo_financials(ticker_symbol: str) -> dict:
    """Fetches quarterly/yearly financials and historical price data from Yahoo Finance.

    Note: For Nifty 50 stocks on Yahoo Finance, append '.NS' (e.g.,
    'RELIANCE.NS').
    """
    print(f"Fetching Yahoo Finance data for {ticker_symbol}...")
    stock = yf.Ticker(ticker_symbol)

    # 1. Historical OHLCV Price Data (Last 5 Years)
    hist_df = stock.history(period="5y").reset_index()
    hist_df["symbol"] = ticker_symbol

    # 2. Quarterly Financials
    quarterly_fin = stock.quarterly_financials.T.reset_index()
    quarterly_fin["symbol"] = ticker_symbol

    # 3. Yearly Financials
    yearly_fin = stock.financials.T.reset_index()
    yearly_fin["symbol"] = ticker_symbol

    return {
        "history": hist_df,
        "quarterly": quarterly_fin,
        "yearly": yearly_fin,
    }


def process_and_stream_ticker(ticker_symbol: str):
    """Fetches data for a ticker and streams it directly to S3."""
    clean_symbol = ticker_symbol.replace(".NS", "")
    data = fetch_yahoo_financials(ticker_symbol)

    # Define S3 keys (Partitioned by ticker symbol)
    hist_key = f"raw/yahoo/history/ticker={clean_symbol}/price_history.parquet"
    quarterly_key = (
        f"raw/yahoo/quarterly/ticker={clean_symbol}/quarterly_fin.parquet"
    )
    yearly_key = f"raw/yahoo/yearly/ticker={clean_symbol}/yearly_fin.parquet"

    # Stream each dataframe directly to S3
    upload_dataframe_to_s3(data["history"], hist_key)
    upload_dataframe_to_s3(data["quarterly"], quarterly_key)
    upload_dataframe_to_s3(data["yearly"], yearly_key)


if __name__ == "__main__":
    # Quick sanity check test run for Reliance Industries
    sample_ticker = "RELIANCE.NS"
    process_and_stream_ticker(sample_ticker)