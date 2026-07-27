"""
src/ingestion/fetch_news.py

Pulls recent news articles per ticker from marketaux and writes the raw
API response straight to S3 (raw/news/...). No sentiment scoring or
cleaning happens here -- that's the ETL stage's job (src/etl/sentiment.py).

marketaux realistically only covers recent news (weeks/months, not 5
years), so this is meant to run incrementally -- each run's output is
keyed by fetch date, so re-running the same day overwrites that day's
file rather than duplicating it.
"""

import logging
import os
import time

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv

from src.utils.s3_io import put_json, dated_key

load_dotenv()
logger = logging.getLogger("equirisk.ingestion.news")
logging.basicConfig(level=logging.INFO)

MARKETAUX_BASE_URL = "https://api.marketaux.com/v1/news/all"


def _load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _load_ticker_list(config: dict) -> list:
    """Reuses the same master ticker CSV as price ingestion, but returns
    bare symbols (no .NS suffix) -- marketaux expects plain symbols."""
    path = config["tickers"]["master_list_path"]
    df = pd.read_csv(path)
    return df["symbol"].tolist()


def fetch_ticker_news(symbol: str, api_key: str, lookback_days: int, limit: int) -> dict:
    """Fetch recent news for one ticker. Returns empty dict on failure
    rather than raising -- one bad ticker/rate-limit hit shouldn't kill
    the whole ingestion run."""
    params = {
        "symbols": symbol,
        "filter_entities": "true",
        "language": "en",
        "limit": limit,
        "api_token": api_key,
    }
    try:
        resp = requests.get(MARKETAUX_BASE_URL, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch news for {symbol}: {e}")
        return {}


def fetch_all_tickers_news(config_path: str = "config/config.yaml", sleep_sec: float = 1.0) -> None:
    """Main ingestion entrypoint, called by the orchestrator. Fetches
    news for every ticker in the master list and writes each one's raw
    response to S3 as JSON under raw/news/{ticker}/{date}.json.

    sleep_sec throttles requests -- marketaux free-tier plans have a
    daily request cap, so check your plan's quota against len(tickers)
    before running this on the full 150-ticker list; you may need to
    batch across multiple days or upgrade the plan.
    """
    config = _load_config(config_path)
    tickers = _load_ticker_list(config)
    api_key = os.environ["MARKETAUX_API_KEY"]
    lookback_days = config["ingestion"]["news"]["lookback_days"]
    limit = config["ingestion"]["news"]["articles_per_ticker"]
    raw_prefix = config["s3"]["paths"]["raw_news"]

    logger.info(f"Fetching news for {len(tickers)} tickers (limit={limit} articles each)")

    success_count = 0
    for symbol in tickers:
        data = fetch_ticker_news(symbol, api_key, lookback_days, limit)
        if not data or "data" not in data:
            continue

        key = dated_key(raw_prefix, symbol, ext="json")
        put_json(key, data)
        success_count += 1
        time.sleep(sleep_sec)

    logger.info(f"News ingestion complete: {success_count}/{len(tickers)} tickers succeeded")

    if success_count == 0:
        raise RuntimeError("News ingestion failed for all tickers -- check API key/quota/network")


if __name__ == "__main__":
    fetch_all_tickers_news()