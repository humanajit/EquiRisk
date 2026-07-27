"""
dashboard/pages/1_Overview.py

Landing view: a table/heatmap of the latest risk label for every
tracked ticker, sorted by risk so the highest-risk companies surface
first. Clicking through to a specific company is just "go to Company
Detail page and pick it from the dropdown" -- Streamlit's multi-page
nav doesn't easily support passing state via a table click without
extra plumbing, so keeping this simple is the right call for scope.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import streamlit as st
import yaml

from src.utils.s3_io import read_hive_partitioned_parquet_s3

st.set_page_config(page_title="EquiRisk - Overview", layout="wide")


@st.cache_data(ttl=300)
def _load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)


@st.cache_data(ttl=300)
def load_latest_snapshot() -> pd.DataFrame:
    """Loads every ticker's most recent row from the feature table --
    this is what the Overview table displays. Cached for 5 minutes so
    navigating between pages doesn't re-hit S3 every time; the Refresh
    Pipeline button explicitly clears this cache when new data lands.
    """
    config = _load_config()
    bucket = config["s3"]["bucket"]
    prefix = config["s3"]["paths"]["processed_features"]

    try:
        df = read_hive_partitioned_parquet_s3(prefix, bucket, partition_col="ticker")
    except RuntimeError:
        return pd.DataFrame()

    latest = df.sort_values("date").groupby("ticker").tail(1)
    return latest[[
        "ticker", "date", "close", "risk_label", "volatility_20d",
        "volatility_60d", "sentiment_3d_avg", "article_count",
    ]].sort_values("risk_label")


def _risk_color(label: str) -> str:
    return {"Low": "\U0001F7E2", "Medium": "\U0001F7E1", "High": "\U0001F534"}.get(label, "\u26AA")


st.title("Risk Overview")
st.caption("Latest computed risk label for every tracked Nifty150 midcap company")

snapshot = load_latest_snapshot()

if snapshot.empty:
    st.warning("No processed data available yet. Go to the main page and click 'Refresh Pipeline'.")
else:
    col1, col2, col3 = st.columns(3)
    col1.metric("Companies tracked", len(snapshot))
    col2.metric("High risk", int((snapshot["risk_label"] == "High").sum()))
    col3.metric("Low risk", int((snapshot["risk_label"] == "Low").sum()))

    st.divider()

    risk_filter = st.multiselect(
        "Filter by risk level", options=["Low", "Medium", "High"], default=["Low", "Medium", "High"]
    )
    filtered = snapshot[snapshot["risk_label"].isin(risk_filter)]

    display_df = filtered.copy()
    display_df["risk"] = display_df["risk_label"].apply(lambda l: f"{_risk_color(l)} {l}")
    display_df = display_df.drop(columns=["risk_label"])

    st.dataframe(
        display_df.rename(columns={
            "ticker": "Ticker", "date": "As of", "close": "Last Close",
            "volatility_20d": "20d Volatility", "volatility_60d": "60d Volatility",
            "sentiment_3d_avg": "Sentiment (3d avg)", "article_count": "Recent Articles",
            "risk": "Risk",
        }),
        use_container_width=True,
        hide_index=True,
    )