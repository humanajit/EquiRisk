"""
dashboard/pages/2_Company_Detail.py

Per-company deep dive: price chart with moving averages, rolling
volatility trend, sentiment trend, and the current risk label with a
plain-language explanation. Ticker selection is a dropdown (no URL/query
param state needed for a course project's scope).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

from src.utils.s3_io import read_hive_partitioned_parquet_s3

st.set_page_config(page_title="EquiRisk - Company Detail", layout="wide")


@st.cache_data(ttl=300)
def _load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)


@st.cache_data(ttl=300)
def load_full_table() -> pd.DataFrame:
    config = _load_config()
    bucket = config["s3"]["bucket"]
    prefix = config["s3"]["paths"]["processed_features"]

    try:
        return read_hive_partitioned_parquet_s3(prefix, bucket, partition_col="ticker")
    except RuntimeError:
        return pd.DataFrame()


def price_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["close"], name="Close", line=dict(color="#2563eb")))
    if "ma_20d" in df.columns:
        fig.add_trace(go.Scatter(x=df["date"], y=df["ma_20d"], name="20d MA", line=dict(color="#f59e0b", dash="dot")))
    if "ma_60d" in df.columns:
        fig.add_trace(go.Scatter(x=df["date"], y=df["ma_60d"], name="60d MA", line=dict(color="#10b981", dash="dot")))
    fig.update_layout(title=f"{ticker} -- Price", height=350, margin=dict(t=40, b=20))
    return fig


def volatility_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["volatility_20d"], name="20d Volatility", line=dict(color="#ef4444")))
    fig.add_trace(go.Scatter(x=df["date"], y=df["volatility_60d"], name="60d Volatility", line=dict(color="#f97316")))
    fig.update_layout(title=f"{ticker} -- Rolling Volatility", height=300, margin=dict(t=40, b=20))
    return fig


def sentiment_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["date"], y=df["daily_sentiment"], name="Daily Sentiment", marker_color="#94a3b8"))
    fig.add_trace(go.Scatter(x=df["date"], y=df["sentiment_3d_avg"], name="3d Avg", line=dict(color="#7c3aed")))
    fig.update_layout(title=f"{ticker} -- News Sentiment", height=300, margin=dict(t=40, b=20))
    return fig


st.title("Company Detail")

full_df = load_full_table()

if full_df.empty:
    st.warning("No processed data available yet. Go to the main page and click 'Refresh Pipeline'.")
else:
    tickers = sorted(full_df["ticker"].unique().tolist())
    selected_ticker = st.selectbox("Select a company", tickers)

    ticker_df = full_df[full_df["ticker"] == selected_ticker].sort_values("date")
    latest = ticker_df.iloc[-1]

    risk_colors = {"Low": "green", "Medium": "orange", "High": "red"}
    label = latest.get("risk_label", "Unknown")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Risk Label", label)
    col2.metric("Last Close", f"\u20B9{latest['close']:.2f}")
    col3.metric("20d Volatility", f"{latest.get('volatility_20d', 0):.4f}")
    col4.metric("Sentiment (3d avg)", f"{latest.get('sentiment_3d_avg', 0):.3f}")

    st.markdown(f":{risk_colors.get(label, 'gray')}[**Current risk level: {label}**]")

    st.plotly_chart(price_chart(ticker_df, selected_ticker), use_container_width=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(volatility_chart(ticker_df, selected_ticker), use_container_width=True)
    with col_b:
        st.plotly_chart(sentiment_chart(ticker_df, selected_ticker), use_container_width=True)

    st.session_state["selected_ticker"] = selected_ticker  # so Chat page can default to this