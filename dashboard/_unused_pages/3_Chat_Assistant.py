"""
dashboard/pages/3_Chat_Assistant.py

Chat interface: user picks a ticker (defaults to whatever was last
viewed on the Company Detail page, via st.session_state) and asks
questions. Each query goes through src/rag/llm_client.answer_query(),
which handles retrieval + Groq generation internally -- this page only
deals with UI and chat history state.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import yaml

from src.rag.llm_client import answer_query
from src.utils.s3_io import read_hive_partitioned_parquet_s3
import pandas as pd

st.set_page_config(page_title="EquiRisk - Chat Assistant", layout="wide")


@st.cache_data(ttl=300)
def _load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)


@st.cache_data(ttl=300)
def get_available_tickers() -> list:
    config = _load_config()
    bucket = config["s3"]["bucket"]
    prefix = config["s3"]["paths"]["processed_features"]
    try:
        full = read_hive_partitioned_parquet_s3(prefix, bucket, partition_col="ticker")
        return sorted(full["ticker"].unique().tolist())
    except RuntimeError:
        return []


st.title("Chat Assistant")
st.caption("Ask questions about a specific company's risk profile -- answers are grounded in retrieved news and computed stats.")

tickers = get_available_tickers()

if not tickers:
    st.warning("No processed data available yet. Go to the main page and click 'Refresh Pipeline'.")
else:
    default_ticker = st.session_state.get("selected_ticker", tickers[0])
    default_index = tickers.index(default_ticker) if default_ticker in tickers else 0
    selected_ticker = st.selectbox("Company", tickers, index=default_index)

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = {}
    if selected_ticker not in st.session_state["chat_history"]:
        st.session_state["chat_history"][selected_ticker] = []

    history = st.session_state["chat_history"][selected_ticker]

    for role, message in history:
        with st.chat_message(role):
            st.write(message)

    user_query = st.chat_input(f"Ask about {selected_ticker}'s risk profile...")

    if user_query:
        history.append(("user", user_query))
        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = answer_query(selected_ticker, user_query)
                st.write(answer)
        history.append(("assistant", answer))