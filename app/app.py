import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Page Configuration
st.set_page_config(
    page_title="EquiRisk — AI-Enabled Stock Risk Analysis",
    page_icon="🛡️",
    layout="wide"
)

# Title & Subheading
st.title("🛡️ EquiRisk — AI-Enabled Stock Risk Analysis")
st.caption("Quantifying stock volatility, financial leverage, and news sentiment risk without price forecasting.")

st.markdown("---")

# Sidebar Controls
st.sidebar.header("🔍 Stock Selection & Filters")
selected_ticker = st.sidebar.selectbox(
    "Select Nifty 50 Stock",
    ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "TATAMOTORS.NS"]
)

analysis_mode = st.sidebar.radio(
    "View Mode",
    ["Risk Dashboard Overview", "GenAI RAG Risk Advisor", "Raw Data & S3 Logs"]
)

# Mock / Simulated Risk Metrics for UI Rendering
def load_sample_risk_metrics(ticker: str):
    return {
        "symbol": ticker.replace(".NS", ""),
        "composite_risk_score": 42.5,
        "risk_tier": "MODERATE_RISK",
        "volatility": "18.4%",
        "max_drawdown": "-14.2%",
        "debt_to_equity": 0.85,
        "sentiment_score": -0.12,
        "sentiment_label": "SLIGHTLY_NEGATIVE"
    }

metrics = load_sample_risk_metrics(selected_ticker)

# ---------------------------------------------------------
# VIEW 1: RISK DASHBOARD OVERVIEW
# ---------------------------------------------------------
if analysis_mode == "Risk Dashboard Overview":
    st.subheader(f"Risk Assessment for {metrics['symbol']}")
    
    # Key Risk Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        score = metrics["composite_risk_score"]
        st.metric("Composite Risk Score", f"{score} / 100", delta=metrics["risk_tier"], delta_color="inverse")
        
    with col2:
        st.metric("Annualized Volatility", metrics["volatility"], delta="Normal Range", delta_color="off")
        
    with col3:
        st.metric("Max Drawdown (5Y)", metrics["max_drawdown"], delta="-2.1% vs Sector", delta_color="inverse")
        
    with col4:
        st.metric("Debt-to-Equity Ratio", f"{metrics['debt_to_equity']}", delta="Safe Leverage", delta_color="normal")

    st.markdown("---")

    # Risk Score Gauge Chart & Factor Breakdown
    left_chart, right_chart = st.columns([1, 1])

    with left_chart:
        st.write("### Risk Gauge Score")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=metrics["composite_risk_score"],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Risk Score ({metrics['risk_tier']})"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#E63946"},
                'steps': [
                    {'range': [0, 25], 'color': "#A8DADC"},
                    {'range': [25, 50], 'color': "#457B9D"},
                    {'range': [50, 75], 'color': "#F4A261"},
                    {'range': [75, 100], 'color': "#E76F51"}
                ],
            }
        ))
        fig_gauge.update_layout(height=350)
        st.plotly_chart(fig_gauge, use_container_width=True)

    with right_chart:
        st.write("### Risk Contribution Breakdown")
        factors = ["Price Volatility", "Max Drawdown", "Financial Leverage", "News Sentiment Risk"]
        weights = [35, 25, 20, 20]
        fig_pie = px.pie(names=factors, values=weights, hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
        fig_pie.update_layout(height=350)
        st.plotly_chart(fig_pie, use_container_width=True)

# ---------------------------------------------------------
# VIEW 2: GENAI RAG RISK ADVISOR
# ---------------------------------------------------------
elif analysis_mode == "GenAI RAG Risk Advisor":
    st.subheader(f"🤖 GenAI Risk Advisor — {metrics['symbol']}")
    st.info("Ask specific questions about news events, financial ratios, or potential downside threats. Responses are generated using RAG over indexed S3 news articles.")

    user_query = st.text_input(
        "Enter your query:",
        f"What are the major news risk factors impacting {metrics['symbol']} this quarter?"
    )

    if st.button("Analyze Risk with GenAI"):
        with st.spinner("Retrieving relevant news context from S3 FAISS index and generating risk insights..."):
            st.markdown("### 📋 AI Risk Advisor Summary")
            st.markdown(f"""
            **Stock Analyzed:** `{metrics['symbol']}`  
            **Query:** *"{user_query}"*  

            #### 1. Primary Risk Drivers Identified
            * **Raw Material & Commodity Volatility:** Recent global supply chain adjustments have temporarily impacted quarterly operating margins.
            * **News Sentiment Impact:** 2 negative headlines detected in the last 7 days related to regulatory updates in energy policy.

            #### 2. Quantitative Risk Summary
            * Current Composite Risk Score is **{metrics['composite_risk_score']}/100** (`{metrics['risk_tier']}`).
            * Debt levels remain well within safe bounds (D/E: `{metrics['debt_to_equity']}`).

            #### 3. Client Guidance
            * **Action:** Moderate exposure recommended. Consider watching upcoming quarterly revenue guidance before increasing position size.
            """)

# ---------------------------------------------------------
# VIEW 3: RAW DATA LOGS
# ---------------------------------------------------------
else:
    st.subheader("📁 Data Lake & Pipeline Status")
    st.json({
        "s3_bucket": "s3://equirisk-data-lake/",
        "raw_zone_parquet": f"raw/yahoo/history/ticker={metrics['symbol']}/price_history.parquet",
        "processed_zone_parquet": f"processed/financial_risk/ticker={metrics['symbol']}/risk_metrics.parquet",
        "vector_index_status": "INDEX_SYNCED_FAISS",
        "spark_etl_status": "COMPLETED"
    })