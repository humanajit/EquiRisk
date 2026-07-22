import numpy as np
import pandas as pd


def categorize_risk_tier(score: float) -> str:
    """Classifies risk score (0-100) into actionable risk categories."""
    if score < 25:
        return "LOW_RISK"
    elif score < 50:
        return "MODERATE_RISK"
    elif score < 75:
        return "HIGH_RISK"
    else:
        return "VERY_HIGH_RISK"


def predict_stock_risk(model, feature_dict: dict) -> dict:
    """
    Evaluates stock features and returns composite risk metrics.

    :param model: Trained XGBoost model instance
    :param feature_dict: Dictionary containing stock risk indicators
    """
    feature_order = [
        "annualized_volatility",
        "max_drawdown",
        "debt_to_equity_ratio",
        "sentiment_risk_impact",
        "compound_score"
    ]

    input_df = pd.DataFrame([feature_dict])[feature_order]
    predicted_score = float(model.predict(input_df)[0])
    bounded_score = round(float(np.clip(predicted_score, 0, 100)), 2)

    return {
        "symbol": feature_dict.get("symbol", "UNKNOWN"),
        "composite_risk_score": bounded_score,
        "risk_tier": categorize_risk_tier(bounded_score),
        "key_risk_drivers": {
            "volatility_contribution": "HIGH" if feature_dict.get("annualized_volatility", 0) > 0.3 else "NORMAL",
            "leverage_warning": feature_dict.get("debt_to_equity_ratio", 0) > 2.0,
            "negative_news_flag": feature_dict.get("compound_score", 0) < -0.2
        }
    }