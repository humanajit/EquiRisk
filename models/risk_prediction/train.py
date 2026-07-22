import os
import io
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from data_pipeline.s3_uploader import upload_dataframe_to_s3


class RiskModelTrainer:
    def __init__(self):
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            random_state=42
        )

    def generate_composite_target(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculates a composite ground-truth risk score (0-100) based on weighted risk factors:
        - Volatility Impact (35%)
        - Max Drawdown Impact (25%)
        - Leverage/Debt Impact (20%)
        - Negative Sentiment Impact (20%)
        """
        vol_score = np.clip(df["annualized_volatility"] * 100, 0, 100) * 0.35
        dd_score = np.clip(abs(df["max_drawdown"]) * 100, 0, 100) * 0.25
        de_score = np.clip(df["debt_to_equity_ratio"] * 25, 0, 100) * 0.20
        sent_score = np.clip(df["sentiment_risk_impact"], 0, 100) * 0.20

        composite_score = vol_score + dd_score + de_score + sent_score
        return np.clip(composite_score, 0, 100)

    def train_and_evaluate(self, feature_df: pd.DataFrame):
        """Trains XGBoost Regressor on transformed features."""
        if feature_df.empty:
            print("Error: Training DataFrame is empty.")
            return

        feature_cols = [
            "annualized_volatility",
            "max_drawdown",
            "debt_to_equity_ratio",
            "sentiment_risk_impact",
            "compound_score"
        ]

        X = feature_df[feature_cols]
        y = self.generate_composite_target(feature_df)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("Training XGBoost Risk Prediction Model...")
        self.model.fit(X_train, y_train)

        predictions = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        print(f"Model Evaluation complete — RMSE: {rmse:.4f}, R2 Score: {r2:.4f}")
        return self.model


if __name__ == "__main__":
    # Dummy feature matrix to verify pipeline mechanics
    sample_data = pd.DataFrame({
        "annualized_volatility": [0.15, 0.35, 0.50, 0.22, 0.18],
        "max_drawdown": [-0.10, -0.30, -0.45, -0.15, -0.08],
        "debt_to_equity_ratio": [0.5, 2.1, 3.5, 1.2, 0.4],
        "sentiment_risk_impact": [10.0, 75.0, 90.0, 30.0, 5.0],
        "compound_score": [0.45, -0.65, -0.85, -0.10, 0.60]
    })

    trainer = RiskModelTrainer()
    trained_model = trainer.train_and_evaluate(sample_data)