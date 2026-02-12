import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import math

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "youtube_ad_revenue_dataset.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42


def main():
    df = pd.read_csv(DATA_PATH)
    print("Initial shape:", df.shape)

    # Drop duplicates
    df = df.drop_duplicates()

    # Date features
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["dayofweek"] = df["date"].dt.dayofweek

    # Missing values
    for col in ["likes", "comments", "watch_time_minutes"]:
        df[col] = df[col].fillna(df[col].median())

    # Feature engineering
    df["engagement_rate"] = (df["likes"] + df["comments"]) / df["views"].replace(
        0, np.nan
    )
    df["engagement_rate"] = df["engagement_rate"].fillna(0)

    df["likes_per_view"] = df["likes"] / df["views"].replace(0, np.nan)
    df["likes_per_view"] = df["likes_per_view"].fillna(0)

    df["comments_per_view"] = df["comments"] / df["views"].replace(0, np.nan)
    df["comments_per_view"] = df["comments_per_view"].fillna(0)

    df["avg_watch_time_per_view"] = df["watch_time_minutes"] / df["views"].replace(
        0, np.nan
    )
    df["avg_watch_time_per_view"] = df["avg_watch_time_per_view"].fillna(0)

    for col in ["views", "subscribers", "watch_time_minutes", "ad_revenue_usd"]:
        df[f"log_{col}"] = np.log1p(df[col])

    # One‑hot encoding (NO video_id)
    cat_cols = ["category", "device", "country"]
    df_model = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Final NaN handling
    num_cols = df_model.select_dtypes(include=[np.number]).columns
    df_model[num_cols] = df_model[num_cols].fillna(df_model[num_cols].median())
    df_model = df_model.fillna(0)

    target = "ad_revenue_usd"
    drop_cols = ["video_id", "date"]

    X = df_model.drop(columns=drop_cols + [target])
    y = df_model[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Lasso": Lasso(alpha=0.0005, random_state=RANDOM_STATE),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }

    results = []

    for name, model in models.items():
        print(f"\nTraining model: {name}")
        if name in ["RandomForest", "GradientBoosting"]:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        else:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

        rmse = math.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results.append(
            {
                "Model": name,
                "RMSE": rmse,
                "MAE": mae,
                "R2": r2,
            }
        )

        # save each model separately for dropdown
        joblib.dump(model, MODELS_DIR / f"model_{name}.pkl")

    results_df = pd.DataFrame(results).sort_values(by="R2", ascending=False)
    print("\nModel performance:")
    print(results_df)

    # Choose best for feature importances
    best_name = results_df.iloc[0]["Model"]
    best_model = joblib.load(MODELS_DIR / f"model_{best_name}.pkl")

    if best_name in ["RandomForest", "GradientBoosting"]:
        importances = pd.Series(best_model.feature_importances_, index=X.columns)
    else:
        coefs = pd.Series(best_model.coef_, index=X.columns)
        importances = coefs.abs()
    importances = importances.sort_values(ascending=False)

    importances.to_csv(MODELS_DIR / "feature_importances.csv")
    results_df.to_csv(MODELS_DIR / "model_results.csv", index=False)

    # Save common artifacts
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(X.columns.tolist(), MODELS_DIR / "feature_columns.pkl")

    print(f"Saved models and metrics to {MODELS_DIR}")


if __name__ == "__main__":
    main()
