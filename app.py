import numpy as np
import pandas as pd
import streamlit as st
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# Load shared artifacts
scaler = joblib.load(MODELS_DIR / "scaler.pkl")
feature_columns = joblib.load(MODELS_DIR / "feature_columns.pkl")
metrics_df = pd.read_csv(MODELS_DIR / "model_results.csv")

# Load raw data for options + charts
full_df = pd.read_csv(DATA_DIR / "youtube_ad_revenue_dataset.csv")

# Load all models into dict
MODEL_NAMES = [
    "LinearRegression",
    "Ridge",
    "Lasso",
    "RandomForest",
    "GradientBoosting",
]
MODELS = {
    name: joblib.load(MODELS_DIR / f"model_{name}.pkl") for name in MODEL_NAMES
}

st.set_page_config(page_title="YouTube Monetization Modeler", layout="wide")

st.title("YouTube Monetization Modeler")
st.markdown(
    "Choose a **regression algorithm** from the dropdown, compare metrics, "
    "and see predictions for your input."
)

# Sidebar: algorithm choice
st.sidebar.header("Model & Inputs")
model_choice = st.sidebar.selectbox("Choose Regression Algorithm", MODEL_NAMES)

# Sidebar: feature inputs
views = st.sidebar.number_input("Views", min_value=0, value=10000, step=1000)
likes = st.sidebar.number_input("Likes", min_value=0, value=500, step=50)
comments = st.sidebar.number_input("Comments", min_value=0, value=50, step=10)
watch_time = st.sidebar.number_input(
    "Total Watch Time (minutes)", min_value=0.0, value=10000.0, step=500.0
)
video_length = st.sidebar.number_input(
    "Video Length (minutes)", min_value=0.1, value=10.0, step=0.5
)
subscribers = st.sidebar.number_input(
    "Channel Subscribers", min_value=0, value=50000, step=5000
)

category = st.sidebar.selectbox("Category", sorted(full_df["category"].unique()))
device = st.sidebar.selectbox("Device", sorted(full_df["device"].unique()))
country = st.sidebar.selectbox("Country", sorted(full_df["country"].unique()))

predict_button = st.sidebar.button("Predict Revenue")


def build_feature_vector():
    data = {
        "views": views,
        "likes": likes,
        "comments": comments,
        "watch_time_minutes": watch_time,
        "video_length_minutes": video_length,
        "subscribers": subscribers,
        "engagement_rate": (likes + comments) / views if views > 0 else 0,
        "likes_per_view": likes / views if views > 0 else 0,
        "comments_per_view": comments / views if views > 0 else 0,
        "avg_watch_time_per_view": watch_time / views if views > 0 else 0,
        "year": 2025,
        "month": 1,
        "dayofweek": 0,
        "log_views": np.log1p(views),
        "log_subscribers": np.log1p(subscribers),
        "log_watch_time_minutes": np.log1p(watch_time),
    }

    X_input = pd.DataFrame([data])

    # one-hot columns
    for col in feature_columns:
        if col.startswith("category_"):
            cat_val = col.split("category_")[1]
            X_input[col] = 1 if category == cat_val else 0
        elif col.startswith("device_"):
            dev_val = col.split("device_")[1]
            X_input[col] = 1 if device == dev_val else 0
        elif col.startswith("country_"):
            c_val = col.split("country_")[1]
            X_input[col] = 1 if country == c_val else 0
        elif col not in X_input.columns:
            if col.startswith("log_"):
                base = col.replace("log_", "")
                if base in X_input.columns:
                    X_input[col] = np.log1p(X_input[base])
                else:
                    X_input[col] = 0
            else:
                X_input[col] = 0

    X_input = X_input[feature_columns]
    return X_input


col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Prediction")

    if predict_button:
        X_input = build_feature_vector()
        # scale for all models (we trained RF/GB on raw, but scaling doesn't break them badly;
        # if you want to be exact, you can branch by model name like in training script)
        X_scaled = scaler.transform(X_input)

        model = MODELS[model_choice]
        # branch exactly like training:
        if model_choice in ["RandomForest", "GradientBoosting"]:
            y_pred = model.predict(X_input)
        else:
            y_pred = model.predict(X_scaled)

        revenue = float(y_pred[0])
        st.metric(
            f"Estimated Ad Revenue (USD) using {model_choice}",
            f"${revenue:,.2f}",
        )
    else:
        st.info("Select a model and inputs, then click **Predict Revenue**.")

    st.subheader("Sample: Views vs Revenue")
    sample = full_df.sample(2000, random_state=42)
    st.scatter_chart(sample[["views", "ad_revenue_usd"]], x="views", y="ad_revenue_usd")

with col_right:
    st.subheader("Model Metrics (Test Set)")

    st.dataframe(
        metrics_df.set_index("Model").loc[MODEL_NAMES],
        use_container_width=True,
    )

    st.markdown(
        "Use the dropdown to switch between algorithms and see how RMSE, "
        "MAE, and R² change."
    )
