text
# YouTube Monetization Modeler

An end‑to‑end regression project to predict YouTube ad revenue for individual videos and compare multiple regression algorithms using an interactive Streamlit web application. [file:1]

---

## 1. Project Overview

### 1.1 Objective

This project builds machine learning models to estimate **daily YouTube ad revenue** (`ad_revenue_usd`) for a video using its performance and contextual features such as views, likes, comments, watch time, channel subscribers, category, device, and country. [file:1]

### 1.2 Business Use Cases

- **Content Strategy Optimization** – Identify what kind of videos (category, length, engagement) bring higher revenue.
- **Revenue Forecasting** – Estimate likely ad revenue for upcoming videos based on expected performance.
- **Creator Support Tools** – Integrate into dashboards/analytics tools for YouTubers.
- **Ad Campaign Planning** – Help advertisers approximate ROI from specific content profiles. [file:1]

---

## 2. Dataset

### 2.1 Source

- Name: **YouTube Monetization Modeler (Synthetic Dataset)**  
- Format: CSV  
- Size: ~122,400 rows, 12 columns. [file:1]  
- Source link (Google Drive):  

**Download here:**  
https://drive.google.com/drive/folders/1ybhXuva11b6zm20j35vXC33E75FS0sHN?usp=sharing [file:1]

Download `youtube_ad_revenue_dataset.csv` and place it in the `data/` folder (see structure below).

### 2.2 Data Description

Each row represents the performance metrics of one video on a specific day. [file:1]

Columns:

- `video_id` – Unique identifier of the video.
- `date` – Date/time of the observation.
- `views` – Number of views.
- `likes` – Number of likes.
- `comments` – Number of comments.
- `watch_time_minutes` – Total watch time (minutes).
- `video_length_minutes` – Length of the video (minutes).
- `subscribers` – Channel subscriber count at that time.
- `category` – Content category (e.g., Gaming, Tech, Education).
- `device` – Device type (e.g., Mobile, Desktop, Tablet, TV).
- `country` – Viewer country.
- `ad_revenue_usd` – Daily revenue generated (target variable). [file:1]

Notes:

- Dataset is **synthetic**, created for learning and experimentation.
- Approximately **5%** missing values in `likes`, `comments`, and `watch_time_minutes`. [file:1]
- Approximately **2%** duplicated records that should be removed in preprocessing. [file:1]

---

## 3. Project Structure

Recommended folder structure:

```text
youtube-monetization-modeler/
├─ data/
│   └─ youtube_ad_revenue_dataset.csv
├─ models/
│   ├─ model_LinearRegression.pkl
│   ├─ model_Ridge.pkl
│   ├─ model_Lasso.pkl
│   ├─ model_RandomForest.pkl
│   ├─ model_GradientBoosting.pkl
│   ├─ scaler.pkl
│   ├─ feature_columns.pkl
│   ├─ feature_importances.csv
│   └─ model_results.csv
├─ src/
│   └─ train_models.py
├─ app.py
├─ requirements.txt
└─ README.md
data/ – Raw dataset (downloaded from Drive). [file:1]

models/ – Trained models and preprocessing artifacts.

src/ – Python training script with EDA, preprocessing, model training, and evaluation.

app.py – Streamlit app for interactive usage.

requirements.txt – Python dependencies.

README.md – Project documentation (this file).

4. Tech Stack
Language: Python

Data Handling: Pandas, NumPy

Visualization / EDA: Matplotlib, Seaborn

Machine Learning: scikit‑learn

Web App: Streamlit

Model Persistence: joblib

5. Installation & Setup
5.1 Clone or Create Project Folder
Create the folder youtube-monetization-modeler and open it in VS Code (or clone your own repo with this structure).

5.2 Create Virtual Environment (Recommended)
bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
5.3 Install Dependencies
Create requirements.txt with:

text
pandas
numpy
scikit-learn
matplotlib
seaborn
streamlit
joblib
Then install:

bash
pip install -r requirements.txt
5.4 Download and Place Dataset
Open the Drive link:
https://drive.google.com/drive/folders/1ybhXuva11b6zm20j35vXC33E75FS0sHN?usp=sharing

Download youtube_ad_revenue_dataset.csv. [file:1]

Create a data/ directory in the project root and move the CSV into it:

text
youtube-monetization-modeler/
└─ data/
    └─ youtube_ad_revenue_dataset.csv
6. Modeling Approach
6.1 EDA (Exploratory Data Analysis)
Performed within src/train_models.py (printed summaries & saved plots):

Shape, basic statistics, missing values per column. [file:1]

Duplicate percentage and removal. [file:1]

Correlation matrix of numeric variables:

views, likes, comments, watch_time_minutes, video_length_minutes, subscribers, ad_revenue_usd. [file:1]

Visual checks (optional PNGs):

Correlation heatmap.

Scatter plots (e.g., views vs revenue).

Boxplots of revenue by category. [file:1]

6.2 Preprocessing
Steps implemented in train_models.py:

Drop duplicates from the dataset. [file:1]

Parse dates:

Convert date to datetime.

Create additional features: year, month, dayofweek. [file:1]

Handle missing values:

For likes, comments, watch_time_minutes: fill with median of each column. [file:1]

Feature engineering:

engagement_rate = (likes + comments) / views

likes_per_view = likes / views

comments_per_view = comments / views

avg_watch_time_per_view = watch_time_minutes / views

Log transforms:

log_views

log_subscribers

log_watch_time_minutes

log_ad_revenue_usd (optionally used for analysis) [file:1]

Encoding categorical variables:

One‑hot encoding using pd.get_dummies:

category

device

country

video_id is treated as an ID and not one‑hot encoded. [file:1]

Final NaN safety:

All numeric columns imputed with median if any NaNs remain.

Non‑numeric NaNs filled as 0. [file:1]

6.3 Train–Test Split and Scaling
Input matrix X: all engineered features and one‑hot columns except video_id, date, and target. [file:1]

Target y: ad_revenue_usd. [file:1]

Train/test split: 80% train, 20% test (random_state=42).

Standardization (mean 0, std 1) using StandardScaler:

Applied to X_train / X_test for linear models (Linear, Ridge, Lasso).

Tree‑based models (RandomForest, GradientBoosting) trained on unscaled X. [file:1]

6.4 Models Implemented
Five regression algorithms:

Linear Regression

Ridge Regression

Lasso Regression

Random Forest Regressor

Gradient Boosting Regressor

Each model is trained and evaluated on the same train/test split.

6.5 Evaluation Metrics
For each model, the following are computed for both train and test sets:

R² Score – Proportion of variance explained.

Root Mean Squared Error (RMSE) – Penalizes larger errors more.

Mean Absolute Error (MAE) – Average absolute difference between actual and predicted revenue. [file:1]

Metrics are saved to:

models/model_results.csv

Example columns:

Model

RMSE_train, MAE_train, R2_train

RMSE_test, MAE_test, R2_test

Use these to identify overfitting (train >> test performance) or underfitting (both poor). [file:1]

6.6 Feature Importance
For the best‑performing model (by R² test score):

If tree‑based:

Use feature_importances_ attribute.

If linear:

Use absolute value of coefficients.

Top features are saved to:

models/feature_importances.csv

These help understand which features (e.g., views, watch time, subscribers, category, country) drive ad revenue the most. [file:1]

7. Training the Models
All training logic is in src/train_models.py.

Run from project root:

bash
python src/train_models.py
This script will:

Load data/youtube_ad_revenue_dataset.csv. [file:1]

Run EDA and preprocessing.

Engineer features and encode categorical variables.

Split into train and test.

Train 5 regression models.

Evaluate on both train and test sets using RMSE, MAE, R².

Save:

Individual models: models/model_<ModelName>.pkl

Scaler: models/scaler.pkl

Feature column list: models/feature_columns.pkl

Feature importances: models/feature_importances.csv

Model metrics: models/model_results.csv

You must run this once before launching the Streamlit app.

8. Streamlit Application
The web app is defined in app.py.

8.1 Running the App
From the project root:

bash
streamlit run app.py
A local URL (e.g., http://localhost:8501) will open in your browser.

8.2 App Features
Model Selection (Dropdown)

Choose one of the 5 regression algorithms:

LinearRegression

Ridge

Lasso

RandomForest

GradientBoosting

The selected model is used for prediction and highlighted in the metrics table.

User Inputs (Sidebar)
Input fields for:

Views

Likes

Comments

Total watch time (minutes)

Video length (minutes)

Channel subscribers

Category (select box)

Device (select box)

Country (select box)

Prediction Panel

On clicking Predict Revenue:

A feature vector is built (with same engineered features and one‑hot encoding as training).

Features are scaled if required for the chosen model.

The selected model predicts ad_revenue_usd.

Negative predictions (possible from linear models) are clipped to 0 for business sense.

Displayed as:

Estimated Ad Revenue (USD) using <ModelName>

Model Metrics Table

Reads models/model_results.csv and shows for each model:

RMSE_test

MAE_test

R2_test

Allows you to compare model quality on the same test set.

Visual Analytics

Scatter chart: sample views vs ad_revenue_usd to visualize revenue distribution. [file:1]

Optional bar chart: top feature importances using feature_importances.csv.

9. Interpreting Model Quality
Use model_results.csv and the app metrics table:

Overfitting:

Train R² very high, test R² notably lower.

Train RMSE/MAE much smaller than test RMSE/MAE.

Underfitting:

Both train and test R² low.

Both RMSE/MAE high and similar.

Good Generalization:

Train and test metrics close.

Test R² reasonably high and errors acceptable.

You can switch models in the dropdown and quickly see how their performance and predictions differ.

10. Possible Extensions
Train on log_ad_revenue_usd and reverse transform (expm1) during prediction to naturally avoid negative outputs.

Perform hyperparameter tuning (GridSearchCV / RandomizedSearchCV) to optimize RandomForest and GradientBoosting models.

Add more visual analytics in the Streamlit app (category‑wise revenue, country‑wise patterns, device‑level comparison).

Save and load multiple experiment runs (e.g., different feature sets, different splits). [file:1]

11. How to Use This Project for Learning
Study the EDA and preprocessing parts in train_models.py to understand how real‑world regression problems are structured.

Experiment by:

Adding / removing features.

Changing the target to log‑transformed revenue.

Modifying algorithms and hyperparameters.

Use the Streamlit app to interactively see how model type and input values affect predictions, which is useful for building intuition about regression models. [file:1]

text
undefined
