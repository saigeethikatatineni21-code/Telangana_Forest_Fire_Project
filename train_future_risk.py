import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# 1. LOAD DATA
# ============================================================

INPUT_FILE = "telangana_forest_fire_dataset.csv"

print("=" * 60)
print("TELANGANA FUTURE FIRE RISK MODEL")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print(f"Loaded records: {len(df):,}")

# Convert date
df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")

df = df.dropna(subset=["acq_date", "latitude", "longitude"])

# ============================================================
# 2. CREATE SPATIAL GRID
# ============================================================
# Each fire is assigned to a 0.25 degree grid cell.
# This lets us calculate fire activity by location and date.

GRID_SIZE = 0.25

df["grid_lat"] = np.floor(df["latitude"] / GRID_SIZE) * GRID_SIZE
df["grid_lon"] = np.floor(df["longitude"] / GRID_SIZE) * GRID_SIZE

print(f"Grid size: {GRID_SIZE} degrees")

# ============================================================
# 3. DAILY FIRE ACTIVITY
# ============================================================

daily = (
    df.groupby(["grid_lat", "grid_lon", "acq_date"])
      .agg(
          fire_count=("fire_id", "count"),
          mean_frp=("frp", "mean"),
          max_frp=("frp", "max")
      )
      .reset_index()
)

print(f"Daily grid records: {len(daily):,}")

# ============================================================
# 4. CREATE COMPLETE DATE RANGE
# ============================================================

min_date = daily["acq_date"].min()
max_date = daily["acq_date"].max()

dates = pd.date_range(min_date, max_date, freq="D")

grid = daily[["grid_lat", "grid_lon"]].drop_duplicates()

print(f"Date range: {min_date.date()} to {max_date.date()}")
print(f"Grid cells: {len(grid):,}")

# Create every grid-cell/date combination
grid["key"] = 1
date_df = pd.DataFrame({"acq_date": dates})
date_df["key"] = 1

full = grid.merge(date_df, on="key").drop(columns="key")

# Add observed fire information
full = full.merge(
    daily,
    on=["grid_lat", "grid_lon", "acq_date"],
    how="left"
)

# No detection = zero fire activity
full["fire_count"] = full["fire_count"].fillna(0)
full["mean_frp"] = full["mean_frp"].fillna(0)
full["max_frp"] = full["max_frp"].fillna(0)

full = full.sort_values(
    ["grid_lat", "grid_lon", "acq_date"]
).reset_index(drop=True)

# ============================================================
# 5. CALENDAR FEATURES
# ============================================================

full["month"] = full["acq_date"].dt.month
full["day_of_year"] = full["acq_date"].dt.dayofyear

# Cyclic representation of the year
full["sin_day"] = np.sin(
    2 * np.pi * full["day_of_year"] / 365.25
)

full["cos_day"] = np.cos(
    2 * np.pi * full["day_of_year"] / 365.25
)

# ============================================================
# 6. HISTORICAL / LAG FEATURES
# ============================================================

print("\nCreating historical features...")

group_cols = ["grid_lat", "grid_lon"]

full = full.sort_values(
    group_cols + ["acq_date"]
).reset_index(drop=True)

# Previous-day fire count
full["fire_lag_1"] = (
    full.groupby(group_cols)["fire_count"].shift(1)
)

# Previous 3-day fire count
full["fire_lag_3"] = (
    full.groupby(group_cols)["fire_count"].shift(3)
)

# Previous 7-day fire count
full["fire_lag_7"] = (
    full.groupby(group_cols)["fire_count"].shift(7)
)

# Previous fire count and FRP
full["previous_fire_count"] = (
    full.groupby(group_cols)["fire_count"].shift(1)
)

full["previous_frp"] = (
    full.groupby(group_cols)["mean_frp"].shift(1)
)

# Previous 7-day fire count
full["fire_rolling_7"] = (
    full.groupby(group_cols)["previous_fire_count"]
    .transform(lambda x: x.rolling(7, min_periods=7).sum())
)

# Previous 30-day fire count
full["fire_rolling_30"] = (
    full.groupby(group_cols)["previous_fire_count"]
    .transform(lambda x: x.rolling(30, min_periods=30).sum())
)

# Previous 7-day average FRP
full["frp_rolling_7"] = (
    full.groupby(group_cols)["previous_frp"]
    .transform(lambda x: x.rolling(7, min_periods=7).mean())
)

# Previous 30-day average FRP
full["frp_rolling_30"] = (
    full.groupby(group_cols)["previous_frp"]
    .transform(lambda x: x.rolling(30, min_periods=30).mean())
)

# Number of fire-active days during previous 30 days
full["fire_days_30"] = (
    full.groupby(group_cols)["previous_fire_count"]
    .transform(
        lambda x: x.gt(0)
        .rolling(30, min_periods=30)
        .sum()
    )
)

# Remove temporary columns
full = full.drop(
    columns=["previous_fire_count", "previous_frp"]
)

print("Historical features created.")


# ============================================================
# 7. FUTURE FIRE TARGET
# ============================================================

print("\nCreating future fire target...")

# Fire count on the NEXT day
full["future_fire_count"] = (
    full.groupby(group_cols)["fire_count"].shift(-1)
)

# 1 = fire activity on next day
# 0 = no fire activity on next day
full["future_fire"] = (
    full["future_fire_count"] > 0
).astype(int)

# Remove final day of each grid cell because
# there is no known next day in the dataset
full = full[
    full["future_fire_count"].notna()
].copy()

full = full.drop(
    columns=["future_fire_count"]
)

print(
    "Future target created."
)

print(
    "Future fire distribution:"
)

print(
    full["future_fire"].value_counts()
)


# ============================================================
# 8. MODEL FEATURES
# ============================================================

features = [
    "grid_lat",
    "grid_lon",
    "month",
    "day_of_year",
    "fire_lag_1",
    "fire_lag_3",
    "fire_lag_7",
    "fire_rolling_7",
    "fire_rolling_30",
    "frp_rolling_7",
    "frp_rolling_30",
    "fire_days_30"
]

target = "future_fire"

# Remove rows where historical features are unavailable
model_data = full.dropna(
    subset=features + [target]
).copy()

print("\nModel-ready records:", len(model_data))


# ============================================================
# 9. TIME-BASED TRAIN / TEST SPLIT
# ============================================================

print("\n============================================================")
print("TIME-BASED SPLIT")
print("------------------------------------------------------------")

split_date = pd.Timestamp("2025-01-01")

train_data = model_data[
    model_data["acq_date"] < split_date
].copy()

test_data = model_data[
    model_data["acq_date"] >= split_date
].copy()

X_train = train_data[features]
y_train = train_data[target]

X_test = test_data[features]
y_test = test_data[target]

print("Training records:", len(train_data))
print("Testing records :", len(test_data))

print(
    "Training fire rate:",
    round(y_train.mean() * 100, 2),
    "%"
)

print(
    "Testing fire rate :",
    round(y_test.mean() * 100, 2),
    "%"
)


# ============================================================
# 10. RANDOM FOREST MODEL
# ============================================================

print("\nTraining Random Forest...")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

model = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced_subsample",
    min_samples_leaf=2
)

model.fit(X_train, y_train)

print("Training completed.")


# ============================================================
# 11. MODEL EVALUATION
# ============================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n============================================================")
print("FUTURE FIRE RISK MODEL RESULTS")
print("============================================================")

print(
    f"Accuracy : {accuracy:.4f} "
    f"({accuracy * 100:.2f}%)"
)

print(
    f"Precision: {precision:.4f} "
    f"({precision * 100:.2f}%)"
)

print(
    f"Recall   : {recall:.4f} "
    f"({recall * 100:.2f}%)"
)

print(
    f"F1 Score : {f1:.4f} "
    f"({f1 * 100:.2f}%)"
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 12. FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\n============================================================")
print("FEATURE IMPORTANCE")
print("============================================================")

print(
    importance.to_string(index=False)
)


# ============================================================
# 13. SAVE MODEL
# ============================================================

print("\nSaving model files...")

import joblib

joblib.dump(
    model,
    "future_fire_risk_model.pkl"
)

importance.to_csv(
    "future_risk_feature_importance.csv",
    index=False
)

metadata = {
    "features": features,
    "split_date": "2025-01-01",
    "grid_size": 0.25,
    "date_start": str(
        model_data["acq_date"].min().date()
    ),
    "date_end": str(
        model_data["acq_date"].max().date()
    ),
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1
}

joblib.dump(
    metadata,
    "future_risk_metadata.pkl"
)

print("\n============================================================")
print("MODEL SAVED")
print("============================================================")

print(
    "future_fire_risk_model.pkl"
)

print(
    "future_risk_feature_importance.csv"
)

print(
    "future_risk_metadata.pkl"
)

print("\nFuture fire risk model is ready.")