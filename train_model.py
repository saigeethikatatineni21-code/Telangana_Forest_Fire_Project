import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
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
# FILE
# ============================================================

DATA_FILE = "telangana_forest_fire_dataset.csv"
MODEL_FILE = "fire_model.pkl"

print("=" * 70)
print("TELANGANA FOREST FIRE - RANDOM FOREST CLASSIFICATION")
print("=" * 70)

# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n[1] Loading dataset...")

df = pd.read_csv(DATA_FILE)

print(f"Total records: {len(df):,}")

# ============================================================
# 2. SELECT FEATURES
# ============================================================

features = [
    "latitude",
    "longitude",
    "month",
    "day_of_year",
    "bright_ti4",
    "bright_ti5",
    "scan",
    "track"
]

target = "fire_severity"

# Keep only required columns
model_df = df[features + [target]].copy()

# Remove rows with missing values
model_df = model_df.dropna()

print(f"Records available for ML: {len(model_df):,}")

# ============================================================
# 3. DISPLAY CLASS DISTRIBUTION
# ============================================================

print("\n[2] Target class distribution:")

print(
    model_df[target].value_counts()
)

# ============================================================
# 4. ENCODE TARGET
# ============================================================

label_mapping = {
    "Low": 0,
    "Moderate": 1,
    "High": 2,
    "Very High": 3
}

model_df["target_encoded"] = model_df[target].map(
    label_mapping
)

# Remove unknown classes if any
model_df = model_df.dropna(
    subset=["target_encoded"]
)

model_df["target_encoded"] = (
    model_df["target_encoded"].astype(int)
)

X = model_df[features]
y = model_df["target_encoded"]

# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

print("\n[3] Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training records: {len(X_train):,}")
print(f"Testing records : {len(X_test):,}")

# ============================================================
# 6. RANDOM FOREST
# ============================================================

print("\n[4] Training Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(X_train, y_train)

print("Model training completed.")

# ============================================================
# 7. PREDICTION
# ============================================================

print("\n[5] Making predictions...")

y_pred = model.predict(X_test)

# ============================================================
# 8. MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

# ============================================================
# 9. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

class_names = [
    "Low",
    "Moderate",
    "High",
    "Very High"
]

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2, 3],
        target_names=class_names,
        zero_division=0
    )
)

# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

print("Confusion Matrix:")

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1, 2, 3]
)

print(cm)

# ============================================================
# 11. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print(
    importance.to_string(index=False)
)

# Save feature importance
importance.to_csv(
    "feature_importance.csv",
    index=False
)

# ============================================================
# 12. SAVE MODEL
# ============================================================

joblib.dump(
    {
        "model": model,
        "features": features,
        "label_mapping": label_mapping
    },
    MODEL_FILE
)

print("\nModel saved as:")
print(MODEL_FILE)

print("\nFeature importance saved as:")
print("feature_importance.csv")

print("\n" + "=" * 70)
print("RANDOM FOREST TRAINING COMPLETED")
print("=" * 70)