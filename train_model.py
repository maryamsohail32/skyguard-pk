"""
SkyGuard PK - Model Training Script
Trains Random Forest classifiers for flood risk and heatwave risk,
and reports honest precision/recall metrics (no inflated claims).
"""

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

FEATURES = ["precipitation", "temperature", "humidity", "soil_wetness", "rain_7day"]


def train_and_evaluate(df, label_col, model_name):
    print(f"\n{'='*50}")
    print(f"Training model for: {label_col}")
    print(f"{'='*50}")

    X = df[FEATURES]
    y = df[label_col]

    # skip training if a class has too few examples for a meaningful split
    class_counts = y.value_counts()
    print(f"Class distribution:\n{class_counts}\n")

    if len(class_counts) < 2:
        print(f"WARNING: only one class present in {label_col} in this dataset. "
              f"Skipping training - real NASA data should have more variety.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        class_weight="balanced",  # important - our classes are imbalanced (few HIGH days)
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Classification Report (HONEST numbers, not inflated):")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Feature importance (what the model relies on most):")
    importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print(importance)

    joblib.dump(model, f"data/{model_name}.pkl")
    print(f"\nModel saved to data/{model_name}.pkl")

    return model


def main():
    df = pd.read_csv("data/labeled_weather_data.csv")

    train_and_evaluate(df, "flood_risk", "flood_model")
    train_and_evaluate(df, "heatwave_risk", "heatwave_model")


if __name__ == "__main__":
    main()
