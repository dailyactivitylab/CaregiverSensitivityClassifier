#! /usr/bin/env python

import os
import joblib
import numpy as np
import pandas as pd

# This file predicts caregiver sensitivity (High vs Low) for each distress episode.
# It combines the frame-level, event-level and state-level features (Steps 2-4) and applies the
# trained random forest saved by train_model.py.

FEATURE_DIR = "../Features"
MODEL_PATH = "../Model/Sensitivity_RF.joblib"
OUTPUT_CSV = "../Results/Predictions.csv"

ID_COL = "Dist_Ep_IDs"

# Not used by the model (dropped in the final model, as in the research code)
UNUSED_FEATURES = ["Episode_dur", "First_entry_resp"]

# A missing latency means the caregiver never responded, which is treated as a very late response:
# it is filled with 3 x the largest latency seen in the training data (stored in the model file).
# Only the latency columns that exist in the training data are used.
LATENCY_COLS = ["Caregiver_first_begin", "Overall_latency_of_response", "Latency_of_response_median"]


def load_features(feature_dir=FEATURE_DIR):
    """Merge the event, state and frame features in feature_dir into one table with one row per distress episode."""
    event = pd.read_csv(os.path.join(feature_dir, "Event_level_features.csv"))
    state = pd.read_csv(os.path.join(feature_dir, "State_level_features.csv")).drop(columns=UNUSED_FEATURES, errors="ignore")
    frame = pd.read_csv(os.path.join(feature_dir, "Frame_level_features.csv"))

    features = event
    for name, other in [("state", state), ("frame", frame)]:
        missing = set(features[ID_COL]) - set(other[ID_COL])
        if missing:
            print(f"Warning: {len(missing)} episode(s) have no {name} features and are excluded: {sorted(missing)}")
        features = features.merge(other, on=ID_COL, how="inner")
    return features


def preprocess(features, latency_fill):
    """Fill missing values: latencies with the 'very late' value, everything else with -1."""
    features = features.copy()
    for col, value in latency_fill.items():
        if col in features.columns:
            features[col] = features[col].fillna(value)
    return features.fillna(-1)


def predict_from_bundle(features, bundle):
    """Predict sensitivity for a feature table (with a Dist_Ep_IDs column) using a saved model bundle."""
    features = preprocess(features, bundle["latency_fill"])

    # Use exactly the features (and order) the model was trained on
    missing_cols = [c for c in bundle["feature_columns"] if c not in features.columns]
    if missing_cols:
        raise ValueError(f"Features the model needs are missing: {missing_cols}")
    X = features[bundle["feature_columns"]].to_numpy()

    model = bundle["model"]
    output = pd.DataFrame({ID_COL: features[ID_COL].to_numpy(),
                           "Predicted_sensitivity": model.predict(X)})
    probabilities = model.predict_proba(X)
    for i, label in enumerate(model.classes_):
        output[f"Prob_{label}"] = probabilities[:, i]
    return output


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No trained model found at {MODEL_PATH}. Run train_model.py first.")
    bundle = joblib.load(MODEL_PATH)

    output = predict_from_bundle(load_features(), bundle)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    output.to_csv(OUTPUT_CSV, index=False)
    print(output["Predicted_sensitivity"].value_counts())


if __name__ == "__main__":
    main()
