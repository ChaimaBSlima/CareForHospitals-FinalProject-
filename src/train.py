from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression


# Paths

DATA_PATH = Path("data/cleaned/model_ready.csv")
MODELS_DIR = Path("models")

MODELS_DIR.mkdir(parents=True, exist_ok=True)


# Config

RANDOM_STATE = 42
CRITICAL_THRESHOLD = 0.17307460986945283   # tuned threshold from notebook


# Features (MUST match notebook)

FEATURE_COLS = [
    # Capacity & stress
    "Percent ICU Beds Occupied",
    "Percent Inpatient Beds Occupied",

    # Lag features
    "icu_pct_last_week",
    "inpatient_pct_last_week",
    "icu_pct_4w_avg",
    "inpatient_pct_4w_avg",

    # Disease burden
    "Total Patients Hospitalized with COVID-19",
    "Total Patients Hospitalized with Influenza",
    "Total Patients Hospitalized with RSV",

    # ICU disease burden
    "Total ICU Patients Hospitalized with COVID-19",
    "Total ICU Patients Hospitalized with Influenza",
    "Total ICU Patients Hospitalized with RSV",

    # Reporting strength
    "Number Hospitals Reporting Number of ICU Beds",
    "Number Hospitals Reporting Number of Inpatient Beds",
    "Percent Hospitals Reporting Number of ICU Beds",
    "Percent Hospitals Reporting Number of Inpatient Beds",
]


# Training Pipeline

def main():

    print(" Loading data...")
    df = pd.read_csv(DATA_PATH)
    df["Week Ending Date"] = pd.to_datetime(df["Week Ending Date"])

    # Build targets

    print(" Building targets...")

    df = df.sort_values(["Geographic aggregation", "Week Ending Date"]).copy()

    # --- ICU target ---
    df["icu_pct_next_week"] = (
        df.groupby("Geographic aggregation")["Percent ICU Beds Occupied"].shift(-1)
    )

    # --- Inpatient target ---
    df["inpatient_pct_next_week"] = (
        df.groupby("Geographic aggregation")["Percent Inpatient Beds Occupied"].shift(-1)
    )

    # --- Disease burden ---
    df["disease_burden"] = (
        df["Total Patients Hospitalized with COVID-19"] +
        df["Total Patients Hospitalized with Influenza"] +
        df["Total Patients Hospitalized with RSV"]
    )

    df["disease_burden_next_week"] = (
        df.groupby("Geographic aggregation")["disease_burden"].shift(-1)
    )

    # --- Critical stress label ---
    # Rule: ICU >= 85% OR Inpatient >= 85%
    df["critical_stress_next_week"] = (
        (df["icu_pct_next_week"] >= 85) |
        (df["inpatient_pct_next_week"] >= 85)
    ).astype(int)


    # Drop rows without targets
 
    df = df.dropna(subset=[
        "icu_pct_next_week",
        "inpatient_pct_next_week",
        "disease_burden_next_week"
    ]).copy()

  
    # Train/Test split (time-based)

    split_date = df["Week Ending Date"].quantile(0.8)

    train = df[df["Week Ending Date"] <= split_date].copy()

    X_train = train[FEATURE_COLS]

    y_icu = train["icu_pct_next_week"]
    y_inpatient = train["inpatient_pct_next_week"]
    y_disease = train["disease_burden_next_week"]
    y_critical = train["critical_stress_next_week"]

    print(" Training rows:", len(train))

   
    # Models
   
    print(" Training models...")

    # ICU model
    model_icu = RandomForestRegressor(
        n_estimators=300,
        max_depth=14,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model_icu.fit(X_train, y_icu)

    # Inpatient model
    model_inpatient = RandomForestRegressor(
        n_estimators=300,
        max_depth=14,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model_inpatient.fit(X_train, y_inpatient)

    # Critical risk model
    model_critical = LogisticRegression(max_iter=2000)
    model_critical.fit(X_train, y_critical)

    # Disease burden model
    model_disease = LinearRegression()
    model_disease.fit(X_train, y_disease)

    # Save models
    print(" Saving models...")

    joblib.dump(model_icu, MODELS_DIR / "model_icu.joblib")
    joblib.dump(model_inpatient, MODELS_DIR / "model_inpatient.joblib")
    joblib.dump(model_critical, MODELS_DIR / "model_critical.joblib")
    joblib.dump(model_disease, MODELS_DIR / "model_disease.joblib")

    joblib.dump(FEATURE_COLS, MODELS_DIR / "feature_cols.joblib")

    metadata = {
        "random_state": RANDOM_STATE,
        "critical_threshold": CRITICAL_THRESHOLD,
        "split_date": str(split_date),
        "features": FEATURE_COLS
    }

    joblib.dump(metadata, MODELS_DIR / "meta.joblib")

    print(" Training complete.")
    print(" Models saved in:", MODELS_DIR.resolve())


if __name__ == "__main__":
    main()
