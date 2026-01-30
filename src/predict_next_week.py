from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd



# Full 50-state neighbors map (land borders)
# Alaska/Hawaii have no land neighbors

NEIGHBORS = {
    "AL": ["FL", "GA", "MS", "TN"],
    "AK": [],
    "AZ": ["CA", "CO", "NM", "NV", "UT"],
    "AR": ["LA", "MO", "MS", "OK", "TN", "TX"],
    "CA": ["AZ", "NV", "OR"],
    "CO": ["AZ", "KS", "NE", "NM", "OK", "UT", "WY"],
    "CT": ["MA", "NY", "RI"],
    "DE": ["MD", "NJ", "PA"],
    "FL": ["AL", "GA"],
    "GA": ["AL", "FL", "NC", "SC", "TN"],
    "HI": [],
    "ID": ["MT", "NV", "OR", "UT", "WA", "WY"],
    "IL": ["IN", "IA", "KY", "MO", "WI", "MI"],
    "IN": ["IL", "KY", "MI", "OH"],
    "IA": ["IL", "MN", "MO", "NE", "SD", "WI"],
    "KS": ["CO", "MO", "NE", "OK"],
    "KY": ["IL", "IN", "MO", "OH", "TN", "VA", "WV"],
    "LA": ["AR", "MS", "TX"],
    "ME": ["NH"],
    "MD": ["DE", "PA", "VA", "WV"],
    "MA": ["CT", "NH", "NY", "RI", "VT"],
    "MI": ["IN", "OH", "WI", "IL"],
    "MN": ["IA", "ND", "SD", "WI"],
    "MS": ["AL", "AR", "LA", "TN"],
    "MO": ["AR", "IA", "IL", "KS", "KY", "NE", "OK", "TN"],
    "MT": ["ID", "ND", "SD", "WY"],
    "NE": ["CO", "IA", "KS", "MO", "SD", "WY"],
    "NV": ["AZ", "CA", "ID", "OR", "UT"],
    "NH": ["MA", "ME", "VT"],
    "NJ": ["DE", "NY", "PA"],
    "NM": ["AZ", "CO", "OK", "TX", "UT"],
    "NY": ["CT", "MA", "NJ", "PA", "VT"],
    "NC": ["GA", "SC", "TN", "VA"],
    "ND": ["MN", "MT", "SD"],
    "OH": ["IN", "KY", "MI", "PA", "WV"],
    "OK": ["AR", "CO", "KS", "MO", "NM", "TX"],
    "OR": ["CA", "ID", "NV", "WA"],
    "PA": ["DE", "MD", "NJ", "NY", "OH", "WV"],
    "RI": ["CT", "MA"],
    "SC": ["GA", "NC"],
    "SD": ["IA", "MN", "MT", "ND", "NE", "WY"],
    "TN": ["AL", "AR", "GA", "KY", "MO", "MS", "NC", "VA"],
    "TX": ["AR", "LA", "NM", "OK"],
    "UT": ["AZ", "CO", "ID", "NM", "NV", "WY"],
    "VT": ["MA", "NH", "NY"],
    "VA": ["KY", "MD", "NC", "TN", "WV"],
    "WA": ["ID", "OR"],
    "WV": ["KY", "MD", "OH", "PA", "VA"],
    "WI": ["IA", "IL", "MI", "MN"],
    "WY": ["CO", "ID", "MT", "NE", "SD", "UT"],
}


def recommend_action(row: pd.Series) -> str:
    """
    Portfolio-friendly operational recommendation text.
    Not clinical guidance. This is decision-support language based on forecast signals.
    """
    icu = float(row["icu_pct_next_week_pred"])
    inp = float(row["inpatient_pct_next_week_pred"])
    risk = int(row["critical_risk_next_week_pred"])
    proba = float(row["critical_risk_proba"])
    neighbor = str(row.get("suggested_neighbor_state", "") or "")

    if risk == 1 or (icu >= 85 and inp >= 85):
        msg = (
            "HIGH RISK: Increase surge monitoring, review staffing/bed capacity plans, "
            "and coordinate regionally for potential load balancing."
        )
        if neighbor:
            msg += f" Potential lower-risk neighbor: {neighbor}."
        return msg

    if (icu >= 80) or (inp >= 85) or (proba >= 0.12):
        msg = "MODERATE: Monitor closely and prepare contingency plans."
        if neighbor:
            msg += f" Nearby alternative option: {neighbor}."
        return msg

    return "LOW: Normal monitoring."


def suggest_neighbor(state: str, lookup: pd.DataFrame) -> str:
    """
    Suggest a neighboring state with lower predicted risk probability and lower ICU/inpatient values.
    """
    if state not in NEIGHBORS:
        return ""
    candidates = []
    for nb in NEIGHBORS[state]:
        if nb in lookup.index:
            candidates.append(
                (
                    nb,
                    float(lookup.loc[nb, "critical_risk_proba"]),
                    float(lookup.loc[nb, "icu_pct_next_week_pred"]),
                    float(lookup.loc[nb, "inpatient_pct_next_week_pred"]),
                )
            )
    if not candidates:
        return ""
    # Choose best neighbor: lowest risk proba, then lowest ICU, then lowest inpatient
    best = sorted(candidates, key=lambda x: (x[1], x[2], x[3]))[0][0]
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate next-week forecasts per state using saved models.")
    parser.add_argument("--data", type=str, default="data/cleaned/model_ready.csv")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--out_dir", type=str, default="data/cleaned")
    args = parser.parse_args()

    data_path = Path(args.data)
    models_dir = Path(args.models_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


    # Load models + config

    model_icu = joblib.load(models_dir / "model_icu.joblib")
    model_inpatient = joblib.load(models_dir / "model_inpatient.joblib")
    model_critical = joblib.load(models_dir / "model_critical.joblib")
    model_disease = joblib.load(models_dir / "model_disease.joblib")

    feature_cols = joblib.load(models_dir / "feature_cols.joblib")
    meta = joblib.load(models_dir / "meta.joblib")
    critical_threshold = float(meta.get("critical_threshold", 0.5))


    # Load data

    df = pd.read_csv(data_path)
    df["Week Ending Date"] = pd.to_datetime(df["Week Ending Date"])

    # Current and forecast weeks
    current_week = df["Week Ending Date"].max()
    forecast_week = current_week + pd.Timedelta(days=7)


    # Latest row per state

    latest_per_state = (
        df.sort_values(["Geographic aggregation", "Week Ending Date"])
        .groupby("Geographic aggregation")
        .tail(1)
        .copy()
    )

    # sanity: ensure all features exist
    missing_features = [c for c in feature_cols if c not in latest_per_state.columns]
    if missing_features:
        raise KeyError(f"Missing required feature columns in data: {missing_features}")

    X_forecast = latest_per_state[feature_cols].copy()


    # Predict

    icu_pred = model_icu.predict(X_forecast)
    inpatient_pred = model_inpatient.predict(X_forecast)

    risk_proba = model_critical.predict_proba(X_forecast)[:, 1]
    risk_pred = (risk_proba >= critical_threshold).astype(int)

    disease_pred = model_disease.predict(X_forecast)

    forecast = pd.DataFrame(
        {
            "state": latest_per_state["Geographic aggregation"].values,
            "current_week": current_week,
            "forecast_week": forecast_week,
            "icu_pct_next_week_pred": icu_pred,
            "inpatient_pct_next_week_pred": inpatient_pred,
            "critical_risk_proba": risk_proba,
            "critical_risk_next_week_pred": risk_pred,
            "disease_burden_next_week_pred": disease_pred,
        }
    )


    # Sort (critical first)

    forecast_sorted = forecast.sort_values(
        by=["critical_risk_next_week_pred", "critical_risk_proba", "icu_pct_next_week_pred"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    # Neighbor suggestion requires lookup
    lookup = forecast_sorted.set_index("state")
    forecast_sorted["suggested_neighbor_state"] = forecast_sorted["state"].apply(lambda s: suggest_neighbor(s, lookup))

    # Recommendation text
    forecast_sorted["recommendation"] = forecast_sorted.apply(recommend_action, axis=1)


    # Save outputs

    out_all = out_dir / "next_week_forecast_enhanced.csv"
    out_critical = out_dir / "next_week_forecast_critical_only_enhanced.csv"

    forecast_sorted.to_csv(out_all, index=False)
    forecast_sorted[forecast_sorted["critical_risk_next_week_pred"] == 1].to_csv(out_critical, index=False)

    print(" Current week:", current_week.date())
    print(" Forecast week:", forecast_week.date())
    print(" Saved:", out_all.as_posix())
    print(" Saved:", out_critical.as_posix())
    print("Critical states:", int((forecast_sorted["critical_risk_next_week_pred"] == 1).sum()))


if __name__ == "__main__":
    main()
