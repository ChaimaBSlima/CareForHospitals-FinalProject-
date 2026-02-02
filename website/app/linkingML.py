from __future__ import annotations

from pathlib import Path
import pandas as pd

FORECAST_PATH = Path(__file__).resolve().parents[2] / "data" / "cleaned" / "next_week_forecast_enhanced.csv"


REQUIRED_COLS = [
    "state",
    "current_week",
    "forecast_week",
    "icu_pct_next_week_pred",
    "inpatient_pct_next_week_pred",
    "critical_risk_proba",
    "critical_risk_next_week_pred",
    "disease_burden_next_week_pred",
    "suggested_neighbor_state",
    "recommendation",
]

US_STATE_NAMES = {
    "AL": "Alabama","AK": "Alaska","AZ": "Arizona","AR": "Arkansas","CA": "California",
    "CO": "Colorado","CT": "Connecticut","DE": "Delaware","FL": "Florida","GA": "Georgia",
    "HI": "Hawaii","IA": "Iowa","ID": "Idaho","IL": "Illinois","IN": "Indiana",
    "KS": "Kansas","KY": "Kentucky","LA": "Louisiana","ME": "Maine","MD": "Maryland",
    "MA": "Massachusetts","MI": "Michigan","MN": "Minnesota","MO": "Missouri","MS": "Mississippi",
    "MT": "Montana","NE": "Nebraska","NV": "Nevada","NH": "New Hampshire","NJ": "New Jersey",
    "NM": "New Mexico","NY": "New York","NC": "North Carolina","ND": "North Dakota","OH": "Ohio",
    "OK": "Oklahoma","OR": "Oregon","PA": "Pennsylvania","RI": "Rhode Island","SC": "South Carolina",
    "SD": "South Dakota","TN": "Tennessee","TX": "Texas","UT": "Utah","VT": "Vermont",
    "VA": "Virginia","WA": "Washington","WI": "Wisconsin","WV": "West Virginia","WY": "Wyoming",
}

def state_label(code: str) -> str:
    # exactly: NY — New York
    return f"{code} — {US_STATE_NAMES.get(code, code)}"


def load_forecast(path: Path = FORECAST_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Forecast file not found: {path.as_posix()}\n"
            "Run: python src/predict_next_week.py"
        )

    df = pd.read_csv(path)

    # Standardize column names just in case
    df.columns = [c.strip() for c in df.columns]

    # Basic validation (don’t hard-fail if missing optional cols)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        # keep going, but signal in UI later if needed
        df.attrs["missing_cols"] = missing
    else:
        df.attrs["missing_cols"] = []

    # Parse dates if present
    for c in ["current_week", "forecast_week"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # Sort by risk if present
    if "critical_risk_proba" in df.columns:
        df = df.sort_values("critical_risk_proba", ascending=False)

    return df


def get_state_row(df: pd.DataFrame, state: str):
    sub = df[df["state"] == state]
    if sub.empty:
        return None
    return sub.iloc[0]


def fmt_pct(x):
    try:
        return f"{float(x):.1f}%"
    except Exception:
        return "N/A"


def fmt_num(x):
    try:
        return f"{float(x):.0f}"
    except Exception:
        return "N/A"


def fmt_proba(x):
    try:
        return f"{float(x):.2f}"
    except Exception:
        return "N/A"

