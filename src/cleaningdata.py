
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


US_STATES = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA",
    "HI","IA","ID","IL","IN","KS","KY","LA","ME","MD",
    "MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC",
    "SD","TN","TX","UT","VT","VA","WA","WI","WV","WY"
]


@dataclass(frozen=True)
class PreprocessConfig:
    raw_csv: Path
    out_state_week_csv: Path
    out_model_ready_csv: Path
    keep_only_50_states: bool = True
    missing_strategy: str = "state_median"  
    normalize_percent_columns: bool = True



ID_COLS = ["Week Ending Date", "Geographic aggregation"]

CAPACITY_COLS = [
    "Number of Inpatient Beds",
    "Number of Inpatient Beds Occupied",
    "Number of ICU Beds",
    "Number of ICU Beds Occupied",
]

STRESS_COLS = [
    "Percent Inpatient Beds Occupied",
    "Percent ICU Beds Occupied",
]


DISEASE_COLS = [
    "Total Patients Hospitalized with COVID-19",
    "Total Patients Hospitalized with Influenza",
    "Total Patients Hospitalized with RSV",
    "Total ICU Patients Hospitalized with COVID-19",
    "Total ICU Patients Hospitalized with Influenza",
    "Total ICU Patients Hospitalized with RSV",
]


REPORTING_COLS = [
    "Number Hospitals Reporting Number of Inpatient Beds",
    "Number Hospitals Reporting Number of ICU Beds",
    "Percent Hospitals Reporting Number of Inpatient Beds",
    "Percent Hospitals Reporting Number of ICU Beds",
]

COLS_TO_KEEP = ID_COLS + CAPACITY_COLS + STRESS_COLS + DISEASE_COLS + REPORTING_COLS



def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _normalize_percent_if_needed(series: pd.Series) -> pd.Series:
    """
    If percent columns are stored as proportions (0–1) instead of 0–100,
    convert to percent.
    """
    s = series.dropna()
    if s.empty:
        return series
    if s.max() <= 1.5:
        return series * 100.0
    return series


def _strip_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    CDC exports sometimes include trailing spaces in column names.
    Stripping once avoids KeyError nightmares.
    """
    df.columns = df.columns.astype(str).str.strip()
    return df


def make_clean_state_week(raw_df: pd.DataFrame, cfg: PreprocessConfig) -> pd.DataFrame:

    raw_df = _strip_column_names(raw_df)


    missing = [c for c in COLS_TO_KEEP if c not in raw_df.columns]
    if missing:
        raise KeyError(
            "Missing required columns in raw CSV:\n"
            + "\n".join(f"- {m}" for m in missing)
        )


    df = raw_df[COLS_TO_KEEP].copy()


    df["Week Ending Date"] = pd.to_datetime(df["Week Ending Date"], errors="coerce")
    df = df.dropna(subset=["Week Ending Date", "Geographic aggregation"])


    if cfg.keep_only_50_states:
        df = df[df["Geographic aggregation"].isin(US_STATES)].copy()


    numeric_cols = CAPACITY_COLS + STRESS_COLS + DISEASE_COLS + REPORTING_COLS
    df = _coerce_numeric(df, numeric_cols)


    if cfg.normalize_percent_columns:
        df["Percent Inpatient Beds Occupied"] = _normalize_percent_if_needed(df["Percent Inpatient Beds Occupied"])
        df["Percent ICU Beds Occupied"] = _normalize_percent_if_needed(df["Percent ICU Beds Occupied"])


        if "Percent Hospitals Reporting Number of Inpatient Beds" in df.columns:
            df["Percent Hospitals Reporting Number of Inpatient Beds"] = _normalize_percent_if_needed(
                df["Percent Hospitals Reporting Number of Inpatient Beds"]
            )
        if "Percent Hospitals Reporting Number of ICU Beds" in df.columns:
            df["Percent Hospitals Reporting Number of ICU Beds"] = _normalize_percent_if_needed(
                df["Percent Hospitals Reporting Number of ICU Beds"]
            )


    agg_dict = {
        # capacity counts
        "Number of Inpatient Beds": "sum",
        "Number of Inpatient Beds Occupied": "sum",
        "Number of ICU Beds": "sum",
        "Number of ICU Beds Occupied": "sum",

        # stress % targets
        "Percent Inpatient Beds Occupied": "mean",
        "Percent ICU Beds Occupied": "mean",

        # disease totals
        "Total Patients Hospitalized with COVID-19": "sum",
        "Total Patients Hospitalized with Influenza": "sum",
        "Total Patients Hospitalized with RSV": "sum",
        "Total ICU Patients Hospitalized with COVID-19": "sum",
        "Total ICU Patients Hospitalized with Influenza": "sum",
        "Total ICU Patients Hospitalized with RSV": "sum",

        # reporting
        "Number Hospitals Reporting Number of Inpatient Beds": "sum",
        "Number Hospitals Reporting Number of ICU Beds": "sum",
        "Percent Hospitals Reporting Number of Inpatient Beds": "mean",
        "Percent Hospitals Reporting Number of ICU Beds": "mean",
    }

    state_week = (
        df.groupby(["Geographic aggregation", "Week Ending Date"], as_index=False)
          .agg(agg_dict)
    )


    state_week = state_week.sort_values(["Geographic aggregation", "Week Ending Date"]).reset_index(drop=True)

    all_numeric = [c for c in numeric_cols if c in state_week.columns]

    if cfg.missing_strategy == "drop":
        state_week = state_week.dropna()
    elif cfg.missing_strategy == "ffill":
        state_week[all_numeric] = state_week.groupby("Geographic aggregation")[all_numeric].ffill()
        state_week = state_week.dropna()
    elif cfg.missing_strategy == "state_median":
        for col in all_numeric:
            med = state_week.groupby("Geographic aggregation")[col].transform("median")
            state_week[col] = state_week[col].fillna(med)

        state_week = state_week.dropna(subset=["Percent ICU Beds Occupied", "Percent Inpatient Beds Occupied"])
    else:
        raise ValueError("missing_strategy must be one of: drop, ffill, state_median")

    return state_week


def make_model_ready(state_week: pd.DataFrame) -> pd.DataFrame:
    """
    Create lag features for forecasting:
    - ICU % last week
    - Inpatient % last week
    - disease totals last week (optional but powerful)
    - rolling 4-week averages (smooth trends)
    """
    df = state_week.copy()

    df["Week Ending Date"] = pd.to_datetime(df["Week Ending Date"], errors="coerce")
    df = df.dropna(subset=["Week Ending Date"])
    df = df.sort_values(["Geographic aggregation", "Week Ending Date"]).reset_index(drop=True)

    grp = df.groupby("Geographic aggregation")

    df["icu_pct_last_week"] = grp["Percent ICU Beds Occupied"].shift(1)
    df["inpatient_pct_last_week"] = grp["Percent Inpatient Beds Occupied"].shift(1)

 
    for col in [
        "Total Patients Hospitalized with COVID-19",
        "Total Patients Hospitalized with Influenza",
        "Total Patients Hospitalized with RSV",
        "Total ICU Patients Hospitalized with COVID-19",
        "Total ICU Patients Hospitalized with Influenza",
        "Total ICU Patients Hospitalized with RSV",
    ]:
        if col in df.columns:
            df[f"{col}_last_week"] = grp[col].shift(1)


    df["icu_pct_4w_avg"] = grp["Percent ICU Beds Occupied"].rolling(4).mean().reset_index(level=0, drop=True)
    df["inpatient_pct_4w_avg"] = grp["Percent Inpatient Beds Occupied"].rolling(4).mean().reset_index(level=0, drop=True)

 
    df = df.dropna(subset=["icu_pct_last_week", "inpatient_pct_last_week", "icu_pct_4w_avg", "inpatient_pct_4w_avg"]).reset_index(drop=True)

    return df


def summarize(df_state_week: pd.DataFrame, df_model_ready: pd.DataFrame) -> None:
    print("Preprocess summary")
    print(f"- Rows (state_week): {len(df_state_week):,}")
    print(f"- Rows (model_ready): {len(df_model_ready):,}")
    print(f"- States: {df_state_week['Geographic aggregation'].nunique()}")
    print(
        f"- Date range: {df_state_week['Week Ending Date'].min().date()} → {df_state_week['Week Ending Date'].max().date()}"
    )
    print("- state_week columns:", list(df_state_week.columns))
    print("- model_ready columns:", list(df_model_ready.columns))


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess CDC hospital data into clean state-week + model-ready datasets."
    )
    parser.add_argument("--raw", type=str, default="data/raw/Weekly_HospitaWeekly_Hospital_Respiratory_Data.csv")
    parser.add_argument("--out_state_week", type=str, default="data/cleaned/state_week_50.csv")
    parser.add_argument("--out_model_ready", type=str, default="data/cleaned/model_ready.csv")
    parser.add_argument("--missing", type=str, default="state_median", choices=["drop", "ffill", "state_median"])
    parser.add_argument("--keepall", action="store_true", help="Keep territories/regions too (not recommended).")
    parser.add_argument("--no-normalize", action="store_true", help="Disable auto percent normalization.")
    args = parser.parse_args(argv)

    cfg = PreprocessConfig(
        raw_csv=Path(args.raw),
        out_state_week_csv=Path(args.out_state_week),
        out_model_ready_csv=Path(args.out_model_ready),
        keep_only_50_states=(not args.keepall),
        missing_strategy=args.missing,
        normalize_percent_columns=(not args.no_normalize),
    )


    raw_df = pd.read_csv(cfg.raw_csv)
    raw_df = _strip_column_names(raw_df)


    state_week = make_clean_state_week(raw_df, cfg)
    model_ready = make_model_ready(state_week)


    cfg.out_state_week_csv.parent.mkdir(parents=True, exist_ok=True)
    state_week.to_csv(cfg.out_state_week_csv, index=False)
    model_ready.to_csv(cfg.out_model_ready_csv, index=False)

    summarize(state_week, model_ready)

    print(f" Saved state_week: {cfg.out_state_week_csv.as_posix()}")
    print(f" Saved model_ready: {cfg.out_model_ready_csv.as_posix()}")


if __name__ == "__main__":
    main()
