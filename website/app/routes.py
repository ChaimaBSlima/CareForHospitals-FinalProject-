from __future__ import annotations

from flask import Blueprint, render_template, request, redirect, url_for
from .linkingML import load_forecast, get_state_row, fmt_pct, fmt_num, fmt_proba, state_label

bp = Blueprint("main", __name__)

@bp.route("/", methods=["GET"])
def index():
    df = load_forecast()

    # dropdown values
    states = sorted(df["state"].dropna().unique().tolist())
    state_options = [(s, state_label(s)) for s in states]

    # if user selected from dropdown (GET ?state=TX), go to state page
    selected = request.args.get("state")
    if selected:
        return redirect(url_for("main.state_page", state=selected))

    return render_template(
        "index.html",
        state_options=state_options,
        missing_cols=df.attrs.get("missing_cols", []),
    )

    


@bp.route("/state/<state>", methods=["GET"])
def state_page(state: str):
    df = load_forecast()
    row = get_state_row(df, state)
    if row is None:
        return render_template("state.html", not_found=True, state=state)

    neighbor = row.get("suggested_neighbor_state", None)
    nrow = get_state_row(df, neighbor) if neighbor else None


    data = {
        "state": state_label(state),
        "current_week": row.get("current_week"),
        "forecast_week": row.get("forecast_week"),
        "icu": fmt_pct(row.get("icu_pct_next_week_pred")),
        "inpatient": fmt_pct(row.get("inpatient_pct_next_week_pred")),
        "risk_proba": fmt_proba(row.get("critical_risk_proba")),
        "risk_pred": int(row.get("critical_risk_next_week_pred", 0)),
        "disease": fmt_num(row.get("disease_burden_next_week_pred")),
        "recommendation": row.get("recommendation", ""),
        "neighbor": state_label(neighbor) if neighbor else None,
    }

    neighbor_data = None
    if nrow is not None:
        neighbor_data = {
            "state": state_label(neighbor),
            "icu": fmt_pct(nrow.get("icu_pct_next_week_pred")),
            "inpatient": fmt_pct(nrow.get("inpatient_pct_next_week_pred")),
            "risk_proba": fmt_proba(nrow.get("critical_risk_proba")),
            "risk_pred": int(nrow.get("critical_risk_next_week_pred", 0)),
            "disease": fmt_num(nrow.get("disease_burden_next_week_pred")),
        }

    return render_template(
        "state.html",
        data=data,
        neighbor_data=neighbor_data,
        missing_cols=df.attrs.get("missing_cols", []),
    )


@bp.route("/top-risk", methods=["GET"])
def top_risk():
    df = load_forecast()

    n = request.args.get("n", "15")
    try:
        n = max(5, min(50, int(n)))
    except Exception:
        n = 15


    top_risks = (
        df[df["critical_risk_next_week_pred"] == 1]
          .sort_values("critical_risk_proba", ascending=False)
          .head(n)
          .copy()
    )


    top_risks["state_label"] = top_risks["state"].apply(state_label)

    if "suggested_neighbor_state" in top_risks.columns:
        top_risks["neighbor_label"] = top_risks["suggested_neighbor_state"].apply(state_label)


    if "icu_pct_next_week_pred" in top_risks.columns:
        top_risks["icu_pct_next_week_pred"] = top_risks["icu_pct_next_week_pred"].map(lambda x: f"{x:.1f}%")
    if "inpatient_pct_next_week_pred" in top_risks.columns:
        top_risks["inpatient_pct_next_week_pred"] = top_risks["inpatient_pct_next_week_pred"].map(lambda x: f"{x:.1f}%")
    if "critical_risk_proba" in top_risks.columns:
        top_risks["critical_risk_proba"] = top_risks["critical_risk_proba"].map(lambda x: f"{x:.2f}")

    top_risks_rows = top_risks.to_dict(orient="records")

    return render_template(
        "top_risk.html",
        top_risks=top_risks_rows,
        missing_cols=df.attrs.get("missing_cols", []),
    )

