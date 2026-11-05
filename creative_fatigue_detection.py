import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path

# CONFIG THE CSV FILES

DATA_DIR = Path("data")

FILES = {
    "users":     DATA_DIR / "users.csv",
    "campaigns": DATA_DIR / "campaigns.csv",
    "ads":       DATA_DIR / "ads.csv",
    "events":    DATA_DIR / "ad_events.csv",
}

COLUMNS = {
    "ad_id_in_ads":      "ad_id",        # ads.csv: unique ad id
    "ad_id_in_events":   "ad_id",        # ad_events.csv: FK to ads table
    "event_type":        "event_type",   # ad_events.csv: e.g. "Impression" / "Click"
    "event_timestamp":   "timestamp",    # ad_events.csv: when the event happened
    "impression_label":  "Impression",   # event_type signifying was shown to user
    "click_label":       "Click",        # event_type signifying the user clicked on the ad
    "ad_launch_date":     None,          # ads.csv: creative start date (None = derive from first event)
}

# LOAD DATA

def load_data():
    ads = pd.read_csv(FILES["ads"])
    events = pd.read_csv(FILES["events"])
    return ads, events


def build_daily_ad_timeseries(ads: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:

    c = COLUMNS
    events = events.copy()
    events[c["event_timestamp"]] = pd.to_datetime(events[c["event_timestamp"]])
    events["event_date"] = events[c["event_timestamp"]].dt.date

    # Flag each row as an impression or a click (1/0), then just sum per group
    events["is_impression"] = (events[c["event_type"]] == c["impression_label"]).astype(int)
    events["is_click"] = (events[c["event_type"]] == c["click_label"]).astype(int)

    # each row contains (ad, day) with a concise view of how many impressions, clicks
    daily = (
        events.groupby([c["ad_id_in_events"], "event_date"])[["is_impression", "is_click"]]
        .sum()
        .reset_index()
        .rename(columns={
            c["ad_id_in_events"]: "ad_id",
            "event_date": "date",
            "is_impression": "impressions",
            "is_click": "clicks",
        })
    )

    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values(["ad_id", "date"])

    # when impressions are more than 0, calculate the ctr or else replace with nan value
    daily["ctr"] = np.where(daily["impressions"] > 0,
                             daily["clicks"] / daily["impressions"],
                             np.nan)

    # day-of-life for each ad (0 = launch day)
    daily["day_of_life"] = daily.groupby("ad_id")["date"].transform(
        lambda x: (x - x.min()).dt.days
    )
    return daily


# MAKING THE FATIGUE MODEL THAT DETECTS ANOMALIES

def exp_decay(t, a, b, c):
    # exponential decay curve where the ctr starts high and then decays to the floor (ad fatigue curve)

    # a = initial quantity
    # b = exponential decay constant
    # t = time
    return a * np.exp(-b * t) + c


def fit_and_flag(ad_df: pd.DataFrame, z_thresh: float = 2.0, min_days: int = 7):
    """
    Fits an exponential decay curve to one ad's CTR-over-time.
    Flags days where actual CTR is anomalously BELOW the fitted curve
    (i.e., decaying faster than the ad's own expected pattern) using
    a rolling z-score on the residuals.

    Returns: (fitted_params, dataframe with predicted + residual + flag columns)
    """
    df = ad_df.dropna(subset=["ctr"]).copy()

    # initializing the columns
    df["predicted_ctr"] = np.nan
    df["residual"] = np.nan
    df["z_score"] = np.nan
    df["fatigue_anomaly"] = False

    if len(df) < min_days:
        return None, df 

    t = df["day_of_life"].values.astype(float)
    y = df["ctr"].values.astype(float)

    try:
        # initial guesses: a=starting CTR, b=decay rate, c=floor CTR
        p0 = [max(y[0], 0.01), 0.05, max(y.min(), 0.001)]
        params, _ = curve_fit(exp_decay, t, y, p0=p0, maxfev=5000) # curve_fit fits a function to data
    except RuntimeError:
        return None, df  # curve didn't converge — flag ad for manual review

    df["predicted_ctr"] = exp_decay(t, *params)
    df["residual"] = df["ctr"] - df["predicted_ctr"]

    resid_std = df["residual"].std()
    resid_std = resid_std if resid_std > 1e-6 else 1e-6
    df["z_score"] = df["residual"] / resid_std

    # anomaly = actual CTR is falling well BELOW the expected curve
    df["fatigue_anomaly"] = df["z_score"] < -z_thresh

    return params, df


def run_fatigue_detection(daily: pd.DataFrame, z_thresh: float = 2.0, verbose: bool = True):
    # runs fit_and_flag for every ad, returns one combined dataframe
    results = []
    n_success, n_too_short, n_no_converge = 0, 0, 0

    for ad_id, ad_df in daily.groupby("ad_id"):
        n_days = ad_df["ctr"].dropna().shape[0]
        params, scored = fit_and_flag(ad_df, z_thresh=z_thresh)
        scored["ad_id"] = ad_id
        scored["curve_fit_success"] = params is not None
        results.append(scored)

        if params is not None:
            n_success += 1
        elif n_days < 7:
            n_too_short += 1
        else:
            n_no_converge += 1

    if verbose:
        print(f"\n[Diagnostic] {n_success} ads fitted successfully, "
              f"{n_too_short} had too few days (<7), "
              f"{n_no_converge} failed to converge.")
        if n_success == 0:
            print("[Diagnostic] No ad had a successful curve fit — "
                  "fatigue_anomaly will be False for everything. "
                  "Check: (1) are impressions/clicks being counted correctly? "
                  "print(daily.head(20)) to inspect. "
                  "(2) try lowering min_days in fit_and_flag if your ads run "
                  "for fewer than 7 days each.")

    return pd.concat(results, ignore_index=True)


# REPORTING: SUMMARY TABLE + PLOTS

def summarize_flags(scored: pd.DataFrame) -> pd.DataFrame:
    """One row per ad: total anomaly days, first anomaly day, status."""
    scored = scored.copy()
    scored["fatigue_anomaly"] = scored["fatigue_anomaly"].fillna(False)

    # how many days each ad was run and how many anomalies did it cause
    summary = (
        scored.groupby("ad_id")
        .agg(total_days=("day_of_life", "count"),
             anomaly_days=("fatigue_anomaly", "sum"))
        .reset_index()
    )

    # First day each ad was flagged (ads with no flagged days just won't appear here, so we left-merge and let those be NaN)
    first_anomaly = (
        scored[scored["fatigue_anomaly"]]
        .groupby("ad_id")["day_of_life"]
        .min()
        .rename("first_anomaly_day")
        .reset_index()
    )
    summary = summary.merge(first_anomaly, on="ad_id", how="left")

    # wherever the anomalies are more than 2, flag the ad
    summary["status"] = np.where(
        summary["anomaly_days"] >= 2, "⚠ Refresh recommended",
        np.where(summary["anomaly_days"] == 1, "Monitor", "Healthy")
    )
    return summary.sort_values("anomaly_days", ascending=False)


def plot_ad_fatigue(scored: pd.DataFrame, ad_id, save_path=None):
    """Actual vs expected CTR curve, with anomaly days marked in red."""
    df = scored[scored["ad_id"] == ad_id].sort_values("day_of_life")
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(df["day_of_life"], df["ctr"], "o-", label="Actual CTR", color="#2563eb")
    if "predicted_ctr" in df.columns:
        ax.plot(df["day_of_life"], df["predicted_ctr"], "--",
                 label="Expected decay curve", color="#94a3b8")

    flagged = df[df.get("fatigue_anomaly", False) == True]
    ax.scatter(flagged["day_of_life"], flagged["ctr"], color="red", zorder=5,
                label="Fatigue anomaly", s=80)

    ax.set_xlabel("Day since launch")
    ax.set_ylabel("CTR")
    ax.set_title(f"Creative Fatigue Detection — Ad {ad_id}")
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig

# MAIN

if __name__ == "__main__":
    ads, events = load_data()
    daily = build_daily_ad_timeseries(ads, events)
    scored = run_fatigue_detection(daily, z_thresh=2.0)

    summary = summarize_flags(scored)
    print("\n=== FATIGUE DETECTION SUMMARY (top 10 most fatigued ads) ===")
    print(summary.head(10).to_string(index=False))

    # Save outputs
    scored.to_csv("results/scored_ad_timeseries.csv", index=False)
    summary.to_csv("results/fatigue_summary.csv", index=False)

    # Plot the single most-fatigued ad as an example for your slides
    if len(summary) > 0 and summary.iloc[0]["anomaly_days"] > 0:
        top_ad = summary.iloc[0]["ad_id"]
        plot_ad_fatigue(scored, top_ad, save_path="example_fatigue_chart.png")
        print(f"\nSaved example chart for ad {top_ad} -> example_fatigue_chart.png")

    print("\nDone. Full results in scored_ad_timeseries.csv and fatigue_summary.csv")