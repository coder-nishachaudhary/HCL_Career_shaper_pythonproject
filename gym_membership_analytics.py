"""
Gym & Fitness Center Membership Analytics
==========================================
Implements the Software Requirements Specification (SRS) for the
Gym & Fitness Center Membership Analytics project (ABES Engineering College).

Input   : 12_gym_membership.csv  (same folder as this script, or current folder)
Output  : gym_analytics_output/
              cleaned_gym_membership.csv      (FR-2.6)
              duplicate_checkins.csv          (FR-1.5)
              iqr_outliers.csv                (FR-4.4)
              analysis_tables.xlsx            (FR-5.2)
              gym_analytics_report.md         (FR-5.3)
              visualizations/*.png            (FR-5.1)

Requires: Python 3.10+, pandas, numpy, matplotlib, seaborn, openpyxl
Run     : python gym_membership_analytics.py [optional_path_to_csv]

Business rules implemented (SRS section 6.5):
  BR-01 duplicate Checkin_ID -> reported, first occurrence kept
  BR-02 missing/invalid duration -> median of valid durations
  BR-03 missing Fitness_Class -> "Not Recorded", excluded from class ranking
  BR-04 time analysis uses only successfully parsed timestamps
  BR-05 low engagement = fewer than 2 visits by a member in a calendar month
  BR-06 estimated revenue = distinct observed members per month & tier x Monthly_Fee
  BR-07 IQR fences = Q1 - 1.5*IQR and Q3 + 1.5*IQR (outliers flagged, not deleted)
  BR-08 Short < 45 | Moderate 45-90 inclusive | Extended > 90
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save charts to files; no display window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ==============================================================================
# CONFIGURATION (documented thresholds - NFR 5.5)
# ==============================================================================
DATA_FILE = "12_gym_membership.csv"
OUTPUT_DIR_NAME = "gym_analytics_output"
REQUIRED_COLUMNS = [
    "Checkin_ID", "Member_ID", "Membership_Tier", "Gender",
    "Checkin_Time", "Workout_Duration_Mins", "Fitness_Class", "Monthly_Fee",
]
LOW_ENGAGEMENT_THRESHOLD = 2        # visits per member per calendar month (BR-05)
SHORT_LIMIT, EXTENDED_LIMIT = 45, 90  # minutes (BR-08)
IQR_MULTIPLIER = 1.5                # BR-07
NOT_RECORDED = "Not Recorded"       # BR-03
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]
TIER_ORDER = ["Basic", "Gold", "Platinum"]

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11})

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)


def banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def fail(message):
    print(f"\nERROR: {message}")
    sys.exit(1)


# ==============================================================================
# MODULE 1 - DATA INGESTION AND QUALITY ASSESSMENT
# ==============================================================================
def locate_csv():
    """Find the CSV: command-line argument, script folder, then current folder."""
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    try:
        script_dir = Path(__file__).resolve().parent
    except NameError:  # running inside a notebook
        script_dir = Path.cwd()
    for folder in (script_dir, Path.cwd()):
        candidate = folder / DATA_FILE
        if candidate.exists():
            return candidate
    return script_dir / DATA_FILE


def load_data(path):
    """FR-1.1 Load dataset; clear error if absent or unreadable."""
    if not path.exists():
        fail(f"Dataset not found: {path}\n"
             f"       Place '{DATA_FILE}' in the same folder as this script.")
    try:
        return pd.read_csv(path)
    except Exception as exc:  # unreadable / malformed file
        fail(f"Could not read '{path}': {exc}")


def validate_schema(df):
    """FR-1.2 Check all eight required columns."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        fail(f"Missing required column(s): {', '.join(missing)}")
    print("Schema validation passed: all 8 required columns are present.")


def inspect_data(df):
    """FR-1.3 Dimensions, sample records, data types, descriptive summary."""
    print(f"\n[A] Dataset dimensions: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\n[B] Data types:")
    print(df.dtypes.to_string())
    print("\n[C] First 5 records:")
    print(df.head().to_string(index=False))
    print("\n[D] Numerical summary:")
    print(df.describe().round(2).to_string())
    print("\n[E] Categorical summary:")
    print(df.describe(include=["object"]).to_string())


def assess_quality(df):
    """FR-1.4 / FR-1.5 Null counts and duplicate Checkin_IDs."""
    nulls = df.isnull().sum()
    print("\n[A] Missing values per column:")
    print(nulls.to_string())

    dup_mask = df.duplicated(subset=["Checkin_ID"], keep=False)
    dup_records = df[dup_mask].sort_values("Checkin_ID")
    n_dup_rows = int(df.duplicated(subset=["Checkin_ID"]).sum())
    print(f"\n[B] Repeated Checkin_ID rows (extra copies): {n_dup_rows}")
    if n_dup_rows:
        print(dup_records.to_string(index=False))
    return nulls, dup_records, n_dup_rows


# ==============================================================================
# MODULE 2 - DATA CLEANING AND FEATURE ENGINEERING
# ==============================================================================
def clean_data(df, dup_records):
    """FR-2.1 to FR-2.5. Returns cleaned frame and a log of what was done."""
    log = {}
    df = df.copy()

    # FR-2.3 / BR-01: keep first occurrence of each Checkin_ID
    before = len(df)
    df = df.drop_duplicates(subset=["Checkin_ID"], keep="first").reset_index(drop=True)
    log["duplicates_removed"] = before - len(df)

    # FR-2.1 / BR-02: numeric conversion + median imputation
    df["Workout_Duration_Mins"] = pd.to_numeric(df["Workout_Duration_Mins"], errors="coerce")
    n_missing_dur = int(df["Workout_Duration_Mins"].isna().sum())
    if df["Workout_Duration_Mins"].notna().sum() == 0:
        fail("No valid workout durations - median cannot be calculated.")
    median_dur = float(df["Workout_Duration_Mins"].median())
    mean_dur = float(df["Workout_Duration_Mins"].mean())
    df["Workout_Duration_Mins"] = df["Workout_Duration_Mins"].fillna(median_dur)
    log.update(durations_imputed=n_missing_dur, duration_median=median_dur,
               duration_mean_before=mean_dur)

    # FR-2.2 / BR-03: unrecorded classes are labelled, never inferred
    blank = df["Fitness_Class"].isna() | (df["Fitness_Class"].astype(str).str.strip() == "")
    log["class_not_recorded"] = int(blank.sum())
    df["Fitness_Class"] = df["Fitness_Class"].where(~blank, NOT_RECORDED)

    # Monthly_Fee should be numeric
    df["Monthly_Fee"] = pd.to_numeric(df["Monthly_Fee"], errors="coerce")
    log["fee_invalid"] = int(df["Monthly_Fee"].isna().sum())

    # FR-2.4: parse timestamps, invalid -> NaT (counted, kept in the data)
    df["Checkin_Time"] = pd.to_datetime(df["Checkin_Time"], errors="coerce")
    log["invalid_timestamps"] = int(df["Checkin_Time"].isna().sum())

    # FR-2.5: derived time features
    ts = df["Checkin_Time"]
    df["Weekday"] = ts.dt.day_name()
    df["Hour"] = ts.dt.hour
    df["Month"] = ts.dt.month
    df["Year"] = ts.dt.year
    df["Month_Year"] = ts.dt.to_period("M").astype("string")

    print(f"\n[A] Duplicate Checkin_ID rows removed (first kept): {log['duplicates_removed']}")
    print(f"[B] Workout duration: {n_missing_dur} missing/invalid values imputed "
          f"with median = {median_dur:.2f} min (mean was {mean_dur:.2f} min)")
    print(f"[C] Fitness_Class missing -> '{NOT_RECORDED}': {log['class_not_recorded']}")
    print(f"[D] Invalid timestamps (excluded from time/month groupings): "
          f"{log['invalid_timestamps']}")
    print(f"[E] Cleaned dataset shape: {df.shape}")
    return df, log


# ==============================================================================
# MODULE 3 - MEMBERSHIP AND VISIT PATTERN ANALYTICS
# ==============================================================================
def overall_statistics(df):
    """FR-3.1"""
    stats = {
        "Total check-ins": len(df),
        "Unique members": df["Member_ID"].nunique(),
        "Mean workout duration (min)": round(df["Workout_Duration_Mins"].mean(), 2),
        "Minimum workout duration (min)": df["Workout_Duration_Mins"].min(),
        "Maximum workout duration (min)": df["Workout_Duration_Mins"].max(),
        "Total workout hours": round(df["Workout_Duration_Mins"].sum() / 60, 2),
    }
    table = pd.DataFrame({"Metric": list(stats),
                          "Value": pd.Series(list(stats.values()), dtype=object)})
    print(table.to_string(index=False))
    return table


def tier_comparison(df):
    """FR-3.2"""
    total = len(df)
    t = df.groupby("Membership_Tier").agg(
        Check_ins=("Checkin_ID", "count"),
        Unique_Members=("Member_ID", "nunique"),
        Avg_Duration_Mins=("Workout_Duration_Mins", "mean"),
    ).reset_index()
    t["Visits_Per_Member"] = (t["Check_ins"] / t["Unique_Members"]).round(2)
    t["Checkin_Share_Pct"] = (t["Check_ins"] / total * 100).round(2)
    t["Avg_Duration_Mins"] = t["Avg_Duration_Mins"].round(2)
    t["Membership_Tier"] = pd.Categorical(t["Membership_Tier"], TIER_ORDER, ordered=True)
    t = t.sort_values("Membership_Tier").reset_index(drop=True)
    t["Membership_Tier"] = t["Membership_Tier"].astype(str)
    print(t.to_string(index=False))
    return t


def peak_hours(df):
    """FR-3.3 (BR-04: valid timestamps only)"""
    valid = df.dropna(subset=["Hour"])
    counts = valid["Hour"].astype(int).value_counts().reindex(range(24), fill_value=0)
    top = counts.max()
    busiest = [f"{h:02d}:00" for h in counts[counts == top].index]
    table = counts.rename_axis("Hour").reset_index(name="Checkins")
    print(f"Busiest hour(s): {', '.join(busiest)} with {top} check-ins")
    print("Top 5 hours:")
    print(table.sort_values("Checkins", ascending=False).head(5).to_string(index=False))
    return table, busiest


def busiest_weekdays(df):
    """FR-3.4"""
    valid = df.dropna(subset=["Weekday"])
    counts = valid["Weekday"].value_counts().reindex(WEEKDAY_ORDER, fill_value=0)
    top = counts.max()
    busiest = list(counts[counts == top].index)
    table = counts.rename_axis("Weekday").reset_index(name="Checkins")
    print(table.to_string(index=False))
    print(f"Busiest day(s): {', '.join(busiest)} with {top} check-ins")
    return table, busiest


def class_ranking(df):
    """FR-3.5 / BR-03: named classes only"""
    named = df[df["Fitness_Class"] != NOT_RECORDED]
    r = named["Fitness_Class"].value_counts().rename_axis("Fitness_Class") \
        .reset_index(name="Attendance")
    r["Percentage"] = (r["Attendance"] / len(named) * 100).round(2)
    r["Rank"] = np.arange(1, len(r) + 1)
    r = r[["Rank", "Fitness_Class", "Attendance", "Percentage"]]
    print(f"(Based on {len(named)} check-ins with a recorded class; "
          f"{len(df) - len(named)} 'Not Recorded' excluded)")
    print(r.to_string(index=False))
    return r


def gender_comparison(df):
    """FR-3.6"""
    g = df.groupby("Gender")["Workout_Duration_Mins"].agg(
        Count="count", Mean_Duration="mean", Median_Duration="median").round(2).reset_index()
    print(g.to_string(index=False))
    return g


# ==============================================================================
# MODULE 4 - REVENUE, ENGAGEMENT AND STATISTICAL ANALYSIS
# ==============================================================================
def estimated_revenue(df):
    """FR-4.1 / BR-06: each distinct member counted once per month and tier."""
    valid = df.dropna(subset=["Checkin_Time"])
    one_per_member = valid.drop_duplicates(subset=["Month_Year", "Member_ID"])
    by_month_tier = one_per_member.groupby(["Month_Year", "Membership_Tier"]).agg(
        Distinct_Members=("Member_ID", "nunique"),
        Estimated_Revenue=("Monthly_Fee", "sum")).reset_index()
    by_month = by_month_tier.groupby("Month_Year")["Estimated_Revenue"].sum().reset_index()
    by_tier = by_month_tier.groupby("Membership_Tier")["Estimated_Revenue"].sum().reset_index()
    by_tier["Share_Pct"] = (by_tier["Estimated_Revenue"] / by_tier["Estimated_Revenue"].sum()
                            * 100).round(2)
    by_tier["Membership_Tier"] = pd.Categorical(by_tier["Membership_Tier"], TIER_ORDER,
                                                ordered=True)
    by_tier = by_tier.sort_values("Membership_Tier")
    by_tier["Membership_Tier"] = by_tier["Membership_Tier"].astype(str)

    print("(ESTIMATED revenue - not verified collected billing)")
    print("\n[A] Estimated revenue per month:")
    print(by_month.round(2).to_string(index=False))
    print("\n[B] Estimated revenue per tier:")
    print(by_tier.round(2).to_string(index=False))
    pivot = by_month_tier.pivot(index="Month_Year", columns="Membership_Tier",
                                values="Estimated_Revenue").fillna(0)
    pivot = pivot.reindex(columns=[t for t in TIER_ORDER if t in pivot.columns])
    print("\n[C] Estimated revenue per month and tier:")
    print(pivot.round(2).to_string())
    return by_month_tier, by_month, by_tier, pivot


def low_engagement(df):
    """FR-4.2 / BR-05: member-months with fewer than 2 visits."""
    valid = df.dropna(subset=["Checkin_Time"])
    mm = valid.groupby(["Member_ID", "Month_Year"]).size().reset_index(name="Visits")
    mm["Low_Engagement"] = mm["Visits"] < LOW_ENGAGEMENT_THRESHOLD
    low = mm[mm["Low_Engagement"]]
    members_low = low["Member_ID"].nunique()
    summary = {
        "member_months": len(mm),
        "low_member_months": len(low),
        "low_pct": round(len(low) / len(mm) * 100, 2) if len(mm) else 0.0,
        "members_with_low": members_low,
        "total_members": valid["Member_ID"].nunique(),
    }
    print(f"Member-months analysed              : {summary['member_months']}")
    print(f"Low-engagement member-months (<{LOW_ENGAGEMENT_THRESHOLD})  : "
          f"{summary['low_member_months']} ({summary['low_pct']}%)")
    print(f"Members with >=1 low-engagement month: {members_low} of {summary['total_members']}")
    print("Note: only months with at least one recorded visit are observable; months with "
          "zero visits are not in the data.")
    return mm, low, summary


def numpy_statistics(df):
    """FR-4.3"""
    arr = df["Workout_Duration_Mins"].to_numpy(dtype=float)
    res = {
        "Mean": np.mean(arr),
        "Median": np.median(arr),
        "Population variance": np.var(arr),
        "Q1 (25th percentile)": np.percentile(arr, 25),
        "Q2 (50th percentile)": np.percentile(arr, 50),
        "Q3 (75th percentile)": np.percentile(arr, 75),
    }
    table = pd.DataFrame({"Statistic": list(res), "Value": np.round(list(res.values()), 4)})
    print(table.to_string(index=False))
    return table


def iqr_outliers(df):
    """FR-4.4 / BR-07: flag, never delete."""
    arr = df["Workout_Duration_Mins"].to_numpy(dtype=float)
    q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
    iqr = q3 - q1
    lower, upper = q1 - IQR_MULTIPLIER * iqr, q3 + IQR_MULTIPLIER * iqr
    mask = (df["Workout_Duration_Mins"] < lower) | (df["Workout_Duration_Mins"] > upper)
    out = df[mask]
    print(f"Q1 = {q1:.2f} | Q3 = {q3:.2f} | IQR = {iqr:.2f}")
    print(f"Lower fence = {lower:.2f} | Upper fence = {upper:.2f}")
    print(f"Outlier sessions flagged: {len(out)} "
          f"({(arr < lower).sum()} short, {(arr > upper).sum()} long) - retained in data")
    if len(out):
        print(out[["Checkin_ID", "Member_ID", "Membership_Tier",
                   "Workout_Duration_Mins"]].to_string(index=False))
    return out, dict(q1=q1, q3=q3, iqr=iqr, lower=lower, upper=upper)


def workout_intensity(df):
    """FR-4.5 / BR-08"""
    d = df["Workout_Duration_Mins"]
    df["Workout_Intensity"] = np.select(
        [d < SHORT_LIMIT, (d >= SHORT_LIMIT) & (d <= EXTENDED_LIMIT), d > EXTENDED_LIMIT],
        ["Short", "Moderate", "Extended"], default="Unknown")
    t = df["Workout_Intensity"].value_counts().reindex(
        ["Short", "Moderate", "Extended"], fill_value=0).rename_axis("Intensity") \
        .reset_index(name="Sessions")
    t["Percentage"] = (t["Sessions"] / len(df) * 100).round(2)
    print(f"Short < {SHORT_LIMIT} | Moderate {SHORT_LIMIT}-{EXTENDED_LIMIT} inclusive | "
          f"Extended > {EXTENDED_LIMIT} minutes")
    print(t.to_string(index=False))
    return t


# ==============================================================================
# MODULE 5 - VISUALIZATION AND REPORT GENERATION
# ==============================================================================
def label_bars(ax, fmt="{:.0f}", offset_frac=0.01, inside=False):
    top = max((p.get_height() for p in ax.patches), default=0)
    for p in ax.patches:
        h = p.get_height()
        if np.isnan(h):
            continue
        y = h / 2 if inside else h + top * offset_frac
        ax.annotate(fmt.format(h), (p.get_x() + p.get_width() / 2, y), ha="center",
                    va="center" if inside else "bottom", fontsize=9, fontweight="bold",
                    color="white" if inside else "black")


def make_charts(df, T, viz_dir):
    """FR-5.1: ten analytical charts saved as PNG."""
    saved = []

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(viz_dir / name, dpi=130)
        plt.close(fig)
        saved.append(name)

    # 1. Tier comparison: check-ins and average duration
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
    sns.barplot(data=T["tier"], x="Membership_Tier", y="Check_ins", hue="Membership_Tier",
                palette="Blues_r", legend=False, ax=a1)
    a1.set_title("Check-ins by Membership Tier"); a1.set_xlabel("Membership Tier")
    a1.set_ylabel("Number of Check-ins"); label_bars(a1)
    sns.barplot(data=T["tier"], x="Membership_Tier", y="Avg_Duration_Mins",
                hue="Membership_Tier", palette="Greens_r", legend=False, ax=a2)
    a2.set_title("Average Workout Duration by Tier"); a2.set_xlabel("Membership Tier")
    a2.set_ylabel("Minutes"); label_bars(a2, "{:.1f}")
    save(fig, "01_tier_comparison.png")

    # 2. Weekday
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.barplot(data=T["weekday"], x="Weekday", y="Checkins", hue="Weekday",
                palette="viridis", legend=False, ax=ax)
    ax.set_title("Busiest Days of the Week"); ax.set_xlabel("Day of Week")
    ax.set_ylabel("Check-ins"); label_bars(ax)
    save(fig, "02_busiest_weekdays.png")

    # 3. Hour
    fig, ax = plt.subplots(figsize=(12, 4.8))
    hrs = T["hour"]
    top = hrs["Checkins"].max()
    colors = ["#ff7f0e" if c == top else "#1f77b4" for c in hrs["Checkins"]]
    ax.bar(hrs["Hour"], hrs["Checkins"], color=colors)
    ax.set_xticks(range(24)); ax.set_xticklabels([f"{h:02d}" for h in range(24)])
    ax.set_title("Peak Hours (Check-ins by Hour of Day; busiest highlighted)")
    ax.set_xlabel("Hour of Day (24-hour clock)"); ax.set_ylabel("Check-ins")
    save(fig, "03_peak_hours.png")

    # 4. Class ranking
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.barplot(data=T["class"], x="Fitness_Class", y="Attendance", hue="Fitness_Class",
                palette="mako", legend=False, ax=ax)
    ax.set_title("Fitness Class Attendance Ranking (recorded classes only)")
    ax.set_xlabel("Fitness Class"); ax.set_ylabel("Attendance"); label_bars(ax)
    save(fig, "04_class_ranking.png")

    # 5. Gender
    fig, ax = plt.subplots(figsize=(8, 4.8))
    g = T["gender"].melt(id_vars="Gender", value_vars=["Mean_Duration", "Median_Duration"],
                         var_name="Measure", value_name="Minutes")
    sns.barplot(data=g, x="Gender", y="Minutes", hue="Measure", ax=ax)
    ax.set_title("Workout Duration by Gender (Mean vs Median)")
    ax.set_ylabel("Minutes"); ax.legend(title="Measure")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.1f", fontsize=8)
    save(fig, "05_gender_duration.png")

    # 6. Estimated revenue by month and tier (stacked)
    fig, ax = plt.subplots(figsize=(11, 5))
    T["rev_pivot"].plot(kind="bar", stacked=True, ax=ax, colormap="crest"
                        if "crest" in plt.colormaps() else "viridis")
    ax.set_title("Estimated Monthly Revenue by Tier (distinct members x Monthly_Fee)")
    ax.set_xlabel("Month"); ax.set_ylabel("Estimated Revenue ($)")
    ax.legend(title="Tier"); plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    save(fig, "06_estimated_revenue.png")

    # 7. Duration histogram
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.histplot(df["Workout_Duration_Mins"], bins=25, kde=True, ax=ax)
    ax.axvline(SHORT_LIMIT, color="red", ls="--", lw=1, label=f"{SHORT_LIMIT} min")
    ax.axvline(EXTENDED_LIMIT, color="red", ls="--", lw=1, label=f"{EXTENDED_LIMIT} min")
    ax.set_title("Distribution of Workout Duration"); ax.set_xlabel("Minutes")
    ax.set_ylabel("Sessions"); ax.legend()
    save(fig, "07_duration_distribution.png")

    # 8. Boxplot with IQR fences
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.boxplot(data=df, x="Membership_Tier", y="Workout_Duration_Mins",
                order=TIER_ORDER, hue="Membership_Tier", palette="pastel",
                legend=False, ax=ax)
    ax.axhline(T["iqr"]["lower"], color="red", ls=":", lw=1)
    ax.axhline(T["iqr"]["upper"], color="red", ls=":", lw=1,
               label="Overall IQR fences")
    ax.set_title("Workout Duration by Tier with IQR Outlier Fences")
    ax.set_xlabel("Membership Tier"); ax.set_ylabel("Minutes"); ax.legend()
    save(fig, "08_duration_boxplot.png")

    # 9. Intensity
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.barplot(data=T["intensity"], x="Intensity", y="Sessions", hue="Intensity",
                palette="Spectral", legend=False, ax=ax)
    ax.set_title("Workout Intensity Distribution")
    ax.set_xlabel("Intensity Category"); ax.set_ylabel("Sessions")
    for p, pct in zip(ax.patches, T["intensity"]["Percentage"]):
        ax.annotate(f"{int(p.get_height())} ({pct:.1f}%)",
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontweight="bold", fontsize=9)
    save(fig, "09_workout_intensity.png")

    # 10. Low engagement by month
    mm = T["member_months"]
    by_m = mm.groupby("Month_Year")["Low_Engagement"].agg(["sum", "count"]).reset_index()
    by_m["Low_Pct"] = by_m["sum"] / by_m["count"] * 100
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(by_m["Month_Year"], by_m["Low_Pct"], marker="o", color="#d62728", lw=2.2)
    ax.set_ylim(0, 100)
    ax.set_title(f"Low-Engagement Member-Months (<{LOW_ENGAGEMENT_THRESHOLD} visits) by Month")
    ax.set_xlabel("Month"); ax.set_ylabel("% of active members")
    for x, y in zip(by_m["Month_Year"], by_m["Low_Pct"]):
        ax.annotate(f"{y:.0f}%", (x, y + 3), ha="center", fontsize=9)
    save(fig, "10_low_engagement.png")
    return saved


def build_recommendations(T, busiest_h, busiest_d):
    """Five evidence-based recommendations computed from the results (FR-5.3)."""
    tier = T["tier"].set_index("Membership_Tier")
    top_tier = tier["Check_ins"].idxmax()
    cls = T["class"]
    rev_tier = T["rev_tier"].set_index("Membership_Tier")
    top_rev_tier = rev_tier["Estimated_Revenue"].idxmax()
    ls = T["low_summary"]
    classes_txt = ", ".join(f"{r.Fitness_Class} ({r.Attendance})" for r in cls.itertuples())

    return [
        f"Retention onboarding: {top_tier} members generate the most check-ins "
        f"({tier.loc[top_tier, 'Checkin_Share_Pct']:.1f}% of visits). Offer visit-milestone "
        f"rewards (e.g. 5, 10, 20 visits) in the first 60 days to build routine.",
        f"Re-engagement: {ls['low_member_months']} of {ls['member_months']} member-months "
        f"({ls['low_pct']}%) had fewer than {LOW_ENGAGEMENT_THRESHOLD} visits, affecting "
        f"{ls['members_with_low']} of {ls['total_members']} members. Contact these members "
        f"with a check-in call or guest pass after a period of inactivity.",
        f"Class scheduling: recorded attendance ranks as {classes_txt}. Give the leading "
        f"class more slots at peak times and review low-attendance classes for timing "
        f"or format changes.",
        f"Peak staffing: the busiest hour is {', '.join(busiest_h)} and the busiest day is "
        f"{', '.join(busiest_d)}. Schedule floor staff, cleaning and equipment maintenance "
        f"around these periods.",
        f"Tier upsell: {top_rev_tier} contributes the largest share of estimated revenue "
        f"({rev_tier.loc[top_rev_tier, 'Share_Pct']:.1f}%). Target highly engaged members "
        f"of lower tiers with upgrade offers. Revenue figures are estimates, not verified "
        f"billing.",
    ]


def write_report(path, df, log, T, recs, charts, csv_path):
    """FR-5.3: objective, dataset, cleaning, findings, limitations, recommendations."""
    def md(t):
        return t.to_markdown(index=False) if hasattr(t, "to_markdown") else t.to_string(index=False)

    try:
        import tabulate  # noqa: F401
        fmt = md
    except ImportError:
        fmt = lambda t: "```\n" + t.to_string(index=False) + "\n```"

    ts = df["Checkin_Time"].dropna()
    lines = [
        "# Gym & Fitness Center Membership Analytics Report", "",
        "## 1. Objective",
        "Summarise visit patterns, tier activity, class attendance, workout duration, "
        "estimated revenue and low-engagement members to support scheduling and "
        "retention decisions.", "",
        "## 2. Dataset",
        f"- Source file: `{csv_path.name}`",
        f"- Records after cleaning: {len(df)}",
        f"- Date range: {ts.min():%d %b %Y} to {ts.max():%d %b %Y}", "",
        "## 3. Data Cleaning Summary",
        f"- Duplicate Checkin_ID rows removed (first kept): {log['duplicates_removed']}",
        f"- Missing workout durations imputed with median "
        f"({log['duration_median']:.2f} min): {log['durations_imputed']}",
        f"- Fitness_Class marked '{NOT_RECORDED}': {log['class_not_recorded']}",
        f"- Invalid timestamps excluded from time groupings: {log['invalid_timestamps']}", "",
        "## 4. Key Findings", "### Overall statistics", fmt(T["overall"]), "",
        "### Membership tier comparison", fmt(T["tier"]), "",
        "### Fitness class ranking", fmt(T["class"]), "",
        "### Gender duration comparison", fmt(T["gender"]), "",
        "### Estimated revenue by tier", fmt(T["rev_tier"].round(2)), "",
        "### Workout intensity", fmt(T["intensity"]), "",
        "### Statistics (NumPy)", fmt(T["numpy"]), "",
        "### IQR outliers",
        f"Fences: {T['iqr']['lower']:.2f} to {T['iqr']['upper']:.2f} min; "
        f"{len(T['outliers'])} sessions flagged and retained.", "",
        "## 5. Limitations",
        "- Revenue is an estimate (distinct observed members x Monthly_Fee), not billing.",
        "- Low engagement uses a transparent rule (<2 visits per member-month); months "
        "with zero visits are not observable in check-in data.",
        "- Absence of visits does not prove cancellation.",
        "- Checkin_Time is assumed to be in one consistent local time zone.", "",
        "## 6. Recommendations",
    ]
    lines += [f"{i}. {r}" for i, r in enumerate(recs, 1)]
    lines += ["", "## 7. Charts"] + [f"- visualizations/{c}" for c in charts]
    path.write_text("\n".join(lines), encoding="utf-8")


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    csv_path = locate_csv()
    out_dir = csv_path.resolve().parent / OUTPUT_DIR_NAME
    viz_dir = out_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)  # FR-5.4

    banner("1. DATA INGESTION AND QUALITY ASSESSMENT")
    raw = load_data(csv_path)
    validate_schema(raw)
    inspect_data(raw)
    nulls, dup_records, n_dup = assess_quality(raw)
    if n_dup:
        dup_records.to_csv(out_dir / "duplicate_checkins.csv", index=False)

    banner("2. DATA CLEANING AND FEATURE ENGINEERING")
    df, log = clean_data(raw, dup_records)

    T = {}
    banner("3. OVERALL STATISTICS (FR-3.1)")
    T["overall"] = overall_statistics(df)
    banner("4. MEMBERSHIP TIER COMPARISON (FR-3.2)")
    T["tier"] = tier_comparison(df)
    banner("5. PEAK-HOUR ANALYSIS (FR-3.3)")
    T["hour"], busiest_h = peak_hours(df)
    banner("6. BUSIEST WEEKDAY ANALYSIS (FR-3.4)")
    T["weekday"], busiest_d = busiest_weekdays(df)
    banner("7. FITNESS CLASS RANKING (FR-3.5)")
    T["class"] = class_ranking(df)
    banner("8. GENDER DURATION COMPARISON (FR-3.6)")
    T["gender"] = gender_comparison(df)
    banner("9. ESTIMATED REVENUE (FR-4.1)")
    T["rev_month_tier"], T["rev_month"], T["rev_tier"], T["rev_pivot"] = estimated_revenue(df)
    banner("10. LOW-ENGAGEMENT MEMBER-MONTHS (FR-4.2)")
    T["member_months"], T["low"], T["low_summary"] = low_engagement(df)
    banner("11. NUMPY DESCRIPTIVE STATISTICS (FR-4.3)")
    T["numpy"] = numpy_statistics(df)
    banner("12. IQR OUTLIER DETECTION (FR-4.4)")
    T["outliers"], T["iqr"] = iqr_outliers(df)
    banner("13. WORKOUT INTENSITY (FR-4.5)")
    T["intensity"] = workout_intensity(df)

    banner("14. EXPORTING RESULTS, CHARTS AND REPORT (Module 5)")
    # FR-2.6 cleaned dataset
    df.to_csv(out_dir / "cleaned_gym_membership.csv", index=False)
    T["outliers"].to_csv(out_dir / "iqr_outliers.csv", index=False)

    # FR-5.2 summary tables
    with pd.ExcelWriter(out_dir / "analysis_tables.xlsx", engine="openpyxl") as xw:
        sheets = {
            "Overall": T["overall"], "Tier_Comparison": T["tier"],
            "Peak_Hours": T["hour"], "Weekdays": T["weekday"],
            "Class_Ranking": T["class"], "Gender": T["gender"],
            "Revenue_Month_Tier": T["rev_month_tier"].round(2),
            "Revenue_By_Tier": T["rev_tier"].round(2),
            "Member_Months": T["member_months"], "NumPy_Stats": T["numpy"],
            "IQR_Outliers": T["outliers"], "Intensity": T["intensity"],
        }
        for name, table in sheets.items():
            table.to_excel(xw, sheet_name=name, index=False)

    # FR-5.1 charts
    charts = make_charts(df, T, viz_dir)

    # FR-5.3 report with five recommendations
    recs = build_recommendations(T, busiest_h, busiest_d)
    print("\nRecommendations:")
    for i, r in enumerate(recs, 1):
        print(f"  {i}. {r}")
    write_report(out_dir / "gym_analytics_report.md", df, log, T, recs, charts, csv_path)

    print(f"\nAll outputs saved to: {out_dir}")
    print(f"  - {len(charts)} charts in {viz_dir.name}/")
    print("Analysis completed successfully.")


if __name__ == "__main__":
    main()
