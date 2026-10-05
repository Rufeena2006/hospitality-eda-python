"""
eda.py
Exploratory Data Analysis + visualization for hotel booking data.

Run:  python generate_data.py   (once, creates data/hotel_bookings.csv)
      python eda.py             (prints findings, saves figures/ and outputs/)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # works without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

DATA = Path("data/hotel_bookings.csv")
FIG = Path("figures")
OUT = Path("outputs")
FIG.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({"figure.dpi": 110, "axes.titleweight": "bold"})


def save(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=150)
    plt.close()
    print(f"  saved figures/{name}")


# ----------------------------------------------------------------- 1. LOAD
def load() -> pd.DataFrame:
    df = pd.read_csv(DATA, parse_dates=["arrival_date"])
    print("=" * 70, "\n1. DATA OVERVIEW\n", "=" * 70)
    print("Shape:", df.shape)
    print(df.dtypes.to_string())
    print("\nMissing values:\n", df.isna().sum()[df.isna().sum() > 0].to_string())
    print("Duplicate rows:", df.duplicated().sum())
    return df


# ---------------------------------------------------------------- 2. CLEAN
def clean(df: pd.DataFrame) -> pd.DataFrame:
    print("\n", "=" * 70, "\n2. CLEANING\n", "=" * 70)
    df = df.drop_duplicates().copy()
    df["children"] = df["children"].fillna(0).astype(int)        # assume none
    df["country"] = df["country"].fillna("Unknown")
    # Derived features
    df["total_guests"] = df["adults"] + df["children"]
    df["total_revenue"] = df["adr"] * df["stay_nights"]
    df["arrival_month"] = df["arrival_date"].dt.month
    df["arrival_year"] = df["arrival_date"].dt.year
    df["year_month"] = df["arrival_date"].dt.to_period("M").dt.to_timestamp()
    df["arrival_weekday"] = df["arrival_date"].dt.day_name()
    df["lead_bucket"] = pd.cut(df["lead_time"], [-1, 7, 30, 90, 180, 1000],
                               labels=["0-7d", "8-30d", "31-90d", "91-180d", "180d+"])
    print("Rows after de-duplication:", len(df))
    return df


# ------------------------------------------------------------ 3. STATISTICS
def describe(df: pd.DataFrame) -> None:
    print("\n", "=" * 70, "\n3. DESCRIPTIVE STATISTICS\n", "=" * 70)
    num = ["lead_time", "stay_nights", "adr", "total_guests", "total_revenue",
           "special_requests"]
    summary = df[num].describe().T
    summary["median"] = df[num].median()
    summary["variance"] = df[num].var()
    summary["iqr"] = df[num].quantile(.75) - df[num].quantile(.25)
    summary["skewness"] = df[num].skew()
    summary["kurtosis"] = df[num].kurt()
    summary = summary.round(2)
    print(summary.to_string())
    summary.to_csv(OUT / "descriptive_statistics.csv")

    print("\nCategorical frequencies:")
    for c in ["hotel", "market_segment", "room_type"]:
        print(f"\n{c}\n", df[c].value_counts(normalize=True).mul(100).round(1).to_string())

    print("\nMean ADR by room type:\n", df.groupby("room_type")["adr"].agg(
        ["mean", "median", "std"]).round(2).to_string())


# -------------------------------------------------------------- 4. OUTLIERS
def outliers(df: pd.DataFrame) -> pd.DataFrame:
    print("\n", "=" * 70, "\n4. OUTLIER DETECTION (IQR + Z-score)\n", "=" * 70)
    rows = []
    for col in ["adr", "lead_time", "total_revenue"]:
        q1, q3 = df[col].quantile([.25, .75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_iqr = ((df[col] < lo) | (df[col] > hi)).sum()
        z = np.abs(stats.zscore(df[col]))
        n_z = (z > 3).sum()
        rows.append([col, round(lo, 1), round(hi, 1), n_iqr, n_z])
    res = pd.DataFrame(rows, columns=["variable", "lower_fence", "upper_fence",
                                      "iqr_outliers", "zscore>3_outliers"])
    print(res.to_string(index=False))
    res.to_csv(OUT / "outlier_summary.csv", index=False)

    fig, ax = plt.subplots(1, 3, figsize=(13, 4.5))
    for a, col in zip(ax, ["adr", "lead_time", "total_revenue"]):
        sns.boxplot(y=df[col], ax=a, color="#4C72B0", flierprops={"markersize": 3})
        a.set_title(f"Box plot - {col}")
    save("04_outlier_boxplots.png")

    # Treatment: cap extreme ADR / lead-time (winsorise at 99th pct) for modelling views
    df = df.copy()
    for col in ["adr", "lead_time"]:
        df[f"{col}_capped"] = df[col].clip(upper=df[col].quantile(.99))
    return df


# ---------------------------------------------------------- 5. VISUALISATIONS
def distributions(df: pd.DataFrame) -> None:
    print("\n5. VISUALISATIONS")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    sns.histplot(df["adr_capped"], bins=40, kde=True, ax=ax[0], color="#4C72B0")
    ax[0].axvline(df["adr"].median(), color="red", ls="--", label="median")
    ax[0].set(title="Distribution of ADR (capped at 99th pct)", xlabel="ADR (currency units)")
    ax[0].legend()
    sns.histplot(df["lead_time_capped"], bins=40, kde=True, ax=ax[1], color="#55A868")
    ax[1].set(title="Distribution of Lead Time", xlabel="Days between booking and arrival")
    sns.countplot(x="stay_nights", data=df, ax=ax[2], color="#C44E52")
    ax[2].set(title="Length of Stay", xlabel="Nights")
    save("01_distributions.png")


def categorical(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    order = df["market_segment"].value_counts().index
    sns.countplot(y="market_segment", data=df, order=order, ax=ax[0], hue="hotel")
    ax[0].set(title="Bookings by Market Segment & Hotel", xlabel="Bookings", ylabel="")
    sns.boxplot(x="room_type", y="adr_capped", data=df, ax=ax[1],
                order=["Standard", "Family", "Deluxe", "Suite"])
    ax[1].set(title="ADR by Room Type", xlabel="", ylabel="ADR")
    save("02_segments_room_types.png")


def time_series(df: pd.DataFrame) -> None:
    monthly = df.groupby("year_month").agg(
        bookings=("booking_id", "count"),
        avg_adr=("adr_capped", "mean"),
        revenue=("total_revenue", "sum")).reset_index()
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax[0].plot(monthly["year_month"], monthly["bookings"], marker="o", color="#4C72B0")
    ax[0].set(title="Monthly Bookings (Seasonality)", ylabel="Bookings")
    ax[1].plot(monthly["year_month"], monthly["avg_adr"], marker="s", color="#DD8452",
               label="Avg ADR")
    ax[1].plot(monthly["year_month"], monthly["avg_adr"].rolling(3).mean(), ls="--",
               color="black", label="3-month moving avg")
    ax[1].set(title="Average Daily Rate Over Time", ylabel="ADR", xlabel="Month")
    ax[1].legend()
    save("03_time_series.png")
    monthly.round({"avg_adr": 2, "revenue": 2}).to_csv(OUT / "monthly_summary.csv", index=False)

    # Month x weekday heatmap
    pivot = df.pivot_table(index="arrival_month", columns="arrival_weekday",
                           values="booking_id", aggfunc="count")
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    plt.figure(figsize=(9, 5.5))
    sns.heatmap(pivot[days], annot=True, fmt="d", cmap="YlGnBu")
    plt.title("Arrivals: Month vs Weekday")
    plt.ylabel("Month")
    plt.xlabel("")
    save("07_month_weekday_heatmap.png")


def relationships(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    sample = df.sample(1500, random_state=1)
    sns.scatterplot(data=sample, x="lead_time_capped", y="adr_capped", hue="room_type",
                    alpha=.6, ax=ax[0])
    ax[0].set(title="Lead Time vs ADR", xlabel="Lead time (days)", ylabel="ADR")
    sns.regplot(data=sample, x="stay_nights", y="total_revenue",
                scatter_kws={"alpha": .3}, line_kws={"color": "red"}, ax=ax[1])
    ax[1].set(title="Length of Stay vs Total Revenue", xlabel="Nights", ylabel="Revenue")
    save("05_scatter_relationships.png")


def correlations(df: pd.DataFrame) -> None:
    print("\n", "=" * 70, "\n6. CORRELATION ANALYSIS\n", "=" * 70)
    cols = ["lead_time_capped", "stay_nights", "adr_capped", "total_guests",
            "special_requests", "previous_cancellations", "parking_spaces",
            "total_revenue", "is_canceled"]
    corr = df[cols].corr(method="pearson")
    spear = df[cols].corr(method="spearman")
    print("Pearson correlation with cancellation:\n",
          corr["is_canceled"].drop("is_canceled").sort_values().round(3).to_string())
    corr.round(3).to_csv(OUT / "correlation_matrix.csv")

    plt.figure(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=.5)
    plt.title("Pearson Correlation Matrix")
    save("06_correlation_heatmap.png")

    # Statistical tests
    cancelled = df[df.is_canceled == 1]["lead_time"]
    kept = df[df.is_canceled == 0]["lead_time"]
    t, p = stats.ttest_ind(cancelled, kept, equal_var=False)
    print(f"\nWelch t-test lead time (cancelled vs kept): t={t:.2f}, p={p:.2e}")
    chi, p2, _, _ = stats.chi2_contingency(pd.crosstab(df.market_segment, df.is_canceled))
    print(f"Chi-square market_segment vs cancellation: chi2={chi:.1f}, p={p2:.2e}")
    f, p3 = stats.f_oneway(*[g["adr_capped"].values for _, g in df.groupby("room_type")])
    print(f"ANOVA ADR across room types: F={f:.1f}, p={p3:.2e}")


def cancellations(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    seg = df.groupby("market_segment")["is_canceled"].mean().sort_values() * 100
    seg.plot.barh(ax=ax[0], color="#C44E52")
    ax[0].set(title="Cancellation Rate by Segment (%)", xlabel="%", ylabel="")
    lead = df.groupby("lead_bucket", observed=True)["is_canceled"].mean() * 100
    lead.plot.bar(ax=ax[1], color="#8172B2", rot=0)
    ax[1].set(title="Cancellation Rate by Lead Time (%)", xlabel="", ylabel="%")
    sr = df.groupby("special_requests")["is_canceled"].mean() * 100
    sr.plot.bar(ax=ax[2], color="#64B5CD", rot=0)
    ax[2].set(title="Cancellation Rate by Special Requests (%)", xlabel="Requests", ylabel="%")
    save("08_cancellations.png")
    print("\nOverall cancellation rate: %.1f%%" % (df.is_canceled.mean() * 100))
    print("Cancellation by lead bucket (%):\n", lead.round(1).to_string())


def revenue(df: pd.DataFrame) -> None:
    kept = df[df.is_canceled == 0]
    rev = kept.groupby(["hotel", "room_type"])["total_revenue"].sum().unstack() / 1000
    rev.plot.bar(stacked=True, figsize=(8, 5), rot=0)
    plt.title("Realised Revenue by Hotel & Room Type (non-cancelled, thousands)")
    plt.ylabel("Revenue (k)")
    plt.xlabel("")
    save("09_revenue_by_room.png")
    print("\nRealised revenue share by room type (%):\n",
          (kept.groupby("room_type")["total_revenue"].sum() /
           kept["total_revenue"].sum() * 100).round(1).to_string())


def main() -> None:
    df = clean(load())
    describe(df)
    df = outliers(df)
    distributions(df)
    categorical(df)
    time_series(df)
    relationships(df)
    correlations(df)
    cancellations(df)
    revenue(df)
    df.to_csv(OUT / "hotel_bookings_clean.csv", index=False)
    print("\nDone. Figures in ./figures, tables in ./outputs")


if __name__ == "__main__":
    main()
