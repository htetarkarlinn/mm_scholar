import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Always resolve paths relative to this file's location
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DB_PATH    = os.path.join(BASE_DIR, "mm_scholar.db")
OUTPUT_DIR = os.path.join(BASE_DIR, "static", "eda")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.size":        11,
})

BLUE   = "#1F4E79"
BLUE2  = "#2E75B6"
GREEN  = "#1D9E75"
ORANGE = "#EF9F27"
RED    = "#E24B4A"


def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM scholarships", conn)
    conn.close()
    return df


def print_overview(df):
    print("\n=== Dataset Overview ===")
    print(f"Shape: {df.shape}")
    print(f"Unique scholarships : {df['scholarship_name'].nunique()}")
    print(f"Unique countries    : {df['country_of_study'].nunique()}")
    print(f"Unique fields       : {df['field_of_study'].nunique()}")
    print(f"Unique levels       : {df['level'].nunique()}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nNumeric summary:")
    print(df[["min_gpa", "min_ielts", "deadline_month", "duration_years"]].describe().round(2))


def plot_ml_features(df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Core ML Feature Distributions", fontsize=14, fontweight="bold")

    # level
    level_order  = ["diploma", "undergraduate", "postgraduate", "phd", "short_course"]
    level_counts = df["level"].value_counts().reindex(level_order).fillna(0)
    axes[0, 0].barh(level_order, level_counts.values, color=BLUE)
    axes[0, 0].set_title("Level")
    axes[0, 0].set_xlabel("Rows")
    for i, v in enumerate(level_counts.values):
        axes[0, 0].text(v + 0.5, i, str(int(v)), va="center", fontsize=10)

    # field of study
    field_counts = df["field_of_study"].value_counts()
    colors = [RED if v < 10 else GREEN if v >= 40 else BLUE2 for v in field_counts.values]
    axes[0, 1].barh(field_counts.index, field_counts.values, color=colors)
    axes[0, 1].set_title("Field of study")
    axes[0, 1].set_xlabel("Rows")
    for i, v in enumerate(field_counts.values):
        axes[0, 1].text(v + 0.3, i, str(int(v)), va="center", fontsize=9)
    legend = [
        mpatches.Patch(color=GREEN, label="≥40 rows"),
        mpatches.Patch(color=BLUE2, label="10–39 rows"),
        mpatches.Patch(color=RED,   label="<10 rows"),
    ]
    axes[0, 1].legend(handles=legend, fontsize=8)

    # funding type
    fund_counts = df["funding_type"].value_counts()
    axes[1, 0].bar(fund_counts.index, fund_counts.values, color=[GREEN, ORANGE], width=0.5)
    axes[1, 0].set_title("Funding type")
    axes[1, 0].set_ylabel("Rows")
    for i, (_, v) in enumerate(fund_counts.items()):
        pct = v / len(df) * 100
        axes[1, 0].text(i, v + 2, f"{int(v)} ({pct:.0f}%)", ha="center", fontsize=10)

    # top 15 countries
    country_counts = df["country_of_study"].value_counts().head(15)
    axes[1, 1].barh(country_counts.index[::-1], country_counts.values[::-1], color=BLUE2)
    axes[1, 1].set_title("Top 15 countries")
    axes[1, 1].set_xlabel("Rows")
    for i, v in enumerate(country_counts.values[::-1]):
        axes[1, 1].text(v + 0.2, i, str(int(v)), va="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "ml_feature_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_informative_columns(df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Informative Column Distributions", fontsize=14, fontweight="bold")

    # GPA
    gpa_counts = df["min_gpa"].value_counts().sort_index()
    axes[0].bar(gpa_counts.index.astype(str), gpa_counts.values, color=BLUE)
    axes[0].set_title("Min GPA (0.0 = not stated)")
    axes[0].set_xlabel("GPA")
    axes[0].set_ylabel("Rows")
    for i, v in enumerate(gpa_counts.values):
        axes[0].text(i, v + 1, str(int(v)), ha="center", fontsize=9)

    # IELTS
    ielts_counts = df["min_ielts"].value_counts().sort_index()
    axes[1].bar(ielts_counts.index.astype(str), ielts_counts.values, color=BLUE2)
    axes[1].set_title("Min IELTS (0.0 = not required)")
    axes[1].set_xlabel("Score")
    axes[1].set_ylabel("Rows")
    for i, v in enumerate(ielts_counts.values):
        axes[1].text(i, v + 1, str(int(v)), ha="center", fontsize=9)

    # deadline by month
    months = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]
    month_counts = df["deadline_month"].value_counts().reindex(range(1, 13), fill_value=0)
    axes[2].bar(months, month_counts.values, color=RED)
    axes[2].set_title("Deadline month")
    axes[2].set_xlabel("Month")
    axes[2].set_ylabel("Rows")
    for i, v in enumerate(month_counts.values):
        if v > 0:
            axes[2].text(i, v + 0.5, str(int(v)), ha="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "informative_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_scholarship_coverage(df):
    coverage = df.groupby("scholarship_name")["field_of_study"].count().sort_values()
    colors   = [GREEN if v >= 7 else BLUE2 if v >= 4 else RED for v in coverage.values]

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.barh(coverage.index, coverage.values, color=colors)
    ax.set_title("Rows per scholarship", fontsize=13, fontweight="bold")
    ax.set_xlabel("Rows")
    for i, v in enumerate(coverage.values):
        ax.text(v + 0.1, i, str(int(v)), va="center", fontsize=9)
    legend = [
        mpatches.Patch(color=GREEN, label="≥7 rows"),
        mpatches.Patch(color=BLUE2, label="4–6 rows"),
        mpatches.Patch(color=RED,   label="<4 rows"),
    ]
    ax.legend(handles=legend, fontsize=9)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "scholarship_coverage.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_correlation(df):
    cols = ["country_of_study", "level", "field_of_study", "funding_type",
            "min_gpa", "min_ielts", "deadline_month", "duration_years"]
    df_enc = df[cols].copy()
    df_enc["duration_years"] = pd.to_numeric(df_enc["duration_years"], errors="coerce")
    for col in ["country_of_study", "level", "field_of_study", "funding_type"]:
        df_enc[col] = LabelEncoder().fit_transform(df_enc[col])

    corr = df_enc.corr(numeric_only=True).round(2)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="Blues", center=0, linewidths=0.5,
                ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature correlation matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "feature_correlation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def print_data_quality(df):
    print("\n=== Data Quality Summary ===")
    print(f"Missing values total: {df.isnull().sum().sum()}")

    print("\nField of study counts:")
    print(df["field_of_study"].value_counts().to_string())

    sparse = df["field_of_study"].value_counts()
    sparse = sparse[sparse < 10]
    if len(sparse) > 0:
        print(f"\nWARNING — sparse fields (<10 rows):\n{sparse}")
    else:
        print("\nNo sparse field categories")

    print("\nFunding type balance:")
    print((df["funding_type"].value_counts(normalize=True) * 100).round(1))


if __name__ == "__main__":
    print(f"BASE_DIR:   {BASE_DIR}")
    print(f"DB_PATH:    {DB_PATH}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"Output dir created: {os.path.exists(OUTPUT_DIR)}")

    df = load_data()
    print_overview(df)
    plot_ml_features(df)
    plot_informative_columns(df)
    plot_scholarship_coverage(df)
    plot_correlation(df)
    print_data_quality(df)
    print("\nEDA complete. Charts saved to static/eda/")