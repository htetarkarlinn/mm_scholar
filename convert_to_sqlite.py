"""
convert_to_sqlite.py
CRISP-ML(Q) Phase 2 — Data Engineering
Converts scholarships_dataset.csv into mm_scholar.db (SQLite)
"""

import pandas as pd
import sqlite3
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


def load_csv(path: str) -> pd.DataFrame:
    log.info(f"Loading CSV from: {path}")
    df = pd.read_csv(path)
    log.info(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def validate(df: pd.DataFrame) -> None:
    log.info("Validating dataset...")

    required_columns = [
        "scholarship_id", "scholarship_name", "provider",
        "country_of_study", "level", "field_of_study", "funding_type",
        "min_gpa", "min_ielts", "deadline_month", "duration_years", "link"
    ]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    ml_features = ["country_of_study", "level", "field_of_study", "funding_type"]
    for col in ml_features:
        nulls = df[col].isnull().sum()
        if nulls > 0:
            raise ValueError(f"ML feature '{col}' has {nulls} missing values")

    valid_levels = {"diploma", "undergraduate", "postgraduate", "phd", "short_course"}
    invalid_levels = set(df["level"].unique()) - valid_levels
    if invalid_levels:
        raise ValueError(f"Invalid level values found: {invalid_levels}")

    valid_funding = {"fully_funded", "partial"}
    invalid_funding = set(df["funding_type"].unique()) - valid_funding
    if invalid_funding:
        raise ValueError(f"Invalid funding_type values found: {invalid_funding}")

    log.info("Validation passed")


def convert(df: pd.DataFrame, db_path: str) -> None:
    log.info(f"Converting to SQLite: {db_path}")
    conn = sqlite3.connect(db_path)
    df.to_sql("scholarships", conn, if_exists="replace", index=False)

    cursor = conn.cursor()
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_country ON scholarships(country_of_study)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_level ON scholarships(level)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_field ON scholarships(field_of_study)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_funding ON scholarships(funding_type)")
    conn.commit()

    count = cursor.execute("SELECT COUNT(*) FROM scholarships").fetchone()[0]
    log.info(f"Inserted {count} rows into 'scholarships' table")

    cols = [r[1] for r in cursor.execute("PRAGMA table_info(scholarships)").fetchall()]
    log.info(f"Columns: {cols}")
    conn.close()
    log.info(f"Database saved: {db_path} ")


def verify(db_path: str) -> None:
    log.info("Verifying database...")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM scholarships LIMIT 3", conn)
    log.info(f"Sample rows:\n{df[['scholarship_name','country_of_study','level','field_of_study','funding_type']].to_string()}")

    stats = pd.read_sql("""
        SELECT
            COUNT(*) as total_rows,
            COUNT(DISTINCT scholarship_name) as unique_scholarships,
            COUNT(DISTINCT country_of_study) as unique_countries,
            COUNT(DISTINCT field_of_study) as unique_fields,
            COUNT(DISTINCT level) as unique_levels
        FROM scholarships
    """, conn)
    log.info(f"Database stats:\n{stats.to_string()}")
    conn.close()
    log.info("Verification complete ")


if __name__ == "__main__":
    BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
    CSV_PATH   = os.path.join(BASE_DIR, "data", "scholarships_dataset.csv")
    DB_PATH    = os.path.join(BASE_DIR, "mm_scholar.db")

    df = load_csv(CSV_PATH)
    validate(df)
    convert(df, DB_PATH)
    verify(DB_PATH)
    print("\n mm_scholar.db created successfully. Ready for EDA.")