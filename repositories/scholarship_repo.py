import logging
import sqlite3
import pandas as pd
from config import DB_PATH

logger = logging.getLogger(__name__)

_COLS = (
    "SELECT DISTINCT scholarship_name, provider, country_of_study, level, "
    "field_of_study, funding_type, min_gpa, min_ielts, "
    "deadline_month, duration_years, link FROM scholarships WHERE "
)


class ScholarshipRepository:
    """Data access layer for the scholarship catalogue (read-only SQLite)."""

    def fetch_candidates(self, where_clauses: list, params: list) -> pd.DataFrame:
        conn = sqlite3.connect(DB_PATH)
        try:
            df = pd.read_sql(_COLS + " AND ".join(where_clauses), conn, params=params)
        finally:
            conn.close()
        return df

    def get_all(self) -> pd.DataFrame:
        conn = sqlite3.connect(DB_PATH)
        try:
            df = pd.read_sql("SELECT * FROM scholarships", conn)
        finally:
            conn.close()
        return df

    def get_stats(self) -> dict:
        df = self.get_all()
        return {
            "num_scholarships": int(df["scholarship_name"].nunique()),
            "num_countries":    int(df["country_of_study"].nunique()),
            "num_rows":         len(df),
        }

    def get_dropdown_options(self) -> dict:
        df = self.get_all()
        return {
            "countries":     sorted(df["country_of_study"].unique().tolist()),
            "levels":        ["diploma", "undergraduate", "postgraduate", "phd", "short_course"],
            "fields":        ["STEM", "Business", "Humanities", "Medical", "Education"],
            "funding_types": sorted(df["funding_type"].unique().tolist()),
        }

    def get_region_counts(self, region_map: dict) -> dict:
        df = self.get_all()
        return {
            region: int(df[df["country_of_study"].isin(countries)]["scholarship_name"].nunique())
            for region, countries in region_map.items()
        }

    def is_available(self) -> bool:
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.execute("SELECT 1 FROM scholarships LIMIT 1")
            conn.close()
            return True
        except Exception as exc:
            logger.error("Scholarship DB health check failed: %s", exc)
            return False
