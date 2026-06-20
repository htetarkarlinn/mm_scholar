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

    def get_catalogue(self, country=None, level=None, field=None, funding=None,
                      page=1, per_page=12):
        """Return (rows, total) where each row is one unique scholarship, with
        aggregated levels_available and fields_available strings."""
        where, params = [], []
        if country: where.append("country_of_study = ?"); params.append(country)
        if level:   where.append("level = ?");             params.append(level)
        if field:   where.append("field_of_study = ?");    params.append(field)
        if funding: where.append("funding_type = ?");      params.append(funding)

        where_sql = ("WHERE " + " AND ".join(where)) if where else ""
        offset = (page - 1) * per_page

        count_sql = (
            f"SELECT COUNT(DISTINCT scholarship_name) FROM scholarships {where_sql}"
        )
        data_sql = f"""
            SELECT scholarship_name, provider, country_of_study, funding_type,
                   GROUP_CONCAT(DISTINCT level)          AS levels_available,
                   GROUP_CONCAT(DISTINCT field_of_study) AS fields_available,
                   MIN(min_gpa)        AS min_gpa,
                   MIN(min_ielts)      AS min_ielts,
                   deadline_month, duration_years, link
            FROM scholarships {where_sql}
            GROUP BY scholarship_name
            ORDER BY scholarship_name
            LIMIT ? OFFSET ?
        """
        conn = sqlite3.connect(DB_PATH)
        try:
            total = conn.execute(count_sql, params).fetchone()[0]
            rows  = pd.read_sql(data_sql, conn, params=params + [per_page, offset])
        finally:
            conn.close()
        return rows.to_dict("records"), total

    def get_for_admin(self) -> list:
        """Return all scholarship rows for the admin dashboard table."""
        conn = sqlite3.connect(DB_PATH)
        try:
            df = pd.read_sql(
                "SELECT scholarship_id, scholarship_name, country_of_study, level, "
                "field_of_study, funding_type FROM scholarships "
                "ORDER BY scholarship_name, level",
                conn,
            )
        finally:
            conn.close()
        return df.to_dict("records")

    def delete(self, scholarship_id: int):
        """Delete a single scholarship row by scholarship_id."""
        conn = sqlite3.connect(DB_PATH)
        try:
            conn.execute("DELETE FROM scholarships WHERE scholarship_id = ?", (scholarship_id,))
            conn.commit()
        finally:
            conn.close()

    def is_available(self) -> bool:
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.execute("SELECT 1 FROM scholarships LIMIT 1")
            conn.close()
            return True
        except Exception as exc:
            logger.error("Scholarship DB health check failed: %s", exc)
            return False
