import logging
import sqlite3
import pandas as pd
from config import DB_PATH, USE_PG, PG_URL

logger = logging.getLogger(__name__)

# Pick the connection and placeholder style once when this module loads.
if USE_PG:
    import psycopg2
    def _conn():
        return psycopg2.connect(PG_URL)
    _PH     = "%s"
    _ID_DEF = "id SERIAL PRIMARY KEY"
else:
    def _conn():
        return sqlite3.connect(DB_PATH)
    _PH     = "?"
    _ID_DEF = "id INTEGER PRIMARY KEY AUTOINCREMENT"


class FeedbackRepository:
    """Store feedback in SQLite or PostgreSQL."""

    def initialise(self):
        conn = _conn()
        cur  = conn.cursor()
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS feedback (
                {_ID_DEF},
                timestamp       TEXT,
                country         TEXT,
                level           TEXT,
                field           TEXT,
                funding         TEXT,
                model_used      TEXT,
                recommendation  TEXT,
                rating          INTEGER,
                comment         TEXT
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_fb_timestamp ON feedback(timestamp)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_fb_rating    ON feedback(rating)")
        conn.commit()
        cur.close()
        conn.close()
        logger.info("Feedback table initialised (backend: %s)", "PostgreSQL" if USE_PG else "SQLite")

    def insert(self, vals: tuple):
        conn = _conn()
        cur  = conn.cursor()
        cur.execute(
            f"INSERT INTO feedback "
            f"(timestamp, country, level, field, funding, model_used, recommendation, rating, comment) "
            f"VALUES ({_PH},{_PH},{_PH},{_PH},{_PH},{_PH},{_PH},{_PH},{_PH})",
            vals,
        )
        conn.commit()
        cur.close()
        conn.close()

    def read(self, query: str) -> pd.DataFrame:
        conn = _conn()
        df   = pd.read_sql(query, conn)
        conn.close()
        return df

    def get_by_id(self, record_id: int):
        conn = _conn()
        df   = pd.read_sql(f"SELECT * FROM feedback WHERE id = {int(record_id)}", conn)
        conn.close()
        return df.to_dict("records")[0] if not df.empty else None

    def delete(self, record_id: int):
        conn = _conn()
        cur  = conn.cursor()
        cur.execute(f"DELETE FROM feedback WHERE id = {_PH}", (record_id,))
        conn.commit()
        cur.close()
        conn.close()

    def update(self, record_id: int, rating: int, comment: str):
        conn = _conn()
        cur  = conn.cursor()
        cur.execute(
            f"UPDATE feedback SET rating = {_PH}, comment = {_PH} WHERE id = {_PH}",
            (rating, comment, record_id),
        )
        conn.commit()
        cur.close()
        conn.close()

    def get_recent_with_comments(self, n: int = 3) -> pd.DataFrame:
        conn = _conn()
        df   = pd.read_sql(
            f"SELECT * FROM feedback WHERE comment != '' AND comment IS NOT NULL "
            f"ORDER BY timestamp DESC LIMIT {int(n)}",
            conn,
        )
        conn.close()
        return df

    def get_recent(self, limit: int = 200) -> pd.DataFrame:
        conn = _conn()
        df   = pd.read_sql(
            f"SELECT * FROM feedback ORDER BY timestamp DESC LIMIT {int(limit)}",
            conn,
        )
        conn.close()
        return df

    def get_all_for_admin(self, limit: int = 500) -> pd.DataFrame:
        conn = _conn()
        df   = pd.read_sql(
            f"SELECT * FROM feedback ORDER BY timestamp DESC LIMIT {int(limit)}",
            conn,
        )
        conn.close()
        return df

    def is_available(self) -> bool:
        try:
            self.read("SELECT 1 FROM feedback LIMIT 1")
            return True
        except Exception as exc:
            logger.error("Feedback DB health check failed: %s", exc)
            return False
