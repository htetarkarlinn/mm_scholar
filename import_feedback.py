"""
Import the most recent feedback CSV from feedback_backups/ into PostgreSQL,
skipping rows that already exist (deduped on timestamp + recommendation).
Usage: python import_feedback.py
Requires DATABASE_URL env var (set in .env or environment).
"""
import os
import sys
import glob
import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()

_PG_URL = os.environ.get("DATABASE_URL", "")
if _PG_URL.startswith("postgres://"):
    _PG_URL = _PG_URL.replace("postgres://", "postgresql://", 1)

if not _PG_URL:
    print("ERROR: DATABASE_URL is not set.")
    sys.exit(1)

BACKUP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback_backups")
csvs = sorted(glob.glob(os.path.join(BACKUP_DIR, "feedback_*.csv")))
if not csvs:
    print(f"No CSV files found in {BACKUP_DIR}/")
    sys.exit(1)

src = csvs[-1]
print(f"Reading: {src}")
df = pd.read_csv(src)

conn = psycopg2.connect(_PG_URL)
cur = conn.cursor()

cur.execute("""
    CREATE TABLE IF NOT EXISTS feedback (
        id              SERIAL PRIMARY KEY,
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
conn.commit()

cur.execute("SELECT timestamp, recommendation FROM feedback")
existing = set(cur.fetchall())

imported = 0
for _, row in df.iterrows():
    key = (str(row.get("timestamp", "") or ""), str(row.get("recommendation", "") or ""))
    if key in existing:
        continue
    cur.execute(
        "INSERT INTO feedback "
        "(timestamp, country, level, field, funding, model_used, recommendation, rating, comment) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
        (
            row.get("timestamp"),
            row.get("country"),
            row.get("level"),
            row.get("field"),
            row.get("funding"),
            row.get("model_used"),
            row.get("recommendation"),
            int(row["rating"]) if pd.notna(row.get("rating")) else None,
            row.get("comment"),
        )
    )
    existing.add(key)
    imported += 1

conn.commit()
cur.close()
conn.close()

print(f"Imported {imported} new row(s) from {os.path.basename(src)} (skipped {len(df) - imported} duplicate(s))")
