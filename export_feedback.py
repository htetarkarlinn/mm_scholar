"""
Export all rows from the PostgreSQL feedback table to a dated CSV file.
Usage: python export_feedback.py
Requires DATABASE_URL env var (set in .env or environment).
"""
import os
import sys
import datetime
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
os.makedirs(BACKUP_DIR, exist_ok=True)

conn = psycopg2.connect(_PG_URL)
df = pd.read_sql("SELECT * FROM feedback ORDER BY timestamp ASC", conn)
conn.close()

today = datetime.date.today().strftime("%Y-%m-%d")
out_path = os.path.join(BACKUP_DIR, f"feedback_{today}.csv")
df.to_csv(out_path, index=False)

print(f"Exported {len(df)} row(s) → {out_path}")
