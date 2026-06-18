import os
import logging

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DB_PATH    = os.path.join(BASE_DIR, "mm_scholar.db")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Feedback database — SQLite locally, PostgreSQL on Render via DATABASE_URL
_PG_URL = os.environ.get("DATABASE_URL", "")
if _PG_URL.startswith("postgres://"):
    _PG_URL = _PG_URL.replace("postgres://", "postgresql://", 1)
USE_PG = bool(_PG_URL)
PG_URL = _PG_URL

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-2.5-flash"

ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")
SECRET_KEY     = os.environ.get("SECRET_KEY", "mm-scholar-dev-key-2026")

PORT        = int(os.environ.get("PORT", 5001))
DEBUG       = os.environ.get("FLASK_DEBUG", "0") == "1"

VALID_LEVELS   = ["diploma", "undergraduate", "postgraduate", "phd", "short_course"]
VALID_FIELDS   = ["STEM", "Business", "Humanities", "Medical", "Education"]
GPA_MIN, GPA_MAX     = 0.0, 4.0
IELTS_MIN, IELTS_MAX = 0.0, 9.0


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
