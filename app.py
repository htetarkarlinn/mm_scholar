import os
import json
import sqlite3
import joblib
import datetime
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, jsonify
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
gemini = genai.GenerativeModel("gemini-2.5-flash")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DB_PATH    = os.path.join(BASE_DIR, "mm_scholar.db")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ── Feedback DB: SQLite locally, PostgreSQL on Render / Heroku ───────────────
# Both platforms inject DATABASE_URL automatically when a Postgres addon is
# attached.  Heroku uses the legacy "postgres://" scheme; normalize it.
_PG_URL = os.environ.get("DATABASE_URL", "")
if _PG_URL.startswith("postgres://"):
    _PG_URL = _PG_URL.replace("postgres://", "postgresql://", 1)
_USE_PG = bool(_PG_URL)

if _USE_PG:
    import psycopg2
    def _fb_conn():
        return psycopg2.connect(_PG_URL)
    _PH     = "%s"
    _ID_DEF = "id SERIAL PRIMARY KEY"
else:
    def _fb_conn():
        return sqlite3.connect(DB_PATH)
    _PH     = "?"
    _ID_DEF = "id INTEGER PRIMARY KEY AUTOINCREMENT"

def _fb_read(query):
    conn = _fb_conn()
    df   = pd.read_sql(query, conn)
    conn.close()
    return df

def _fb_insert(vals):
    conn = _fb_conn()
    cur  = conn.cursor()
    cur.execute(
        f"INSERT INTO feedback "
        f"(timestamp, country, level, field, funding, model_used, recommendation, rating, comment) "
        f"VALUES ({_PH},{_PH},{_PH},{_PH},{_PH},{_PH},{_PH},{_PH},{_PH})",
        vals
    )
    conn.commit()
    cur.close()
    conn.close()

app      = Flask(__name__)
encoders   = joblib.load(os.path.join(MODELS_DIR, "encoders.pkl"))
scaler     = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
knn        = joblib.load(os.path.join(MODELS_DIR, "knn_model.pkl"))
dt         = joblib.load(os.path.join(MODELS_DIR, "dt_model.pkl"))
rf         = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))
# best_model.pkl kept for compare-page metrics only; knn is the production ranker

# load dropdown options from database
conn          = sqlite3.connect(DB_PATH)
df_all        = pd.read_sql("SELECT * FROM scholarships", conn)
conn.close()

countries     = sorted(df_all["country_of_study"].unique())
levels        = ["diploma", "undergraduate", "postgraduate", "phd", "short_course"]
fields        = ["STEM", "Business", "Humanities", "Medical", "Education"]
funding_types = sorted(df_all["funding_type"].unique())

num_scholarships = int(df_all["scholarship_name"].nunique())
num_countries    = int(df_all["country_of_study"].nunique())
num_rows         = len(df_all)

_REGION_COUNTRIES = {
    "Asia":       ["Japan","South Korea","China","Singapore","Thailand","India",
                   "Malaysia","Vietnam","Indonesia","Philippines","Taiwan"],
    "Europe":     ["UK","Germany","Sweden","Hungary","Belgium","France","Netherlands",
                   "Italy","Spain","Finland","Norway","Denmark","Switzerland",
                   "Austria","Czech Republic","Poland"],
    "Americas":   ["USA","Canada"],
    "MiddleEast": ["Turkey","UAE","Saudi Arabia","Qatar"],
    "Pacific":    ["Australia","New Zealand","Hong Kong"],
}
region_counts = {
    r: int(df_all[df_all["country_of_study"].isin(c)]["scholarship_name"].nunique())
    for r, c in _REGION_COUNTRIES.items()
}

def load_metrics():
    with open(os.path.join(MODELS_DIR, "metrics.json")) as _f:
        _data     = json.load(_f)
        _by_model = {m["model"]: m for m in _data["models"]}
        return {
            "best_accuracy":   _data["best_model_acc"],
            "best_model_name": _data["best_model_name"],
            "best_model_acc":  _data["best_model_acc"],
            "dt_accuracy":     _by_model["Decision Tree"]["accuracy"],
            "rf_accuracy":     _by_model["Random Forest"]["accuracy"],
            "knn_accuracy":    _by_model["k-NN"]["accuracy"],
        }


def init_feedback_table():
    conn = _fb_conn()
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

init_feedback_table()


_COLS = ("SELECT DISTINCT scholarship_name, provider, country_of_study, level, "
         "field_of_study, funding_type, min_gpa, min_ielts, "
         "deadline_month, duration_years, link FROM scholarships WHERE ")


def _fetch(where, params):
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql(_COLS + " AND ".join(where), conn, params=params)
    conn.close()
    return df


def _rank(candidates, student):
    """Rank SQL candidates by k-NN similarity to the student profile.

    The student's preferences are encoded as a feature vector in the same
    space as the training data.  k-NN predict_proba gives the fraction of
    the k nearest training neighbours that belong to each scholarship class —
    a genuine similarity score derived from the student side, not from
    feeding each scholarship's own features back through the model.
    """
    gpa   = float(student.get("gpa")   or 0)
    ielts = float(student.get("ielts") or 0)
    field = student.get("field", "any")

    try:
        # For "any" field, use the midpoint of label-encoded field values so
        # no single field dominates similarity; SQL already filtered upstream.
        field_enc = (int(encoders["field_of_study"].transform([field])[0])
                     if field and field != "any"
                     else int(len(encoders["field_of_study"].classes_) / 2))
        student_vec = [
            int(encoders["country_of_study"].transform([student["country"]])[0]),
            int(encoders["level"].transform([student["level"]])[0]),
            field_enc,
            int(encoders["funding_type"].transform([student["funding"]])[0]),
            gpa,
            ielts,
        ]
    except (ValueError, KeyError):
        # Encoding failed (unseen label) — return candidates unranked
        seen, results = set(), []
        for _, row in candidates.iterrows():
            name = row["scholarship_name"]
            if name in seen:
                continue
            seen.add(name)
            record = {k: (None if isinstance(v, float) and pd.isna(v) else v)
                      for k, v in row.to_dict().items()}
            record["match_pct"] = 0.0
            results.append(record)
            if len(results) >= 3:
                break
        return results

    # One predict_proba call for the student profile
    proba = knn.predict_proba(scaler.transform([student_vec]))[0]
    classes_idx = {int(c): i for i, c in enumerate(knn.classes_)}

    scored = []
    for _, row in candidates.iterrows():
        name = row["scholarship_name"]
        if name not in encoders["scholarship_name"].classes_:
            continue
        target_enc = int(encoders["scholarship_name"].transform([name])[0])
        if target_enc not in classes_idx:
            continue
        match_score = float(proba[classes_idx[target_enc]])
        record = {k: (None if isinstance(v, float) and pd.isna(v) else v)
                  for k, v in row.to_dict().items()}
        record["match_pct"] = round(match_score * 100, 1)
        scored.append((match_score, record))

    scored.sort(key=lambda x: x[0], reverse=True)
    seen, results = set(), []
    for _, record in scored:
        name = record["scholarship_name"]
        if name in seen:
            continue
        seen.add(name)
        results.append(record)
        if len(results) >= 3:
            break
    return results


def get_recommendations(country, level, funding, field, gpa, ielts):
    gpa   = float(gpa)   if gpa   else 0.0
    ielts = float(ielts) if ielts else 0.0

    student = {
        "country": country,
        "level":   level,
        "field":   field,
        "funding": funding,
        "gpa":     gpa,
        "ielts":   ielts,
    }

    # Eligibility filters — applied at every fallback level so students
    # are never shown scholarships whose GPA/IELTS requirement they can't meet.
    score_opt, score_p = [], []
    if gpa > 0:
        score_opt.append("(min_gpa = 0.0 OR min_gpa <= ?)")
        score_p.append(gpa)
    if ielts > 0:
        score_opt.append("(min_ielts = 0.0 OR min_ielts <= ?)")
        score_p.append(ielts)

    # Preference filters (field) — only applied at levels 1 & 2 where the
    # search is already scoped tightly enough.
    pref_opt = list(score_opt)
    pref_p   = list(score_p)
    if field and field != "any":
        pref_opt.insert(0, "field_of_study = ?")
        pref_p.insert(0, field)

    # Level 1 — exact match
    candidates = _fetch(
        ["country_of_study = ?", "level = ?", "funding_type = ?"] + pref_opt,
        [country, level, funding] + pref_p
    )
    if not candidates.empty:
        results = _rank(candidates, student)
        if results:
            return results, "exact"

    # Level 2 — relax funding_type
    candidates = _fetch(
        ["country_of_study = ?", "level = ?"] + pref_opt,
        [country, level] + pref_p
    )
    if not candidates.empty:
        results = _rank(candidates, student)
        if results:
            return results, "relaxed_funding"

    # Level 3 — country only (score filters kept, field/funding relaxed)
    candidates = _fetch(["country_of_study = ?"] + score_opt, [country] + score_p)
    if not candidates.empty:
        results = _rank(candidates, student)
        if results:
            return results, "country_only"

    # Level 4 — popular fallback (score filters kept)
    candidates = _fetch(["funding_type = ?", "level = ?"] + score_opt,
                        ["fully_funded", level] + score_p)
    if not candidates.empty:
        results = _rank(candidates, student)
        if results:
            return results, "popular_fallback"

    return [], "no_results"


def generate_explanation(scholarship, student):
    prompt = f"""
You are a scholarship advisor helping a Myanmar student.

Student is looking for:
- Country: {student['country']}
- Level: {student['level']}
- Funding: {student['funding']}
- Field: {student.get('field', 'not specified')}
- GPA: {student.get('gpa', 'not provided')}
- IELTS: {student.get('ielts', 'not provided')}

Scholarship details:
- Name: {scholarship['scholarship_name']}
- Provider: {scholarship['provider']}
- Country: {scholarship['country_of_study']}
- Level: {scholarship['level']}
- Field: {scholarship['field_of_study']}
- Funding: {scholarship['funding_type']}
- Min GPA: {scholarship['min_gpa']}
- Min IELTS: {scholarship['min_ielts']}
- Deadline: month {int(scholarship['deadline_month']) if scholarship.get('deadline_month') else 'not specified'}
- Duration: {scholarship['duration_years']} years

Write a short explanation with exactly 3 sections:

WHY THIS MATCHES YOU
(2 sentences — explain why this scholarship suits this student specifically)

WHAT YOU NEED TO APPLY
(3-4 bullet points — practical requirements)

THINGS TO CHECK
(1-2 warnings — important eligibility notes for Myanmar students)

Keep total response under 150 words.
Be encouraging and specific.
Write for a Myanmar student applying for the first time.
"""
    response = gemini.generate_content(prompt)
    return response.text


@app.route("/explain", methods=["POST"])
def explain():
    data            = request.get_json()
    scholarship     = data.get("scholarship")
    student_profile = data.get("student_profile")
    try:
        explanation = generate_explanation(scholarship, student_profile)
        return {"explanation": explanation}
    except Exception as e:
        app.logger.error(f"/explain error: {e}")
        return {"explanation": "Explanation unavailable. Please visit the official scholarship website for more details."}, 200


@app.route("/")
def index():
    fb_df = _fb_read(
        "SELECT * FROM feedback WHERE comment != '' AND comment IS NOT NULL "
        "ORDER BY timestamp DESC LIMIT 3"
    )
    return render_template("index.html",
                           countries=countries,
                           levels=levels,
                           fields=fields,
                           funding_types=funding_types,
                           metrics=load_metrics(),
                           num_scholarships=num_scholarships,
                           num_countries=num_countries,
                           region_counts=region_counts,
                           recent_feedback=fb_df.to_dict("records"))


@app.route("/recommend", methods=["POST"])
def recommend():
    country      = request.form.get("country")
    level        = request.form.get("level")
    funding      = request.form.get("funding")
    field        = request.form.get("field", "any") or "any"
    gpa          = request.form.get("gpa", "").strip()
    ielts        = request.form.get("ielts", "").strip()
    model_choice = request.form.get("model", "dt")

    results, match_type = get_recommendations(country, level, funding, field, gpa, ielts)

    recommendation = results[0]["scholarship_name"] if results else "None"

    student_profile = {
        "country": country,
        "level":   level,
        "field":   field,
        "funding": funding,
        "gpa":     gpa,
        "ielts":   ielts,
    }

    return render_template("results.html",
                           results=results,
                           match_type=match_type,
                           country=country,
                           level=level,
                           field=field,
                           funding=funding,
                           gpa=gpa,
                           ielts=ielts,
                           recommendation=recommendation,
                           student_profile=student_profile)


@app.route("/feedback", methods=["POST"])
def feedback():
    _fb_insert((
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        request.form.get("country"),
        request.form.get("level"),
        request.form.get("field"),
        request.form.get("funding"),
        request.form.get("model_used"),
        request.form.get("recommendation"),
        int(request.form.get("rating", 0)),
        request.form.get("comment", "").strip()
    ))
    return redirect(url_for("thank_you"))


@app.route("/thank-you")
def thank_you():
    return render_template("thank_you.html")


@app.route("/feedback-results")
def feedback_results():
    df = _fb_read("SELECT * FROM feedback ORDER BY timestamp DESC")

    if df.empty:
        avg_rating = 0
        total      = 0
        ratings    = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    else:
        avg_rating = round(df["rating"].mean(), 2)
        total      = len(df)
        ratings    = df["rating"].value_counts().reindex([1,2,3,4,5], fill_value=0).to_dict()

    return render_template("feedback_results.html",
                           feedbacks=df.to_dict("records"),
                           avg_rating=avg_rating,
                           total=total,
                           ratings=ratings)


@app.route("/about")
def about():
    return render_template("about.html",
                           num_scholarships=num_scholarships,
                           num_countries=num_countries,
                           num_rows=num_rows)


@app.route("/compare")
def compare():
    with open(os.path.join(MODELS_DIR, "metrics.json")) as f:
        data = json.load(f)
    return render_template("compare.html",
                           metrics=data["models"],
                           num_rows=num_rows,
                           num_scholarships=num_scholarships,
                           num_countries=num_countries)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5001)))