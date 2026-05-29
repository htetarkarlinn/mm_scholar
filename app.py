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
gemini = genai.GenerativeModel("gemini-1.5-flash")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DB_PATH    = os.path.join(BASE_DIR, "mm_scholar.db")
MODELS_DIR = os.path.join(BASE_DIR, "models")

app      = Flask(__name__)
encoders   = joblib.load(os.path.join(MODELS_DIR, "encoders.pkl"))
scaler     = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
knn        = joblib.load(os.path.join(MODELS_DIR, "knn_model.pkl"))
dt         = joblib.load(os.path.join(MODELS_DIR, "dt_model.pkl"))
rf         = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))
best_model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))

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
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
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
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fb_timestamp ON feedback(timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fb_rating    ON feedback(rating)")
    conn.commit()
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


def _rank(candidates):
    classes_idx = {int(c): i for i, c in enumerate(best_model.classes_)}
    scored = []
    for _, row in candidates.iterrows():
        name = row["scholarship_name"]
        if name not in encoders["scholarship_name"].classes_:
            continue
        target_enc = int(encoders["scholarship_name"].transform([name])[0])
        if target_enc not in classes_idx:
            continue
        try:
            enc = [
                encoders["country_of_study"].transform([row["country_of_study"]])[0],
                encoders["level"].transform([row["level"]])[0],
                encoders["field_of_study"].transform([row["field_of_study"]])[0],
                encoders["funding_type"].transform([row["funding_type"]])[0],
                float(row["min_gpa"] or 0),
                float(row["min_ielts"] or 0),
            ]
            proba = best_model.predict_proba(scaler.transform([enc]))[0]
            scored.append((proba[classes_idx[target_enc]], row))
        except (ValueError, KeyError):
            continue

    scored.sort(key=lambda x: x[0], reverse=True)
    seen, results = set(), []
    for _, row in scored:
        name = row["scholarship_name"]
        if name in seen:
            continue
        seen.add(name)
        record = {k: (None if isinstance(v, float) and pd.isna(v) else v)
                  for k, v in row.to_dict().items()}
        results.append(record)
        if len(results) >= 3:
            break
    return results


def get_recommendations(country, level, funding, field, gpa, ielts):
    gpa   = float(gpa)   if gpa   else 0.0
    ielts = float(ielts) if ielts else 0.0

    opt, opt_p = [], []
    if field and field != "any":
        opt.append("field_of_study = ?")
        opt_p.append(field)
    if gpa > 0:
        opt.append("(min_gpa = 0.0 OR min_gpa <= ?)")
        opt_p.append(gpa)
    if ielts > 0:
        opt.append("(min_ielts = 0.0 OR min_ielts <= ?)")
        opt_p.append(ielts)

    # Level 1 — exact match
    candidates = _fetch(
        ["country_of_study = ?", "level = ?", "funding_type = ?"] + opt,
        [country, level, funding] + opt_p
    )
    if not candidates.empty:
        results = _rank(candidates)
        if results:
            return results, "exact"

    # Level 2 — relax funding_type
    candidates = _fetch(
        ["country_of_study = ?", "level = ?"] + opt,
        [country, level] + opt_p
    )
    if not candidates.empty:
        results = _rank(candidates)
        if results:
            return results, "relaxed_funding"

    # Level 3 — country only
    candidates = _fetch(["country_of_study = ?"], [country])
    if not candidates.empty:
        results = _rank(candidates)
        if results:
            return results, "country_only"

    # Level 4 — popular fallback (fully_funded + same level, any country)
    candidates = _fetch(["funding_type = ?", "level = ?"], ["fully_funded", level])
    if not candidates.empty:
        results = _rank(candidates)
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
        return {"explanation": "Explanation unavailable. Please visit the official scholarship website for more details."}, 200


@app.route("/")
def index():
    return render_template("index.html",
                           countries=countries,
                           levels=levels,
                           fields=fields,
                           funding_types=funding_types,
                           metrics=load_metrics(),
                           num_scholarships=num_scholarships,
                           num_countries=num_countries)


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
    _m = load_metrics()

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
                           best_model_name=_m["best_model_name"],
                           best_model_acc=_m["best_model_acc"],
                           student_profile=student_profile)


@app.route("/feedback", methods=["POST"])
def feedback():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO feedback
            (timestamp, country, level, field, funding,
             model_used, recommendation, rating, comment)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
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
    conn.commit()
    conn.close()
    return redirect(url_for("thank_you"))


@app.route("/thank-you")
def thank_you():
    return render_template("thank_you.html")


@app.route("/feedback-results")
def feedback_results():
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql("SELECT * FROM feedback ORDER BY timestamp DESC", conn)
    conn.close()

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


@app.route("/compare")
def compare():
    with open(os.path.join(MODELS_DIR, "metrics.json")) as f:
        data = json.load(f)
    return render_template("compare.html", metrics=data["models"])


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))