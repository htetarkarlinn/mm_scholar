import os
import json
import sqlite3
import joblib
import datetime
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DB_PATH    = os.path.join(BASE_DIR, "mm_scholar.db")
MODELS_DIR = os.path.join(BASE_DIR, "models")

app      = Flask(__name__)
encoders = joblib.load(os.path.join(MODELS_DIR, "encoders.pkl"))
scaler   = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
knn      = joblib.load(os.path.join(MODELS_DIR, "knn_model.pkl"))
dt       = joblib.load(os.path.join(MODELS_DIR, "dt_model.pkl"))
rf       = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))

FEATURES = ["country_of_study", "level", "field_of_study", "funding_type"]

# load dropdown options from database
conn          = sqlite3.connect(DB_PATH)
df_all        = pd.read_sql("SELECT * FROM scholarships", conn)
conn.close()

countries     = sorted(df_all["country_of_study"].unique())
levels        = ["diploma", "undergraduate", "postgraduate", "phd", "short_course"]
fields        = sorted(df_all["field_of_study"].unique())
funding_types = sorted(df_all["funding_type"].unique())


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
    conn.commit()
    conn.close()

init_feedback_table()


def get_recommendations(country, level, field, funding, model):
    try:
        encoded = [
            encoders["country_of_study"].transform([country])[0],
            encoders["level"].transform([level])[0],
            encoders["field_of_study"].transform([field])[0],
            encoders["funding_type"].transform([funding])[0],
        ]
    except ValueError as e:
        return [], str(e)

    x      = scaler.transform([encoded])
    proba  = model.predict_proba(x)[0]
    top3   = model.classes_[proba.argsort()[::-1][:3]]
    names  = encoders["scholarship_name"].inverse_transform(top3)

    cols = ("SELECT DISTINCT scholarship_name, provider, country_of_study, "
            "level, field_of_study, funding_type, min_gpa, min_ielts, "
            "deadline_month, duration_years, link FROM scholarships ")

    conn    = sqlite3.connect(DB_PATH)
    results = []
    seen    = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        row = pd.read_sql(
            cols + "WHERE scholarship_name = ? "
                   "AND country_of_study = ? AND level = ? AND field_of_study = ?",
            conn, params=(name, country, level, field)
        )
        if row.empty:
            row = pd.read_sql(
                cols + "WHERE scholarship_name = ? AND funding_type = ? LIMIT 1",
                conn, params=(name, funding)
            )
        if row.empty:
            row = pd.read_sql(
                cols + "WHERE scholarship_name = ? LIMIT 1",
                conn, params=(name,)
            )
        if not row.empty:
            results.append(row.iloc[0].to_dict())
    conn.close()
    return results, None


@app.route("/")
def index():
    return render_template("index.html",
                           countries=countries,
                           levels=levels,
                           fields=fields,
                           funding_types=funding_types)


@app.route("/recommend", methods=["POST"])
def recommend():
    country      = request.form.get("country")
    level        = request.form.get("level")
    field        = request.form.get("field")
    funding      = request.form.get("funding")
    model_choice = request.form.get("model", "dt")

    model_map = {"knn": knn, "dt": dt, "rf": rf}
    model     = model_map.get(model_choice, dt)

    results, error = get_recommendations(country, level, field, funding, model)

    recommendation = results[0]["scholarship_name"] if results else "None"

    return render_template("results.html",
                           results=results,
                           error=error,
                           country=country,
                           level=level,
                           field=field,
                           funding=funding,
                           model_choice=model_choice,
                           recommendation=recommendation)


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
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    with open(metrics_path) as f:
        metrics = json.load(f)
    return render_template("compare.html", metrics=metrics)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))