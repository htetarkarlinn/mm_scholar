import os
import json
import datetime
import logging
from flask import Flask, render_template, request, redirect, url_for, jsonify

import config
from config import MODELS_DIR, PORT, DEBUG
from repositories.scholarship_repo import ScholarshipRepository
from repositories.feedback_repo import FeedbackRepository
from services.recommendation_service import get_recommendations
from services.explanation_service import generate_explanation

config.configure_logging()
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ── Bootstrap repositories ────────────────────────────────────────────────────
scholarship_repo = ScholarshipRepository()
feedback_repo    = FeedbackRepository()
feedback_repo.initialise()

# ── Startup data (dropdown options, region counts, metrics) ───────────────────
_opts         = scholarship_repo.get_dropdown_options()
countries     = _opts["countries"]
levels        = _opts["levels"]
fields        = _opts["fields"]
funding_types = _opts["funding_types"]

_stats           = scholarship_repo.get_stats()
num_scholarships = _stats["num_scholarships"]
num_countries    = _stats["num_countries"]
num_rows         = _stats["num_rows"]

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
region_counts = scholarship_repo.get_region_counts(_REGION_COUNTRIES)

with open(os.path.join(MODELS_DIR, "metrics.json")) as _mf:
    _METRICS_DATA = json.load(_mf)
_METRICS_BY_MODEL = {m["model"]: m for m in _METRICS_DATA["models"]}
_HOMEPAGE_METRICS = {
    "best_accuracy":   _METRICS_DATA["best_model_acc"],
    "best_model_name": _METRICS_DATA["best_model_name"],
    "best_model_acc":  _METRICS_DATA["best_model_acc"],
    "dt_accuracy":     _METRICS_BY_MODEL["Decision Tree"]["accuracy"],
    "rf_accuracy":     _METRICS_BY_MODEL["Random Forest"]["accuracy"],
    "knn_accuracy":    _METRICS_BY_MODEL["k-NN"]["accuracy"],
}

logger.info(
    "MM Scholar started — %d scholarships, %d countries, feedback backend: %s",
    num_scholarships, num_countries, "PostgreSQL" if config.USE_PG else "SQLite",
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    recent = feedback_repo.read(
        "SELECT * FROM feedback WHERE comment != '' AND comment IS NOT NULL "
        "ORDER BY timestamp DESC LIMIT 3"
    )
    return render_template("index.html",
                           countries=countries, levels=levels,
                           fields=fields, funding_types=funding_types,
                           metrics=_HOMEPAGE_METRICS,
                           num_scholarships=num_scholarships,
                           num_countries=num_countries,
                           region_counts=region_counts,
                           recent_feedback=recent.to_dict("records"))


@app.route("/recommend", methods=["POST"])
def recommend():
    country = request.form.get("country")
    level   = request.form.get("level")
    funding = request.form.get("funding")
    field   = request.form.get("field", "any") or "any"
    gpa     = request.form.get("gpa", "").strip()
    ielts   = request.form.get("ielts", "").strip()

    try:
        results, match_type = get_recommendations(country, level, funding, field, gpa, ielts)
    except ValueError as exc:
        logger.warning("Invalid recommendation request: %s", exc)
        return render_template("400.html", error=str(exc)), 400

    recommendation = results[0]["scholarship_name"] if results else "None"
    student_profile = {"country": country, "level": level, "field": field,
                       "funding": funding, "gpa": gpa, "ielts": ielts}

    return render_template("results.html",
                           results=results, match_type=match_type,
                           country=country, level=level, field=field,
                           funding=funding, gpa=gpa, ielts=ielts,
                           recommendation=recommendation,
                           student_profile=student_profile)


@app.route("/explain", methods=["POST"])
def explain():
    data            = request.get_json()
    scholarship     = data.get("scholarship")
    student_profile = data.get("student_profile")
    try:
        explanation = generate_explanation(scholarship, student_profile)
        return {"explanation": explanation}
    except Exception as exc:
        logger.error("/explain error: %s", exc)
        return {"explanation": "Explanation unavailable. Please visit the official scholarship website for more details."}, 200


@app.route("/feedback", methods=["POST"])
def feedback():
    feedback_repo.insert((
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        request.form.get("country"),
        request.form.get("level"),
        request.form.get("field"),
        request.form.get("funding"),
        request.form.get("model_used"),
        request.form.get("recommendation"),
        int(request.form.get("rating", 0)),
        request.form.get("comment", "").strip(),
    ))
    return redirect(url_for("thank_you"))


@app.route("/thank-you")
def thank_you():
    return render_template("thank_you.html")


@app.route("/feedback-results")
def feedback_results():
    df = feedback_repo.read("SELECT * FROM feedback ORDER BY timestamp DESC LIMIT 200")
    if df.empty:
        avg_rating, total, ratings = 0, 0, {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    else:
        avg_rating = round(df["rating"].mean(), 2)
        total      = len(df)
        ratings    = df["rating"].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0).to_dict()
    return render_template("feedback_results.html",
                           feedbacks=df.to_dict("records"),
                           avg_rating=avg_rating, total=total, ratings=ratings)


@app.route("/about")
def about():
    return render_template("about.html",
                           num_scholarships=num_scholarships,
                           num_countries=num_countries,
                           num_rows=num_rows)


@app.route("/compare")
def compare():
    return render_template("compare.html",
                           metrics=_METRICS_DATA["models"],
                           num_rows=num_rows,
                           num_scholarships=num_scholarships,
                           num_countries=num_countries)


@app.route("/scholarships")
def browse():
    country = request.args.get("country", "")
    level   = request.args.get("level",   "")
    field   = request.args.get("field",   "")
    funding = request.args.get("funding", "")
    page    = max(1, int(request.args.get("page", 1) or 1))
    per_page = 12

    scholarship_list, total = scholarship_repo.get_catalogue(
        country=country or None,
        level=level     or None,
        field=field     or None,
        funding=funding or None,
        page=page, per_page=per_page,
    )
    total_pages = max(1, (total + per_page - 1) // per_page)

    return render_template("browse.html",
                           scholarships=scholarship_list,
                           total=total, page=page,
                           per_page=per_page, total_pages=total_pages,
                           countries=countries, levels=levels,
                           fields=fields, funding_types=funding_types,
                           current_country=country, current_level=level,
                           current_field=field, current_funding=funding,
                           num_scholarships=num_scholarships)


@app.route("/health")
def health():
    sc_ok = scholarship_repo.is_available()
    fb_ok = feedback_repo.is_available()
    status = "ok" if sc_ok and fb_ok else "degraded"
    code   = 200 if status == "ok" else 503
    return jsonify({
        "status":           status,
        "timestamp":        datetime.datetime.utcnow().isoformat() + "Z",
        "scholarship_db":   "ok" if sc_ok else "error",
        "feedback_db":      "ok" if fb_ok else "error",
        "num_scholarships": num_scholarships,
    }), code


@app.route("/admin")
def admin():
    df = feedback_repo.read("SELECT * FROM feedback ORDER BY timestamp DESC LIMIT 500")
    if df.empty:
        avg_rating, total, ratings = 0, 0, {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    else:
        avg_rating = round(df["rating"].mean(), 2)
        total      = len(df)
        ratings    = df["rating"].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0).to_dict()
    return render_template("admin.html",
                           feedbacks=df.to_dict("records"),
                           avg_rating=avg_rating, total=total, ratings=ratings,
                           num_scholarships=num_scholarships,
                           num_countries=num_countries,
                           models=_METRICS_DATA["models"])


# ── Error handlers ────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(exc):
    return render_template("404.html"), 404


@app.errorhandler(500)
def server_error(exc):
    logger.error("Unhandled server error: %s", exc)
    return render_template("500.html"), 500


if __name__ == "__main__":
    app.run(debug=DEBUG, host="0.0.0.0", port=PORT)
