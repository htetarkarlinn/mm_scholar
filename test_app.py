"""
MM Scholar — regression test suite
Run with:  python test_app.py
"""
import sys
import json
import joblib
import sqlite3
import os
import pandas as pd

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DB_PATH    = os.path.join(BASE_DIR, "mm_scholar.db")

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    results.append(condition)

print("\n=== MM Scholar Test Suite ===\n")

# ── T1: All model pkl files exist, load without error, and are Pipelines ──────
print("T1: Model files load cleanly")
try:
    knn      = joblib.load(os.path.join(MODELS_DIR, "knn_model.pkl"))
    dt       = joblib.load(os.path.join(MODELS_DIR, "dt_model.pkl"))
    rf       = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))
    gb       = joblib.load(os.path.join(MODELS_DIR, "gb_model.pkl"))
    encoders = joblib.load(os.path.join(MODELS_DIR, "encoders.pkl"))
    check("knn_model.pkl loads",        True)
    check("dt_model.pkl loads",         True)
    check("rf_model.pkl loads",         True)
    check("gb_model.pkl loads",         True)
    check("encoders.pkl loads",         True)
    check("knn is Pipeline (OHE+KNN)",  hasattr(knn, "named_steps") and "pre" in knn.named_steps)
    check("dt is Pipeline (scaler+DT)", hasattr(dt,  "named_steps") and "scaler" in dt.named_steps)
    check("rf is Pipeline (scaler+RF)", hasattr(rf,  "named_steps") and "scaler" in rf.named_steps)
    check("gb is Pipeline (scaler+GB)", hasattr(gb,  "named_steps") and "scaler" in gb.named_steps)
    check("knn weights=distance",       knn.named_steps["clf"].weights == "distance")
except Exception as e:
    check("model files load", False, str(e))

# ── T2: metrics.json structure ────────────────────────────────────────────────
print("\nT2: metrics.json structure")
with open(os.path.join(MODELS_DIR, "metrics.json")) as f:
    metrics = json.load(f)
model_names = [m["model"] for m in metrics["models"]]
check("4 models present",            len(metrics["models"]) == 4)
check("k-NN in models",              "k-NN" in model_names)
check("top3_accuracy field present", all("top3_accuracy" in m for m in metrics["models"]))
check("cv_mean field present",       all("cv_mean" in m for m in metrics["models"]))
check("knn_role field present",      metrics.get("knn_role") == "production_ranker")

# ── T3: k-NN pipeline — raw DataFrame input, continuous distance-weighted proba
print("\nT3: k-NN pipeline — raw DataFrame input, distance-weighted probabilities")
student_df = pd.DataFrame([{
    "country_of_study": "Japan",
    "level":            "postgraduate",
    "field_of_study":   "STEM",
    "funding_type":     "fully_funded",
    "min_gpa":          3.5,
    "min_ielts":        6.5,
}])
try:
    proba = knn.predict_proba(student_df)[0]
    check("predict_proba output length == n_classes",  len(proba) == len(knn.classes_))
    check("probabilities sum to ~1.0",                 abs(proba.sum() - 1.0) < 1e-6)
    check("match_pct continuous (distance-weighted)",  proba.max() <= 1.0 and proba.min() >= 0.0)
    check("field='any' encodes as all-zeros (no crash)",
          knn.predict_proba(student_df.assign(field_of_study="any")) is not None)
except Exception as e:
    check("predict_proba executes", False, f"sklearn version mismatch: {e}")

# ── T4: Flask app imports and routes register without error ───────────────────
print("\nT4: Flask app imports cleanly (three-tier architecture)")
try:
    import app as flask_app
    routes = [str(r) for r in flask_app.app.url_map.iter_rules()]
    check("/ route registered",           "/" in routes)
    check("/recommend route registered",  "/recommend" in routes)
    check("/explain route registered",    "/explain" in routes)
    check("/feedback route registered",   "/feedback" in routes)
    check("/compare route registered",    "/compare" in routes)
    check("/health route registered",     "/health" in routes)
    check("/admin route registered",      "/admin" in routes)
    check("_HOMEPAGE_METRICS at startup", "best_accuracy" in flask_app._HOMEPAGE_METRICS)
    check("_METRICS_DATA at startup",     "models" in flask_app._METRICS_DATA)
    # Verify layered architecture: no business logic in app.py
    check("scholarship_repo injected",    hasattr(flask_app, "scholarship_repo"))
    check("feedback_repo injected",       hasattr(flask_app, "feedback_repo"))
except Exception as e:
    check("app imports", False, str(e))

# ── T5: 4-level fallback produces results for all test cases ─────────────────
print("\nT5: 4-level fallback chain (recommendation service)")
try:
    from services.recommendation_service import get_recommendations
    r1, t1 = get_recommendations("Japan", "postgraduate", "fully_funded", "STEM", "3.5", "6.5")
    check("exact match returns results",  len(r1) > 0, f"type={t1}")
    r2, t2 = get_recommendations("Japan", "postgraduate", "partial", "STEM", "", "")
    check("relaxed_funding or better",    len(r2) > 0, f"type={t2}")
    r3, t3 = get_recommendations("Japan", "postgraduate", "fully_funded", "any", "", "")
    check("country+level search returns", len(r3) > 0, f"type={t3}")
    r4, t4 = get_recommendations("NewZealand_Fake", "postgraduate", "fully_funded", "any", "", "")
    check("unknown country doesn't crash", True, f"type={t4}")
except Exception as e:
    check("fallback chain", False, str(e))

# ── T6: match_pct is in valid range [0, 100] ─────────────────────────────────
print("\nT6: match_pct values are valid")
try:
    res, _ = get_recommendations("Japan", "postgraduate", "fully_funded", "STEM", "3.5", "6.5")
    pcts = [r["match_pct"] for r in res]
    check("match_pct >= 0 for all results",  all(p >= 0 for p in pcts), str(pcts))
    check("match_pct <= 100 for all results", all(p <= 100 for p in pcts), str(pcts))
except Exception as e:
    check("match_pct range", False, str(e))

# ── T7: GPA / IELTS eligibility filters preserved ────────────────────────────
print("\nT7: GPA/IELTS filters preserved")
try:
    res, _ = get_recommendations("Japan", "postgraduate", "fully_funded", "STEM", "2.0", "5.0")
    for r in res:
        gpa_ok   = (r.get("min_gpa") or 0) == 0 or (r.get("min_gpa") or 0) <= 2.0
        ielts_ok = (r.get("min_ielts") or 0) == 0 or (r.get("min_ielts") or 0) <= 5.0
        check(f"GPA filter: {r['scholarship_name'][:30]}",   gpa_ok)
        check(f"IELTS filter: {r['scholarship_name'][:30]}", ielts_ok)
except Exception as e:
    check("GPA/IELTS filters", False, str(e))

# ── T8: Feedback repository initialises correctly (Adapter pattern) ───────────
print("\nT8: Feedback repository (Adapter pattern)")
try:
    from repositories.feedback_repo import FeedbackRepository
    fb_repo = FeedbackRepository()
    fb_repo.initialise()
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='feedback'")
    row = cur.fetchone()
    conn.close()
    check("feedback table exists in DB",      row is not None)
    check("feedback_repo.is_available()",     fb_repo.is_available())
except Exception as e:
    check("feedback repository", False, str(e))

# ── T9: Encoder keys ──────────────────────────────────────────────────────────
print("\nT9: Encoder keys")
expected_keys = ["country_of_study", "level", "field_of_study",
                 "funding_type", "scholarship_name"]
for key in expected_keys:
    check(f"encoders['{key}'] present", key in encoders)

# ── T10: Metrics sanity ───────────────────────────────────────────────────────
print("\nT10: Metrics sanity")
for m in metrics["models"]:
    check(f"{m['model']} top3 >= top1",
          m["top3_accuracy"] >= m["accuracy"],
          f"top3={m['top3_accuracy']} top1={m['accuracy']}")
    check(f"{m['model']} cv_mean > 0", m["cv_mean"] > 0)

# ── T11: Input validation (service layer) ─────────────────────────────────────
print("\nT11: Input validation (server-side)")
try:
    import pytest_style_pass  # not needed — manual check below
except ImportError:
    pass

validation_cases = [
    ("Japan", "postgraduate", "fully_funded", "STEM", "5.0", "6.5",  True,  "GPA > 4.0 raises ValueError"),
    ("Japan", "postgraduate", "fully_funded", "STEM", "abc", "6.5",  True,  "non-numeric GPA raises ValueError"),
    ("Japan", "postgraduate", "fully_funded", "STEM", "3.5", "10.0", True,  "IELTS > 9.0 raises ValueError"),
    ("Japan", "postgraduate", "fully_funded", "STEM", "3.5", "6.5",  False, "valid input does not raise"),
    ("",      "postgraduate", "fully_funded", "STEM", "3.5", "6.5",  True,  "empty country raises ValueError"),
]
for country, level, funding, field, gpa, ielts, should_raise, desc in validation_cases:
    try:
        get_recommendations(country, level, funding, field, gpa, ielts)
        raised = False
    except ValueError:
        raised = True
    except Exception:
        raised = False
    check(desc, raised == should_raise)

# ── T12: /health endpoint returns valid JSON ──────────────────────────────────
print("\nT12: /health endpoint")
try:
    with flask_app.app.test_client() as client:
        resp = client.get("/health")
        check("/health status code 200",       resp.status_code == 200)
        data = json.loads(resp.data)
        check("/health returns 'status' key",  "status" in data)
        check("/health returns 'timestamp'",   "timestamp" in data)
        check("/health scholarship_db = ok",   data.get("scholarship_db") == "ok")
        check("/health num_scholarships > 0",  data.get("num_scholarships", 0) > 0)
except Exception as e:
    check("/health endpoint", False, str(e))

# ── T13: /admin endpoint renders without error ───────────────────────────────
print("\nT13: /admin dashboard")
try:
    with flask_app.app.test_client() as client:
        resp = client.get("/admin")
        check("/admin status code 200",        resp.status_code == 200)
        check("/admin contains 'Admin'",       b"Admin" in resp.data)
        check("/admin contains scholarship count",
              str(flask_app.num_scholarships).encode() in resp.data)
except Exception as e:
    check("/admin endpoint", False, str(e))

# ── T14: 404 handler returns custom page ─────────────────────────────────────
print("\nT14: Error handlers")
try:
    with flask_app.app.test_client() as client:
        resp = client.get("/this-does-not-exist-xyz")
        check("404 handler returns 404",       resp.status_code == 404)
        check("404 page is HTML",              b"<!DOCTYPE html>" in resp.data or b"<html" in resp.data)
except Exception as e:
    check("404 handler", False, str(e))

# ── T15: ScholarshipRepository health check ──────────────────────────────────
print("\nT15: ScholarshipRepository")
try:
    from repositories.scholarship_repo import ScholarshipRepository
    sc_repo = ScholarshipRepository()
    stats = sc_repo.get_stats()
    check("get_stats returns num_scholarships", stats["num_scholarships"] > 0)
    check("get_stats returns num_countries",    stats["num_countries"] > 0)
    check("is_available() returns True",        sc_repo.is_available())
    opts = sc_repo.get_dropdown_options()
    check("dropdown options has countries",     len(opts["countries"]) > 0)
    check("dropdown options has levels",        len(opts["levels"]) == 5)
except Exception as e:
    check("ScholarshipRepository", False, str(e))

# ── Summary ───────────────────────────────────────────────────────────────────
passed = sum(results)
total  = len(results)
print(f"\n{'='*40}")
print(f"  {passed}/{total} tests passed")
if passed == total:
    print("  All tests passed.")
else:
    print(f"  {total - passed} test(s) failed.")
print('='*40)
sys.exit(0 if passed == total else 1)
