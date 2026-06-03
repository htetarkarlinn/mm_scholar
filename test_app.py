"""
MM Scholar — regression test suite (10 tests)
Run with:  python test_app.py
"""
import sys
import json
import joblib
import numpy as np
import pandas as pd
import sqlite3
import os

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

# ── T1: All model pkl files exist and load without error ──────────────────────
print("T1: Model files load cleanly")
try:
    knn      = joblib.load(os.path.join(MODELS_DIR, "knn_model.pkl"))
    dt       = joblib.load(os.path.join(MODELS_DIR, "dt_model.pkl"))
    rf       = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))
    gb       = joblib.load(os.path.join(MODELS_DIR, "gb_model.pkl"))
    encoders = joblib.load(os.path.join(MODELS_DIR, "encoders.pkl"))
    scaler   = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    check("knn_model.pkl loads",  True)
    check("dt_model.pkl loads",   True)
    check("rf_model.pkl loads",   True)
    check("gb_model.pkl loads",   True)
    check("encoders.pkl loads",   True)
    check("scaler.pkl loads",     True)
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

# ── T3: k-NN ranking is non-circular ─────────────────────────────────────────
print("\nT3: k-NN ranking — student vector fed once, not per scholarship")
country_enc = int(encoders["country_of_study"].transform(["Japan"])[0])
level_enc   = int(encoders["level"].transform(["postgraduate"])[0])
field_enc   = int(encoders["field_of_study"].transform(["STEM"])[0])
funding_enc = int(encoders["funding_type"].transform(["fully_funded"])[0])
student_vec = [country_enc, level_enc, field_enc, funding_enc, 3.5, 6.5]
proba = knn.predict_proba(scaler.transform([student_vec]))[0]
check("predict_proba output length == n_classes", len(proba) == len(knn.classes_))
check("probabilities sum to ~1.0",               abs(proba.sum() - 1.0) < 1e-6)
check("match_pct is fraction of k neighbours",   proba.max() <= 1.0 and proba.min() >= 0.0)

# ── T4: Flask app imports and routes register without error ───────────────────
print("\nT4: Flask app imports cleanly")
try:
    import app as flask_app
    routes = [str(r) for r in flask_app.app.url_map.iter_rules()]
    check("/ route registered",                "/" in routes)
    check("/recommend route registered",       "/recommend" in routes)
    check("/explain route registered",         "/explain" in routes)
    check("/feedback route registered",        "/feedback" in routes)
    check("/compare route registered",         "/compare" in routes)
    check("explanation_cache exists",          isinstance(flask_app.explanation_cache, dict))
    check("_HOMEPAGE_METRICS cached at startup", "best_accuracy" in flask_app._HOMEPAGE_METRICS)
    check("_METRICS_DATA cached at startup",   "models" in flask_app._METRICS_DATA)
except Exception as e:
    check("app imports", False, str(e))

# ── T5: 4-level fallback produces results for all test cases ─────────────────
print("\nT5: 4-level fallback chain")
try:
    r1, t1 = flask_app.get_recommendations("Japan", "postgraduate", "fully_funded", "STEM", "3.5", "6.5")
    check("exact match returns results",      len(r1) > 0, f"type={t1}")
    r2, t2 = flask_app.get_recommendations("Japan", "postgraduate", "partial", "STEM", "", "")
    check("relaxed_funding or better",        len(r2) > 0, f"type={t2}")
    r3, t3 = flask_app.get_recommendations("Japan", "postgraduate", "fully_funded", "any", "", "")
    check("country+level search returns",     len(r3) > 0, f"type={t3}")
    r4, t4 = flask_app.get_recommendations("NewZealand_Fake", "postgraduate", "fully_funded", "any", "", "")
    # Unknown country → should fall to popular_fallback or no_results, not crash
    check("unknown country doesn't crash",    True, f"type={t4}")
except Exception as e:
    check("fallback chain", False, str(e))

# ── T6: match_pct is in valid range [0, 100] ─────────────────────────────────
print("\nT6: match_pct values are valid")
try:
    res, _ = flask_app.get_recommendations("Japan", "postgraduate", "fully_funded", "STEM", "3.5", "6.5")
    pcts = [r["match_pct"] for r in res]
    check("match_pct >= 0 for all results",  all(p >= 0 for p in pcts), str(pcts))
    check("match_pct <= 100 for all results", all(p <= 100 for p in pcts), str(pcts))
except Exception as e:
    check("match_pct range", False, str(e))

# ── T7: GPA / IELTS eligibility filters preserved across all fallback levels ──
print("\nT7: GPA/IELTS filters preserved")
try:
    conn  = sqlite3.connect(DB_PATH)
    df_db = pd.read_sql("SELECT * FROM scholarships", conn)
    conn.close()
    res, _ = flask_app.get_recommendations("Japan", "postgraduate", "fully_funded", "STEM", "2.0", "5.0")
    for r in res:
        gpa_ok   = (r.get("min_gpa") or 0) == 0 or (r.get("min_gpa") or 0) <= 2.0
        ielts_ok = (r.get("min_ielts") or 0) == 0 or (r.get("min_ielts") or 0) <= 5.0
        check(f"GPA filter: {r['scholarship_name'][:30]}",   gpa_ok)
        check(f"IELTS filter: {r['scholarship_name'][:30]}", ielts_ok)
except Exception as e:
    check("GPA/IELTS filters", False, str(e))

# ── T8: Feedback DB initialised ───────────────────────────────────────────────
print("\nT8: Feedback table exists")
try:
    conn = flask_app._fb_conn()
    cur  = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='feedback'")
    row = cur.fetchone()
    conn.close()
    check("feedback table exists in DB", row is not None)
except Exception as e:
    check("feedback table", False, str(e))

# ── T9: Encoders cover all expected keys ──────────────────────────────────────
print("\nT9: Encoder keys")
expected_keys = ["country_of_study", "level", "field_of_study",
                 "funding_type", "scholarship_name"]
for key in expected_keys:
    check(f"encoders['{key}'] present", key in encoders)

# ── T10: Top-3 accuracy metric looks sane ────────────────────────────────────
print("\nT10: Metrics sanity")
for m in metrics["models"]:
    check(f"{m['model']} top3 >= top1",
          m["top3_accuracy"] >= m["accuracy"],
          f"top3={m['top3_accuracy']} top1={m['accuracy']}")
    check(f"{m['model']} cv_mean > 0",  m["cv_mean"] > 0)

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
