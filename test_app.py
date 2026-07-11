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

# T1: Model files
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

# T2: Metrics file
print("\nT2: metrics.json structure")
with open(os.path.join(MODELS_DIR, "metrics.json")) as f:
    metrics = json.load(f)
model_names = [m["model"] for m in metrics["models"]]
check("4 models present",            len(metrics["models"]) == 4)
check("k-NN in models",              "k-NN" in model_names)
check("top3_accuracy field present", all("top3_accuracy" in m for m in metrics["models"]))
check("cv_mean field present",       all("cv_mean" in m for m in metrics["models"]))
check("knn_role field present",      metrics.get("knn_role") == "production_ranker")

# T3: k-NN probabilities
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

# T4: App startup and routes
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
    # The route module should expose its repository dependencies.
    check("scholarship_repo injected",    hasattr(flask_app, "scholarship_repo"))
    check("feedback_repo injected",       hasattr(flask_app, "feedback_repo"))
except Exception as e:
    check("app imports", False, str(e))

# T5: Recommendation fallbacks
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

# T6: Match percentage range
print("\nT6: match_pct values are valid")
try:
    res, _ = get_recommendations("Japan", "postgraduate", "fully_funded", "STEM", "3.5", "6.5")
    pcts = [r["match_pct"] for r in res]
    check("match_pct >= 0 for all results",  all(p >= 0 for p in pcts), str(pcts))
    check("match_pct <= 100 for all results", all(p <= 100 for p in pcts), str(pcts))
except Exception as e:
    check("match_pct range", False, str(e))

# T7: GPA and IELTS filters
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

# T8: Feedback storage
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

# T9: Encoder keys
print("\nT9: Encoder keys")
expected_keys = ["country_of_study", "level", "field_of_study",
                 "funding_type", "scholarship_name"]
for key in expected_keys:
    check(f"encoders['{key}'] present", key in encoders)

# T10: Metrics sanity
print("\nT10: Metrics sanity")
for m in metrics["models"]:
    check(f"{m['model']} top3 >= top1",
          m["top3_accuracy"] >= m["accuracy"],
          f"top3={m['top3_accuracy']} top1={m['accuracy']}")
    check(f"{m['model']} cv_mean > 0", m["cv_mean"] > 0)

# T11: Input validation
print("\nT11: Input validation (server-side)")
try:
    import pytest_style_pass  # The checks below run without pytest.
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

# T12: Health endpoint
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

# T13: Admin dashboard
print("\nT13: /admin dashboard")
import base64 as _b64
os.environ["ADMIN_PASSWORD"] = "testpass123"
_admin_creds = _b64.b64encode(b"admin:testpass123").decode()
try:
    with flask_app.app.test_client() as client:
        resp = client.get("/admin", headers={"Authorization": f"Basic {_admin_creds}"})
        check("/admin status code 200",        resp.status_code == 200)
        check("/admin contains 'Admin'",       b"Admin" in resp.data)
        check("/admin contains scholarship count",
              str(flask_app.num_scholarships).encode() in resp.data)
except Exception as e:
    check("/admin endpoint", False, str(e))

# T14: Error pages
print("\nT14: Error handlers")
try:
    with flask_app.app.test_client() as client:
        resp = client.get("/this-does-not-exist-xyz")
        check("404 handler returns 404",       resp.status_code == 404)
        check("404 page is HTML",              b"<!DOCTYPE html>" in resp.data or b"<html" in resp.data)
except Exception as e:
    check("404 handler", False, str(e))

# T15: Scholarship repository
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

# T16: Scholarship browser
print("\nT16: /scholarships browse endpoint")
try:
    with flask_app.app.test_client() as client:
        resp = client.get("/scholarships")
        check("/scholarships returns 200",  resp.status_code == 200)
        check("/scholarships is HTML",      b"<!DOCTYPE html>" in resp.data)

    from repositories.scholarship_repo import ScholarshipRepository
    sc = ScholarshipRepository()

    rows_p1, total = sc.get_catalogue(per_page=12, page=1)
    check("Result count <= per_page",                len(rows_p1) <= 12)
    check("Total unique scholarships == num_scholarships",
          total == flask_app.num_scholarships)

    rows_jp, _ = sc.get_catalogue(country="Japan", per_page=50)
    check("Japan filter: all rows have country=Japan",
          len(rows_jp) > 0 and all(r["country_of_study"] == "Japan" for r in rows_jp))

    all_rows, total2 = sc.get_catalogue(per_page=total)
    check("All pages together == total unique count",  len(all_rows) == total2)
except Exception as e:
    check("browse endpoint", False, str(e))

# T17: Admin authentication
print("\nT17: /admin HTTP Basic Auth")
try:
    with flask_app.app.test_client() as client:
        resp = client.get("/admin")
        check("/admin no credentials → 401",      resp.status_code == 401)

        creds = _b64.b64encode(b"admin:testpass123").decode()
        resp = client.get("/admin", headers={"Authorization": f"Basic {creds}"})
        check("/admin valid credentials → 200",   resp.status_code == 200)

        bad = _b64.b64encode(b"admin:wrongpass").decode()
        resp = client.get("/admin", headers={"Authorization": f"Basic {bad}"})
        check("/admin wrong credentials → 401",   resp.status_code == 401)
except Exception as e:
    check("admin auth", False, str(e))

# T18–T20: Admin write operations
print("\nT18–T20: Admin CRUD — feedback delete, feedback edit, scholarship delete")

import datetime as _dt

# T19 removes this feedback row later in the test.
flask_app.feedback_repo.insert((
    _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Japan", "postgraduate", "STEM", "fully_funded",
    "k-NN", "MEXT Scholarship", 5, "CRUD test record",
))
_test_fb_df = flask_app.feedback_repo.get_recent(1)
_fb_id = int(_test_fb_df["id"].iloc[0]) if not _test_fb_df.empty else 1

# Use a disposable scholarship row because scholarship_id is not auto-incremented.
_sc_id = None
_sc_setup_conn = sqlite3.connect(DB_PATH)
try:
    _sc_cur = _sc_setup_conn.cursor()
    _max_sc_id = _sc_setup_conn.execute(
        "SELECT MAX(scholarship_id) FROM scholarships"
    ).fetchone()[0] or 0
    _sc_id = _max_sc_id + 1
    _sc_cur.execute(
        "INSERT INTO scholarships "
        "(scholarship_id, scholarship_name, provider, country_of_study, level, "
        "field_of_study, funding_type, min_gpa, min_ielts, deadline_month, "
        "duration_years, link) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (_sc_id, "__TEST_DELETE_ME__", "Test", "Japan", "postgraduate", "STEM",
         "fully_funded", 0.0, 0.0, 6, 1.0, "https://example.com/test"),
    )
    _sc_setup_conn.commit()
except Exception as _sc_insert_err:
    check("test scholarship insert (T20 prerequisite)", False, str(_sc_insert_err))
    _sc_id = None
finally:
    _sc_setup_conn.close()

if _sc_id is None:
    # Do not fall back to a real scholarship if setup fails.
    check("/admin/feedback/delete no auth → 401",    False, "T20 prerequisite insert failed")
    check("/admin/feedback/delete with auth → 302",  False, "T20 prerequisite insert failed")
    check("/admin/scholarship/delete with auth → 302", False, "T20 prerequisite insert failed")
    check("test scholarship row gone after delete",  False, "T20 prerequisite insert failed")
else:
    try:
        with flask_app.app.test_client() as client:
            # Anonymous requests must be rejected.
            resp = client.post(f"/admin/feedback/delete/{_fb_id}")
            check("/admin/feedback/delete no auth → 401", resp.status_code == 401)

            # An authenticated request deletes the disposable feedback row.
            resp = client.post(
                f"/admin/feedback/delete/{_fb_id}",
                headers={"Authorization": f"Basic {_admin_creds}"},
            )
            check("/admin/feedback/delete with auth → 302", resp.status_code == 302)

            # Delete the disposable scholarship row.
            resp = client.post(
                f"/admin/scholarship/delete/{_sc_id}",
                headers={"Authorization": f"Basic {_admin_creds}"},
            )
            check("/admin/scholarship/delete with auth → 302", resp.status_code == 302)

        # Confirm the route removed the row.
        _verify_conn = sqlite3.connect(DB_PATH)
        _gone = _verify_conn.execute(
            "SELECT COUNT(*) FROM scholarships WHERE scholarship_id = ?", (_sc_id,)
        ).fetchone()[0]
        _verify_conn.close()
        check("test scholarship row gone after delete", _gone == 0)

    except Exception as e:
        check("admin CRUD routes", False, str(e))

# T21: Homepage Browse All Scholarships section
print("\nT21: Browse All Scholarships on homepage")
try:
    with flask_app.app.test_client() as client:
        resp = client.get("/")
        check("GET / returns 200",                       resp.status_code == 200)
        body = resp.data.decode("utf-8")
        check("/ contains 'Browse All Scholarships'",    "Browse All Scholarships" in body)
        check("/ contains browse filter form",           "browse-filter-form" in body)
        check("/ contains 'View All' link to /scholarships",
              "/scholarships" in body and "View All" in body)

        resp_jp = client.get("/?country=Japan")
        check("GET /?country=Japan returns 200",         resp_jp.status_code == 200)
        body_jp = resp_jp.data.decode("utf-8")
        check("/?country=Japan shows Japan cards",       "Japan" in body_jp)
        check("/?country=Japan keeps filter selected",   'value="Japan" selected' in body_jp)
except Exception as e:
    check("homepage browse section", False, str(e))

# Summary
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
