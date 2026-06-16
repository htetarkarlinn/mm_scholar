import logging
import joblib
import pandas as pd
from config import MODELS_DIR, GPA_MIN, GPA_MAX, IELTS_MIN, IELTS_MAX
from repositories.scholarship_repo import ScholarshipRepository
import os

logger = logging.getLogger(__name__)

_knn      = joblib.load(os.path.join(MODELS_DIR, "knn_model.pkl"))
_encoders = joblib.load(os.path.join(MODELS_DIR, "encoders.pkl"))
_repo     = ScholarshipRepository()


def _validate_profile(country, level, funding, field, gpa, ielts):
    errors = []
    if not country:
        errors.append("Country of study is required.")
    if not level:
        errors.append("Study level is required.")
    if not funding:
        errors.append("Funding type is required.")
    if gpa:
        try:
            v = float(gpa)
            if not (GPA_MIN <= v <= GPA_MAX):
                errors.append(f"GPA must be between {GPA_MIN} and {GPA_MAX}.")
        except ValueError:
            errors.append("GPA must be a number.")
    if ielts:
        try:
            v = float(ielts)
            if not (IELTS_MIN <= v <= IELTS_MAX):
                errors.append(f"IELTS must be between {IELTS_MIN} and {IELTS_MAX}.")
        except ValueError:
            errors.append("IELTS must be a number.")
    if errors:
        raise ValueError(" ".join(errors))


def _rank(candidates: pd.DataFrame, student: dict) -> list:
    """Score SQL candidates by k-NN distance-weighted probability, return top 3."""
    gpa   = float(student.get("gpa")   or 0)
    ielts = float(student.get("ielts") or 0)
    field = student.get("field", "any")

    student_df = pd.DataFrame([{
        "country_of_study": student["country"],
        "level":            student["level"],
        "field_of_study":   field if field and field != "any" else "any",
        "funding_type":     student["funding"],
        "min_gpa":          gpa,
        "min_ielts":        ielts,
    }])

    try:
        proba = _knn.predict_proba(student_df)[0]
    except Exception as exc:
        logger.warning("k-NN predict_proba failed, returning unranked: %s", exc)
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

    classes_idx = {int(c): i for i, c in enumerate(_knn.classes_)}

    scored = []
    for _, row in candidates.iterrows():
        name = row["scholarship_name"]
        if name not in _encoders["scholarship_name"].classes_:
            continue
        target_enc = int(_encoders["scholarship_name"].transform([name])[0])
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
    """
    Four-level fallback chain: exact → relax funding → country only → popular fallback.
    Returns (results_list, match_type_str).
    Raises ValueError for invalid inputs.
    """
    _validate_profile(country, level, funding, field, gpa, ielts)

    gpa   = float(gpa)   if gpa   else 0.0
    ielts = float(ielts) if ielts else 0.0

    student = {"country": country, "level": level, "field": field,
               "funding": funding, "gpa": gpa, "ielts": ielts}

    score_opt, score_p = [], []
    if gpa > 0:
        score_opt.append("(min_gpa = 0.0 OR min_gpa <= ?)")
        score_p.append(gpa)
    if ielts > 0:
        score_opt.append("(min_ielts = 0.0 OR min_ielts <= ?)")
        score_p.append(ielts)

    pref_opt = list(score_opt)
    pref_p   = list(score_p)
    if field and field != "any":
        pref_opt.insert(0, "field_of_study = ?")
        pref_p.insert(0, field)

    # Level 1 — exact match
    candidates = _repo.fetch_candidates(
        ["country_of_study = ?", "level = ?", "funding_type = ?"] + pref_opt,
        [country, level, funding] + pref_p,
    )
    if not candidates.empty:
        results = _rank(candidates, student)
        if results:
            logger.info("Recommendation: exact match (%d candidates)", len(candidates))
            return results, "exact"

    # Level 2 — relax funding type
    candidates = _repo.fetch_candidates(
        ["country_of_study = ?", "level = ?"] + pref_opt,
        [country, level] + pref_p,
    )
    if not candidates.empty:
        results = _rank(candidates, student)
        if results:
            logger.info("Recommendation: relaxed funding (%d candidates)", len(candidates))
            return results, "relaxed_funding"

    # Level 3 — country only
    candidates = _repo.fetch_candidates(
        ["country_of_study = ?"] + score_opt, [country] + score_p,
    )
    if not candidates.empty:
        results = _rank(candidates, student)
        if results:
            logger.info("Recommendation: country-only fallback (%d candidates)", len(candidates))
            return results, "country_only"

    # Level 4 — popular fully-funded fallback
    candidates = _repo.fetch_candidates(
        ["funding_type = ?", "level = ?"] + score_opt,
        ["fully_funded", level] + score_p,
    )
    if not candidates.empty:
        results = _rank(candidates, student)
        if results:
            logger.info("Recommendation: popular fallback (%d candidates)", len(candidates))
            return results, "popular_fallback"

    logger.warning("Recommendation: no results for profile %s", student)
    return [], "no_results"
