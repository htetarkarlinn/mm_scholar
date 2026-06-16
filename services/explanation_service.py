import logging
import google.generativeai as genai
from config import GEMINI_API_KEY, GEMINI_MODEL
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

genai.configure(api_key=GEMINI_API_KEY)
_gemini = genai.GenerativeModel(GEMINI_MODEL)

# In-memory response cache keyed by (scholarship_name, country, level, funding, field, gpa, ielts).
# Eliminates redundant API calls for repeated identical queries.
_cache: dict = {}


def generate_explanation(scholarship: dict, student: dict) -> str:
    cache_key = (
        scholarship.get("scholarship_name"),
        student.get("country"),
        student.get("level"),
        student.get("funding"),
        student.get("field"),
        str(student.get("gpa", "")),
        str(student.get("ielts", "")),
    )
    if cache_key in _cache:
        logger.info("Explanation cache hit for %s", scholarship.get("scholarship_name"))
        return _cache[cache_key]

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
    logger.info("Calling Gemini API for %s", scholarship.get("scholarship_name"))
    response = _gemini.generate_content(prompt)
    _cache[cache_key] = response.text
    return response.text
