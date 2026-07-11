import logging
import google.generativeai as genai
from config import GEMINI_API_KEY, GEMINI_MODEL
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

genai.configure(api_key=GEMINI_API_KEY)


class ExplanationService:
    def __init__(self):
        self._gemini = genai.GenerativeModel(GEMINI_MODEL)
        self._personalised_cache: dict = {}
        self._general_cache: dict = {}
        self._view_counts: dict = {}

    def generate_explanation(self, scholarship: dict, student: dict) -> str:
        scholarship_name = scholarship.get("scholarship_name", "")

        personalised_key = (
            scholarship_name,
            student.get("country"),
            student.get("level"),
            student.get("funding"),
            student.get("field"),
            str(student.get("gpa", "")),
            str(student.get("ielts", "")),
        )

        self._view_counts[scholarship_name] = (
            self._view_counts.get(scholarship_name, 0) + 1
        )

        if personalised_key in self._personalised_cache:
            logger.info("Personalised cache hit for %s", scholarship_name)
            return self._personalised_cache[personalised_key]

        if scholarship_name in self._general_cache:
            logger.info("General cache hit for %s", scholarship_name)
            return self._general_cache[scholarship_name]

        explanation = self._call_gemini(scholarship, student)
        self._personalised_cache[personalised_key] = explanation
        self._general_cache[scholarship_name] = explanation
        return explanation

    def _call_gemini(self, scholarship: dict, student: dict) -> str:
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
        response = self._gemini.generate_content(prompt)
        return response.text

    def get_popular_scholarships(self, n: int = 10) -> list:
        sorted_views = sorted(
            self._view_counts.items(), key=lambda x: x[1], reverse=True
        )
        return [name for name, _ in sorted_views[:n]]

    def get_cache_stats(self) -> dict:
        return {
            "personalised_cache_size": len(self._personalised_cache),
            "general_cache_size":      len(self._general_cache),
            "total_views":             sum(self._view_counts.values()),
            "top_scholarships":        self.get_popular_scholarships(5),
        }
