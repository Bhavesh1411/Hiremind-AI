"""
HireMind AI - Interview Engine (Refactored)
============================================
Modular evaluation engine for the Normal Mode Interview System.

Architecture:
  - question_bank.py  : Static question data + session selection logic
  - interview_engine.py : Evaluation, scoring, and report generation
  - interview_db.py   : All SQLite persistence operations
  - interview_ui.py   : Streamlit UI rendering

Evaluation Strategy:
  - Coding   : Structural keyword detection + safe exec() test case validation
  - Technical: RapidFuzz similarity scoring + keyword presence check
  - Behavioral: RapidFuzz similarity scoring + keyword presence check

Scoring Tiers (per question, max 10 marks):
  80-100 similarity → 9-10 marks
  60-79             → 7-8 marks
  40-59             → 5-6 marks
  Below 40          → 0-4 marks

Future-Ready Hooks:
  - evaluate_coding_judge0() — placeholder for Judge0 API integration
  - generate_ai_questions()  — Gemini AI mode (already implemented separately)
"""

import re
import json
import logging
import random
from typing import List, Dict, Any

try:
    from rapidfuzz import fuzz
except ImportError:
    # Fallback to standard library difflib if rapidfuzz is not installed
    import difflib
    class FallbackFuzz:
        @staticmethod
        def ratio(s1, s2):
            return difflib.SequenceMatcher(None, str(s1), str(s2)).ratio() * 100
        @staticmethod
        def token_sort_ratio(s1, s2):
            # Simple token sort implementation
            t1 = " ".join(sorted(str(s1).split()))
            t2 = " ".join(sorted(str(s2).split()))
            return difflib.SequenceMatcher(None, t1, t2).ratio() * 100
    fuzz = FallbackFuzz
from modules.llm_analysis import call_llm, PROVIDER_GEMINI
from modules.question_bank import CODING_QUESTIONS, select_interview_questions

logger = logging.getLogger("hiremind.interview_engine")

# Keep for backward compat so interview_ui can still import NORMAL_QUESTIONS
NORMAL_QUESTIONS = CODING_QUESTIONS  # will be replaced by select_interview_questions() at runtime


# ════════════════════════════════════════════════════════════════════════════════
#  SCORING HELPERS
# ════════════════════════════════════════════════════════════════════════════════

def _similarity_to_marks(similarity: float) -> int:
    """Convert a 0-100 similarity score to a 0-10 mark scale."""
    if similarity >= 80:
        return random.randint(9, 10)
    elif similarity >= 60:
        return random.randint(7, 8)
    elif similarity >= 40:
        return random.randint(5, 6)
    else:
        return random.randint(0, 4)


def _keyword_boost(user_answer: str, keywords: List[str]) -> float:
    """Returns a 0-20 bonus score based on keyword presence."""
    if not keywords or not user_answer:
        return 0.0
    matched = [k for k in keywords if k.lower() in user_answer.lower()]
    return (len(matched) / len(keywords)) * 20


# ════════════════════════════════════════════════════════════════════════════════
#  CODING EVALUATION
# ════════════════════════════════════════════════════════════════════════════════

def evaluate_coding_normal(user_code: str, question: dict) -> Dict[str, Any]:
    """
    Evaluates a coding answer using two strategies:
      1. Structural keyword detection (accounts for 40% of score)
      2. Safe exec() test case validation (accounts for 60% of score)
    """
    if not user_code or user_code.strip() == "" or "pass" in user_code.split(":")[-1]:
        return {"marks": 0, "similarity": 0, "evaluation": "No solution provided.", "details": []}

    # Security filter
    banned = ["import os", "import sys", "open(", "subprocess", "__import__"]
    for b in banned:
        if b in user_code:
            return {"marks": 0, "similarity": 0, "evaluation": "Security violation: restricted modules detected.", "details": []}

    keywords     = question.get("keywords", [])
    test_cases   = question.get("test_cases", [])
    expected_ans = question.get("expected_answer", "")

    # ── Step 1: Structural Keyword Check ──────────────────────────────────────
    keyword_score = _keyword_boost(user_code, keywords)

    # ── Step 2: RapidFuzz similarity vs expected answer ───────────────────────
    # Normalise whitespace for fairer comparison
    norm_user     = re.sub(r"\s+", " ", user_code.strip())
    norm_expected = re.sub(r"\s+", " ", expected_ans.strip())
    similarity    = fuzz.ratio(norm_user, norm_expected)

    # ── Step 3: Test case execution ──────────────────────────────────────────
    details      = []
    cases_passed = 0
    func_name    = ""

    if test_cases:
        # Extract function name from first def line
        match = re.search(r"def\s+(\w+)\s*\(", user_code)
        func_name = match.group(1) if match else ""

        exec_globals = {}
        try:
            exec(compile(user_code, "<user_code>", "exec"), exec_globals)
        except Exception as e:
            details.append({"error": f"Compilation error: {e}", "passed": False})

        func = exec_globals.get(func_name)
        if func:
            for tc in test_cases:
                try:
                    actual = func(tc["input"])
                    passed = actual == tc["expected"]
                    if passed:
                        cases_passed += 1
                    details.append({
                        "input":    str(tc["input"]),
                        "expected": str(tc["expected"]),
                        "actual":   str(actual),
                        "passed":   passed,
                    })
                except Exception as e:
                    details.append({
                        "input":    str(tc["input"]),
                        "expected": str(tc["expected"]),
                        "error":    str(e),
                        "passed":   False,
                    })

    # ── Step 4: Composite Score ───────────────────────────────────────────────
    test_score = (cases_passed / len(test_cases) * 100) if test_cases else 50
    composite  = (similarity * 0.4) + (test_score * 0.4) + (keyword_score * 0.2)
    composite  = min(100, composite)
    marks      = _similarity_to_marks(composite)

    passed_count = sum(1 for d in details if d.get("passed", False))
    evaluation = (
        f"Passed {passed_count}/{len(test_cases)} test cases. "
        f"Code similarity: {similarity:.0f}%. "
        f"Keyword coverage: {len([k for k in keywords if k.lower() in user_code.lower()])}/{len(keywords)}."
        if test_cases else f"Structural evaluation: {composite:.0f}% score."
    )

    return {
        "marks":      marks,
        "similarity": round(composite, 1),
        "evaluation": evaluation,
        "details":    details,
    }


# ── Future-ready placeholder ──────────────────────────────────────────────────
def evaluate_coding_judge0(user_code: str, question: dict) -> Dict[str, Any]:
    """
    FUTURE: Judge0 API integration for live code execution.
    Structure is ready — implement when Judge0 endpoint is available.
    """
    raise NotImplementedError("Judge0 integration is not implemented in this version.")


# ════════════════════════════════════════════════════════════════════════════════
#  TEXT EVALUATION (Technical + Behavioral)
# ════════════════════════════════════════════════════════════════════════════════

def evaluate_text_normal(user_answer: str, question: dict) -> Dict[str, Any]:
    """
    Evaluates text responses using:
      1. RapidFuzz token_sort_ratio against expected answer (70% weight)
      2. Keyword presence boost (30% weight)
    """
    if not user_answer or len(user_answer.strip()) < 10:
        return {"marks": 0, "similarity": 0, "evaluation": "Answer too short or not provided."}

    expected_answer = question.get("expected_answer", "")
    keywords        = question.get("keywords", [])

    # RapidFuzz: token_sort_ratio handles word-order variations well
    similarity = fuzz.token_sort_ratio(user_answer.lower(), expected_answer.lower())
    kw_boost   = _keyword_boost(user_answer, keywords)

    composite = min(100, (similarity * 0.7) + (kw_boost * 1.5))
    marks     = _similarity_to_marks(composite)

    matched   = [k for k in keywords if k.lower() in user_answer.lower()]
    evaluation = (
        f"Similarity to expected answer: {similarity:.0f}%. "
        f"Keywords matched: {len(matched)}/{len(keywords)} "
        f"({', '.join(matched) if matched else 'none'})."
    )

    return {
        "marks":      marks,
        "similarity": round(composite, 1),
        "evaluation": evaluation,
    }


# ════════════════════════════════════════════════════════════════════════════════
#  UNIFIED EVALUATOR — called per question
# ════════════════════════════════════════════════════════════════════════════════

def evaluate_answer(question: dict, user_answer: str) -> Dict[str, Any]:
    """
    Single entry point for both coding and text evaluation.
    Returns a dict with: marks, similarity, evaluation, [details]
    """
    if question["type"] == "coding":
        return evaluate_coding_normal(user_answer, question)
    else:
        return evaluate_text_normal(user_answer, question)


# ════════════════════════════════════════════════════════════════════════════════
#  FINAL REPORT GENERATOR
# ════════════════════════════════════════════════════════════════════════════════

def generate_normal_report(answers: List[dict]) -> dict:
    """
    Produces a structured final report from the stored answer records.

    Parameters
    ----------
    answers : List of answer dicts from the DB
              (fields: question_text, answer_text, score, evaluation, type)

    Returns
    -------
    dict with total_marks, max_marks, percentage, strengths, weaknesses, breakdown
    """
    total_marks = sum(a.get("score", 0) for a in answers)
    max_marks   = len(answers) * 10

    strengths  = []
    weaknesses = []
    breakdown  = []

    for a in answers:
        score = a.get("score", 0)
        q_short = a.get("question_text", "")[:60] + "..."
        breakdown.append({
            "question":   q_short,
            "type":       a.get("type", "unknown"),
            "marks":      score,
            "evaluation": a.get("evaluation", ""),
        })
        if score >= 7:
            strengths.append(q_short)
        elif score <= 4:
            weaknesses.append(q_short)

    percentage = (total_marks / max_marks * 100) if max_marks > 0 else 0

    return {
        "total_marks": total_marks,
        "max_marks":   max_marks,
        "percentage":  round(percentage, 1),
        "strengths":   strengths,
        "weaknesses":  weaknesses,
        "breakdown":   breakdown,
    }


# ════════════════════════════════════════════════════════════════════════════════
#  AI MODE FUNCTIONS (Gemini) — unchanged from previous implementation
# ════════════════════════════════════════════════════════════════════════════════

def generate_ai_questions(resume_data: dict, api_key: str) -> List[dict]:
    """Generates 7 personalized interview questions using Gemini."""
    skills = resume_data.get("skills", [])
    prompt = f"""
    You are an expert technical interviewer. Generate exactly 7 interview questions for a candidate
    with the following skills: {', '.join(skills) if skills else 'general software engineering'}.

    Structure:
    - 3 Coding Questions (algorithm/logic challenges with a Python starter function)
    - 2 Technical Questions (CS concepts relevant to their skills)
    - 2 Behavioral Questions (situational/culture fit)

    RETURN A JSON LIST with this exact schema per item:
    {{
        "id": "<string>",
        "type": "coding" | "technical" | "behavioral",
        "title": "<short title>",
        "description": "<full question>",
        "starter_code": "<python function stub for coding, null otherwise>",
        "keywords": ["<keyword1>", "<keyword2>"],
        "expected_answer": "<ideal answer text>"
    }}
    Return ONLY raw JSON list. No markdown. No explanation.
    """
    try:
        response  = call_llm(prompt, PROVIDER_GEMINI, api_key)
        clean_res = re.sub(r"^```(?:json)?\s*\n?", "", response.strip())
        clean_res = re.sub(r"\n?```\s*$", "", clean_res).strip()
        data      = json.loads(clean_res)
        return data[:7]
    except Exception as e:
        logger.error("Failed to generate AI questions: %s", e)
        return select_interview_questions()  # Fallback to normal mode


def evaluate_with_ai(question: dict, answer: str, api_key: str) -> Dict[str, Any]:
    """Uses Gemini to evaluate an individual interview answer."""
    prompt = f"""
    You are an expert technical evaluator. Evaluate this interview response:
    QUESTION: {question['description']}
    CANDIDATE ANSWER: {answer}

    Score on 0-10 scale and write a 2-sentence evaluation.
    RETURN JSON: {{"marks": <int>, "evaluation": "<text>"}}
    Return ONLY raw JSON.
    """
    try:
        response  = call_llm(prompt, PROVIDER_GEMINI, api_key)
        clean_res = re.sub(r"^```(?:json)?\s*\n?", "", response.strip())
        clean_res = re.sub(r"\n?```\s*$", "", clean_res).strip()
        data      = json.loads(clean_res)
        # Normalise key name to 'marks'
        if "score" in data and "marks" not in data:
            data["marks"] = data.pop("score")
        return data
    except Exception as e:
        logger.error("AI Evaluation failed: %s", e)
        return {"marks": 0, "evaluation": "AI evaluation failed to process."}


def generate_deep_ai_report(answers: List[dict], api_key: str) -> dict:
    """Generates a Gemini-powered deep performance report."""
    summary_data = "\n\n".join([
        f"Q: {a['question_text']}\nA: {a['answer_text']}\nScore: {a['score']}/10"
        for a in answers
    ])
    prompt = f"""
    Based on these 7 interview responses, provide a deep performance analysis:
    {summary_data}
    Return JSON:
    {{
        "summary": "<overall summary>",
        "strengths": ["<s1>", "<s2>"],
        "weaknesses": ["<w1>", "<w2>"],
        "hireability_verdict": "Strong Hire" | "Hire" | "No Hire",
        "deep_analysis": "<2-3 paragraphs>"
    }}
    """
    try:
        response  = call_llm(prompt, PROVIDER_GEMINI, api_key)
        clean_res = re.sub(r"^```(?:json)?\s*\n?", "", response.strip())
        clean_res = re.sub(r"\n?```\s*$", "", clean_res).strip()
        return json.loads(clean_res)
    except Exception as e:
        logger.error("Deep AI Report failed: %s", e)
        return {
            "summary": "Error generating AI report.",
            "strengths": [], "weaknesses": [],
            "hireability_verdict": "N/A",
            "deep_analysis": str(e),
        }
