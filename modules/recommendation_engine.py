"""
HireMind AI - Recommendation Engine Module
====================================================
Identifies skill gaps between a candidate and a job description,
then generates personalized learning paths, project ideas,
certification suggestions, and alternative role recommendations.

Pipeline:
    candidate_skills + jd_skills
        → identify_skill_gaps()
        → map_skills_to_learning_paths()
        → generate_learning_recommendations()
        → suggest_alternative_roles()
        → generate_improvement_summary()
        → build_recommendation_output()

NO FAISS, NO embeddings. Knowledge-graph driven + optional LLM.
"""

import logging
from typing import Optional

logger = logging.getLogger("hiremind.recommendation")


# ══════════════════════════════════════════════════════════════════════════════
#  SKILL ANALYSIS UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
# No hardcoded knowledge graph used to ensure 100% data-driven output.


# ══════════════════════════════════════════════════════════════════════════════
#  1. identify_skill_gaps(candidate_skills, jd_skills)
# ══════════════════════════════════════════════════════════════════════════════

def identify_skill_gaps(
    candidate_skills: list[str],
    jd_skills: list[str],
) -> dict:
    """
    Compare candidate skills against JD requirements.
    Strictly data-driven: JD Skills - Resume Skills = Missing Skills.
    """
    if not candidate_skills and not jd_skills:
        return {
            "status": "error",
            "message": "Unable to extract skills from resume due to formatting issues."
        }

    cand_lower = {s.lower().strip() for s in candidate_skills if s.strip()}
    jd_lower   = {s.lower().strip() for s in jd_skills if s.strip()}

    matched = set()
    for jd_skill in jd_lower:
        for cand_skill in cand_lower:
            # Case-insensitive exact and substring matching
            if jd_skill == cand_skill or jd_skill in cand_skill or cand_skill in jd_skill:
                matched.add(jd_skill)
                break

    missing = jd_lower - matched
    match_rate = len(matched) / len(jd_lower) if jd_lower else 1.0

    return {
        "status":         "success",
        "matched_skills": sorted(matched),
        "missing_skills": sorted(missing),
        "match_rate":     round(match_rate, 3),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  2. build_recommendation_output() — Full Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def build_recommendation_output(
    candidate_skills: list[str],
    jd_skills: list[str],
    **kwargs  # Accept extra args for compatibility but ignore them
) -> dict:
    """
    Skill Gap Analysis orchestrator. 
    Strictly 100% based on real resume and JD data.
    """
    gap_data = identify_skill_gaps(candidate_skills, jd_skills)
    
    if gap_data.get("status") == "error":
        return gap_data

    missing = gap_data["missing_skills"]
    
    summary = ""
    if not missing:
        summary = "No significant skill gaps found."
    else:
        summary = f"The candidate is missing {len(missing)} key skill(s) required for this role."

    return {
        "status":               "success",
        "missing_skills":       missing,
        "matched_skills":       gap_data["matched_skills"],
        "match_rate":           gap_data["match_rate"],
        "improvement_summary":  summary,
        "recommended_learning": [], # No dummy courses
        "alternative_roles":    [], # No dummy roles
    }
