"""
HireMind AI - ATS Scoring Module
====================================================
Rule-based ATS (Applicant Tracking System) evaluation engine.

Evaluates a resume against a job description using:
    - Section presence checks
    - Keyword density analysis (spaCy-powered)
    - Experience structure validation (chronological detection)
    - Weighted aggregate scoring

NO LLMs, NO embeddings, NO external API calls.
All logic is deterministic and explainable.

Pipeline:
    parsed_json + jd_text
        → check_section_presence()
        → extract_keywords_from_jd()
        → compute_keyword_density()
        → validate_experience_structure()
        → calculate_ats_score()
        → generate_ats_feedback()
        → final report dict
"""

import re
import logging
from typing import Optional

logger = logging.getLogger("hiremind.ats_scorer")

# ── spaCy lazy loader ──────────────────────────────────────────────────────────
_NLP = None

def _get_nlp():
    """Lazy-load spaCy model to avoid startup cost."""
    global _NLP
    if _NLP is None:
        import spacy
        try:
            _NLP = spacy.load("en_core_web_md")
        except OSError:
            _NLP = spacy.load("en_core_web_sm")
        logger.info("spaCy model loaded: %s", _NLP.meta["name"])
    return _NLP


# ── Constants ──────────────────────────────────────────────────────────────────

# Required resume sections and their display names
REQUIRED_SECTIONS = {
    "skills":     "Skills",
    "experience": "Experience",
    "education":  "Education",
    "projects":   "Projects",
}

# Weight distribution for ATS score
WEIGHTS = {
    "section":   0.40,
    "keyword":   0.40,
    "structure": 0.20,
}

# Regex patterns for date detection in experience entries
_DATE_PATTERNS = [
    r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s,]*\d{4}\b",  # "Jan 2023"
    r"\b\d{4}\s*[-–—]\s*(\d{4}|Present|Current|Now)\b",                          # "2020 - 2023"
    r"\b(20\d{2}|19\d{2})\b",                                                    # standalone year
]

# spaCy POS tags that indicate meaningful technical/content words
_MEANINGFUL_POS = {"NOUN", "PROPN", "ADJ", "VERB"}

# Common stopwords to exclude from keyword extraction
_STOPWORDS = {
    "the", "a", "an", "and", "or", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "can", "could", "to", "of", "in", "on", "at",
    "for", "with", "by", "from", "as", "into", "through", "during", "including",
    "that", "this", "these", "those", "we", "you", "they", "it", "its",
    "our", "your", "their", "i", "my", "he", "she", "us", "team", "work",
    "experience", "ability", "strong", "good", "required", "preferred",
    "role", "position", "candidate", "company", "job", "looking",
}


# ══════════════════════════════════════════════════════════════════════════════
#  1. check_section_presence(parsed_json)
# ══════════════════════════════════════════════════════════════════════════════

def check_section_presence(parsed_json: dict) -> dict:
    """
    Checks which key resume sections are present and non-empty.

    Scoring:
        Each present section = 25 points (4 sections total → 100 max)

    Args:
        parsed_json: Structured resume dict from text_processing module.

    Returns:
        {
            "section_score": float (0-100),
            "sections_present": {"skills": True, "experience": False, ...},
            "missing_sections": ["education"],
            "issues": [...str],
        }
    """
    sections_present = {}
    missing = []

    for key, label in REQUIRED_SECTIONS.items():
        value = parsed_json.get(key)
        # A section is present if it's a non-empty string or non-empty list
        if isinstance(value, list):
            is_present = len(value) > 0
        elif isinstance(value, str):
            is_present = bool(value.strip())
        else:
            is_present = False

        sections_present[key] = is_present
        if not is_present:
            missing.append(label)

    present_count = sum(sections_present.values())
    section_score = (present_count / len(REQUIRED_SECTIONS)) * 100

    issues = []
    for label in missing:
        issues.append(f"Missing section: '{label}' not detected in resume.")

    logger.debug(
        "Section check: %d/%d present | score=%.1f",
        present_count, len(REQUIRED_SECTIONS), section_score
    )

    return {
        "section_score":    section_score,
        "sections_present": sections_present,
        "missing_sections": missing,
        "issues":           issues,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  2. extract_keywords_from_jd(jd_text)
# ══════════════════════════════════════════════════════════════════════════════

def extract_keywords_from_jd(jd_text: str) -> list[str]:
    """
    Extracts technical/meaningful keywords from the Job Description using spaCy.

    Steps:
        1. Tokenize with spaCy
        2. Keep only NOUN, PROPN, ADJ tokens
        3. Remove stopwords and very short tokens
        4. Lowercase + deduplicate

    Args:
        jd_text: Raw job description text.

    Returns:
        Sorted list of unique lowercase keywords.
    """
    if not jd_text or not jd_text.strip():
        return []

    nlp = _get_nlp()
    doc = nlp(jd_text[:50000])  # Limit to avoid spaCy memory issues

    keywords = set()
    for token in doc:
        word = token.text.lower().strip()
        if (
            token.pos_ in _MEANINGFUL_POS
            and not token.is_stop
            and word not in _STOPWORDS
            and len(word) >= 3
            and word.isalpha()
        ):
            keywords.add(word)

    # Also extract multi-word noun phrases (e.g., "machine learning", "data science")
    for chunk in doc.noun_chunks:
        phrase = chunk.text.lower().strip()
        words = phrase.split()
        if 2 <= len(words) <= 4:
            # Only keep if all parts are non-stopwords
            if all(w not in _STOPWORDS and len(w) >= 2 for w in words):
                keywords.add(phrase)

    result = sorted(keywords)
    logger.debug("Extracted %d keywords from JD.", len(result))
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  3. compute_keyword_density(resume_data, jd_keywords)
# ══════════════════════════════════════════════════════════════════════════════

def compute_keyword_density(resume_data: dict, jd_keywords: list[str]) -> dict:
    """
    Measures what fraction of JD keywords appear anywhere in the resume.

    Case-insensitive. Searches across: skills, experience, projects, education.

    density = matched_keywords / total_jd_keywords

    Args:
        resume_data:  Parsed resume dict.
        jd_keywords:  Keywords extracted from JD via extract_keywords_from_jd().

    Returns:
        {
            "keyword_score": float (0-100),
            "keyword_density": float (0.0 - 1.0),
            "matched_keywords": [...],
            "unmatched_keywords": [...],
            "total_jd_keywords": int,
        }
    """
    if not jd_keywords:
        return {
            "keyword_score":       50.0,
            "keyword_density":     0.5,
            "matched_keywords":    [],
            "unmatched_keywords":  [],
            "total_jd_keywords":   0,
        }

    # Build a single lowercase corpus from all resume fields
    corpus_parts = []

    skills = resume_data.get("skills", [])
    if isinstance(skills, list):
        corpus_parts.extend([s.lower() for s in skills])

    for field in ("experience", "projects", "education"):
        val = resume_data.get(field, "")
        if isinstance(val, list):
            corpus_parts.extend([str(v).lower() for v in val])
        elif isinstance(val, str):
            corpus_parts.append(val.lower())

    resume_corpus = " ".join(corpus_parts)

    matched = []
    unmatched = []

    for kw in jd_keywords:
        # Use word boundary matching for single-word keywords
        if len(kw.split()) == 1:
            pattern = r"\b" + re.escape(kw) + r"\b"
        else:
            pattern = re.escape(kw)

        if re.search(pattern, resume_corpus, re.IGNORECASE):
            matched.append(kw)
        else:
            unmatched.append(kw)

    density = len(matched) / len(jd_keywords)
    keyword_score = density * 100

    logger.debug(
        "Keyword density: %.2f (%d/%d matched)",
        density, len(matched), len(jd_keywords)
    )

    return {
        "keyword_score":      keyword_score,
        "keyword_density":    round(density, 3),
        "matched_keywords":   matched,
        "unmatched_keywords": unmatched,
        "total_jd_keywords":  len(jd_keywords),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  4. validate_experience_structure(experience_data)
# ══════════════════════════════════════════════════════════════════════════════

def validate_experience_structure(experience_data) -> dict:
    """
    Validates whether the candidate's experience section is ATS-friendly.

    Checks:
        1. Are dates present in experience entries? (required by ATS)
        2. Is the order roughly chronological (most recent first)?
        3. Is there sufficient content per entry?

    Args:
        experience_data: String or list from parsed_json["experience"].

    Returns:
        {
            "structure_score": float (0-100),
            "has_dates":       bool,
            "is_chronological": bool,
            "structure_type":  "chronological" | "functional" | "unknown",
            "issues":          [...str],
        }
    """
    issues = []

    # Normalize to string for analysis
    if isinstance(experience_data, list):
        exp_text = " ".join(str(e) for e in experience_data)
    elif isinstance(experience_data, str):
        exp_text = experience_data
    else:
        exp_text = ""

    if not exp_text.strip():
        return {
            "structure_score":  0.0,
            "has_dates":        False,
            "is_chronological": False,
            "structure_type":   "unknown",
            "issues":           ["Experience section is empty or not detected."],
        }

    # ── Check 1: Date presence ────────────────────────────────────────────────
    combined_date_pattern = "|".join(f"(?:{p})" for p in _DATE_PATTERNS)
    all_dates_found = re.findall(combined_date_pattern, exp_text, re.IGNORECASE)
    has_dates = len(all_dates_found) >= 2

    if not has_dates:
        issues.append(
            "ATS Alert: Dates missing in Experience section. "
            "ATS systems require date ranges (e.g., 'Jan 2021 – Dec 2023')."
        )

    # ── Check 2: Chronological order (years decreasing) ────────────────────
    years = re.findall(r"\b(20\d{2}|19\d{2})\b", exp_text)
    years_int = [int(y) for y in years]

    is_chronological = False
    if len(years_int) >= 2:
        # Chronological = recent first → years should generally be decreasing
        descending_count = sum(
            1 for i in range(len(years_int) - 1)
            if years_int[i] >= years_int[i + 1]
        )
        is_chronological = (descending_count / (len(years_int) - 1)) >= 0.5
    elif len(years_int) == 1:
        is_chronological = True  # Single entry, assume fine

    if not is_chronological and len(years_int) >= 2:
        issues.append(
            "Structure Warning: Experience may not be in reverse-chronological order. "
            "ATS systems prefer the most recent job listed first."
        )

    # ── Check 3: Content depth ─────────────────────────────────────────────
    word_count = len(exp_text.split())
    if word_count < 30:
        issues.append(
            "Thin Content: Experience section has very few words. "
            "Add bullet points with responsibilities and achievements."
        )

    # ── Score calculation ─────────────────────────────────────────────────
    score = 0.0
    if has_dates:
        score += 50.0
    if is_chronological:
        score += 30.0
    if word_count >= 30:
        score += 20.0

    structure_type = "unknown"
    if has_dates and is_chronological:
        structure_type = "chronological"
    elif has_dates and not is_chronological:
        structure_type = "functional"
    elif not has_dates:
        structure_type = "unknown"

    logger.debug(
        "Structure: type=%s | dates=%s | chrono=%s | score=%.1f",
        structure_type, has_dates, is_chronological, score
    )

    return {
        "structure_score":  score,
        "has_dates":        has_dates,
        "is_chronological": is_chronological,
        "structure_type":   structure_type,
        "issues":           issues,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  5. calculate_ats_score(section_score, keyword_score, structure_score)
# ══════════════════════════════════════════════════════════════════════════════

def calculate_ats_score(
    section_score: float,
    keyword_score: float,
    structure_score: float,
) -> float:
    """
    Computes the weighted final ATS score (0–100).

    Formula:
        ATS = 0.40 * section_score
            + 0.40 * keyword_score
            + 0.20 * structure_score

    Args:
        section_score:   0–100 from check_section_presence()
        keyword_score:   0–100 from compute_keyword_density()
        structure_score: 0–100 from validate_experience_structure()

    Returns:
        Float rounded to 1 decimal place, clamped to [0, 100].
    """
    raw = (
        WEIGHTS["section"]   * section_score
        + WEIGHTS["keyword"] * keyword_score
        + WEIGHTS["structure"] * structure_score
    )
    final = max(0.0, min(100.0, raw))
    logger.info(
        "ATS Score: %.1f (section=%.1f, keyword=%.1f, structure=%.1f)",
        final, section_score, keyword_score, structure_score
    )
    return round(final, 1)


# ══════════════════════════════════════════════════════════════════════════════
#  6. generate_ats_feedback(issues, all_data)
# ══════════════════════════════════════════════════════════════════════════════

def generate_ats_feedback(
    ats_score: float,
    section_result: dict,
    keyword_result: dict,
    structure_result: dict,
) -> list[str]:
    """
    Produces a prioritized list of actionable suggestions based on all analysis results.

    Args:
        ats_score:        Final weighted ATS score.
        section_result:   Output of check_section_presence().
        keyword_result:   Output of compute_keyword_density().
        structure_result: Output of validate_experience_structure().

    Returns:
        List of suggestion strings, ordered by priority (most impactful first).
    """
    suggestions = []

    # Priority 1: Missing sections (high impact)
    for missing in section_result.get("missing_sections", []):
        suggestions.append(
            f"🔴 Add a '{missing}' section — ATS systems score resumes "
            f"that include all standard sections much higher."
        )

    # Priority 2: Keyword gaps (high impact)
    density = keyword_result.get("keyword_density", 0)
    unmatched = keyword_result.get("unmatched_keywords", [])
    if density < 0.3:
        suggestions.append(
            f"🔴 Critical Keyword Gap: Only {density*100:.0f}% of JD keywords appear in your "
            f"resume. Add these key terms naturally: "
            + ", ".join(f"'{k}'" for k in unmatched[:8])
            + ("..." if len(unmatched) > 8 else ".")
        )
    elif density < 0.5:
        suggestions.append(
            f"🟡 Improve Keyword Match: {density*100:.0f}% JD coverage detected. "
            "Consider adding more relevant skills. Missing: "
            + ", ".join(f"'{k}'" for k in unmatched[:5])
            + ("..." if len(unmatched) > 5 else ".")
        )
    elif density < 0.7:
        suggestions.append(
            f"🟡 Good keyword match ({density*100:.0f}%). "
            "A few more targeted additions could push you past ATS filters."
        )
    else:
        suggestions.append(
            f"✅ Excellent keyword alignment ({density*100:.0f}%). Your resume aligns well with the JD."
        )

    # Priority 3: Structure issues
    if not structure_result.get("has_dates"):
        suggestions.append(
            "🔴 Add Date Ranges: Experience entries must include date ranges "
            "(e.g., 'Jan 2021 – Dec 2023'). Most ATS systems reject resumes without them."
        )

    if not structure_result.get("is_chronological") and structure_result.get("has_dates"):
        suggestions.append(
            "🟡 Reorder Experience: Switch to reverse-chronological order "
            "(most recent job first). This is the industry-standard ATS format."
        )

    if structure_result.get("structure_type") == "functional":
        suggestions.append(
            "🟡 Consider switching from a Functional format to a Chronological format. "
            "ATS systems parse chronological resumes more accurately."
        )

    # Priority 4: Score-based general tips
    if ats_score < 40:
        suggestions.append(
            "🔴 Overall ATS Score is Low. This resume may be rejected by ATS before a human reads it. "
            "Focus on adding missing sections and improving keyword density first."
        )
    elif ats_score < 60:
        suggestions.append(
            "🟡 Moderate ATS Score. The resume will pass some filters but may miss others. "
            "Address the keyword gaps and formatting issues listed above."
        )
    elif ats_score < 80:
        suggestions.append(
            "🟢 Good ATS Score. Minor optimizations can push this into the 'Excellent' range."
        )
    else:
        suggestions.append(
            "✅ Excellent ATS Score. This resume is well-optimized for automated screening."
        )

    return suggestions


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE: run_ats_analysis(parsed_json, jd_text)
# ══════════════════════════════════════════════════════════════════════════════

def run_ats_analysis(parsed_json: dict, jd_text: str) -> dict:
    """
    Full ATS scoring pipeline. Orchestrates all sub-functions.

    Args:
        parsed_json: Structured resume dict from text_processing module.
        jd_text:     Raw job description text.

    Returns:
        {
            "status": "success" | "error",
            "ats_score": float (0-100),
            "grade": "Excellent" | "Good" | "Fair" | "Poor",
            "formatting_checklist": {
                "sections_present": {...},
                "keyword_density": float,
                "structure_valid": bool,
            },
            "matched_keywords":   [...],
            "unmatched_keywords": [...],
            "issues":             [...],
            "suggestions":        [...],
            "section_score":      float,
            "keyword_score":      float,
            "structure_score":    float,
        }
    """
    result = {
        "status":    "error",
        "ats_score": 0,
        "issues":    [],
        "suggestions": [],
    }

    try:
        if not parsed_json:
            result["issues"] = ["Resume data is empty."]
            return result

        # ── Step 1: Section Presence ─────────────────────────────────────
        section_result = check_section_presence(parsed_json)

        # ── Step 2: Keyword Extraction from JD ───────────────────────────
        jd_keywords = extract_keywords_from_jd(jd_text) if jd_text else []

        # ── Step 3: Keyword Density ───────────────────────────────────────
        keyword_result = compute_keyword_density(parsed_json, jd_keywords)

        # ── Step 4: Experience Structure ──────────────────────────────────
        structure_result = validate_experience_structure(
            parsed_json.get("experience", "")
        )

        # ── Step 5: Weighted ATS Score ────────────────────────────────────
        ats_score = calculate_ats_score(
            section_result["section_score"],
            keyword_result["keyword_score"],
            structure_result["structure_score"],
        )

        # ── Step 6: Grade ─────────────────────────────────────────────────
        if ats_score >= 80:
            grade = "Excellent"
        elif ats_score >= 60:
            grade = "Good"
        elif ats_score >= 40:
            grade = "Fair"
        else:
            grade = "Poor"

        # ── Step 7: Feedback Generation ───────────────────────────────────
        all_issues = (
            section_result["issues"]
            + keyword_result.get("issues", [])
            + structure_result["issues"]
        )
        suggestions = generate_ats_feedback(
            ats_score, section_result, keyword_result, structure_result
        )

        result.update({
            "status":    "success",
            "ats_score": ats_score,
            "grade":     grade,
            "formatting_checklist": {
                "sections_present": section_result["sections_present"],
                "keyword_density":  keyword_result["keyword_density"],
                "structure_valid":  structure_result["structure_type"] == "chronological",
            },
            "matched_keywords":   keyword_result["matched_keywords"],
            "unmatched_keywords": keyword_result["unmatched_keywords"],
            "total_jd_keywords":  keyword_result["total_jd_keywords"],
            "section_score":      round(section_result["section_score"], 1),
            "keyword_score":      round(keyword_result["keyword_score"], 1),
            "structure_score":    round(structure_result["structure_score"], 1),
            "structure_type":     structure_result["structure_type"],
            "has_dates":          structure_result["has_dates"],
            "issues":             all_issues,
            "suggestions":        suggestions,
        })

        logger.info(
            "ATS Analysis complete: score=%.1f grade=%s",
            ats_score, grade
        )

    except Exception as e:
        result["issues"] = [f"ATS analysis failed: {str(e)}"]
        logger.exception("Error in run_ats_analysis.")

    return result
