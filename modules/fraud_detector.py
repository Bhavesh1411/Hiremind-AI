"""
HireMind AI - Fraud Detection Module
====================================================
Detects potential resume fraud, exaggeration, and manipulation using
rule-based heuristics and optional LLM validation.

Checks performed:
    1. Skill Inconsistency  — listed skills not demonstrated anywhere
    2. Keyword Stuffing     — unnaturally repeated words
    3. Hidden Text Patterns — keyword blocks invisible in context
    4. Temporal Consistency — overlapping or impossible date ranges
    5. (Optional) LLM Validation — AI reasoning on suspicious patterns

Output:
    {
        "fraud_risk":    "Low" | "Medium" | "High",
        "risk_score":    int (0-100),
        "flags":         [...],
        "flag_details":  {category: [flags]},
        "llm_verdict":   str | None,
    }

NO embeddings, NO FAISS. Fully deterministic and explainable.
"""

import re
import logging
from collections import Counter
from typing import Optional

logger = logging.getLogger("hiremind.fraud_detector")

# ── Thresholds ─────────────────────────────────────────────────────────────────
KEYWORD_STUFFING_THRESHOLD = 8    # flag if a single word appears > N times in raw text
HIDDEN_TEXT_RATIO_THRESHOLD = 2.5 # flag if raw word count >>> parsed word count by this ratio
MIN_SKILL_EVIDENCE_WORDS = 3      # minimum chars a skill must match in body to be "evidenced"

# ── Common English words to ignore in stuffing detection ──────────────────────
_IGNORE_WORDS = {
    "the", "a", "an", "and", "or", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "shall", "should", "may", "might", "can", "could", "to",
    "of", "in", "on", "at", "for", "with", "by", "from", "as", "into",
    "that", "this", "these", "those", "we", "you", "they", "it", "its",
    "my", "our", "your", "their", "i", "he", "she", "us", "who", "which",
    "work", "worked", "working", "use", "used", "using", "also", "other",
    "team", "company", "project", "role", "year", "month", "time", "new",
    "experience", "skill", "skills", "responsible", "developed", "managed",
}

# ── Date parsing helpers ───────────────────────────────────────────────────────
_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

def _parse_year_month(text: str) -> Optional[tuple[int, int]]:
    """Try to extract (year, month) from a text fragment. Returns None if unparseable."""
    text = text.lower().strip()
    # "Jan 2022" or "January 2022"
    m = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{4})", text)
    if m:
        return int(m.group(2)), _MONTH_MAP[m.group(1)]
    # "2022"
    m = re.search(r"\b(20\d{2}|19\d{2})\b", text)
    if m:
        return int(m.group(1)), 1
    return None


def _to_months(year: int, month: int) -> int:
    return year * 12 + month


# ══════════════════════════════════════════════════════════════════════════════
#  1. check_skill_inconsistency(parsed_json)
# ══════════════════════════════════════════════════════════════════════════════

def check_skill_inconsistency(parsed_json: dict) -> list[str]:
    """
    Checks if listed skills are actually evidenced in the resume body.

    Logic:
        - Build a corpus from experience + projects + education
        - For each listed skill, check if it appears in the corpus
        - Flag skills that have zero mentions

    Args:
        parsed_json: Structured resume dict.

    Returns:
        List of flag strings for inconsistent skills.
    """
    flags = []

    skills = parsed_json.get("skills", [])
    if not skills:
        return flags

    # Build body corpus from non-skills sections
    body_parts = []
    for field in ("experience", "projects", "education"):
        val = parsed_json.get(field, "")
        if isinstance(val, list):
            body_parts.extend(str(v) for v in val)
        elif isinstance(val, str):
            body_parts.append(val)

    body_corpus = " ".join(body_parts).lower()

    if len(body_corpus.strip()) < 50:
        # Not enough body text to make a judgment
        return flags

    undemonstrated = []
    for skill in skills:
        skill_lower = skill.lower().strip()
        if len(skill_lower) < MIN_SKILL_EVIDENCE_WORDS:
            continue  # Skip very short tokens (e.g., "C", "R")

        # Match the skill as a word/phrase in the body
        pattern = r"\b" + re.escape(skill_lower) + r"\b"
        if not re.search(pattern, body_corpus):
            undemonstrated.append(skill)

    if undemonstrated:
        flag_msg = (
            f"Skill Inconsistency: {len(undemonstrated)} skill(s) listed "
            f"but not evidenced in experience/projects: "
            + ", ".join(f'"{s}"' for s in undemonstrated[:6])
            + ("..." if len(undemonstrated) > 6 else ".")
        )
        flags.append(flag_msg)
        logger.debug("Skill inconsistency: %d undemonstrated skills.", len(undemonstrated))

    return flags


# ══════════════════════════════════════════════════════════════════════════════
#  2. detect_keyword_stuffing(raw_text)
# ══════════════════════════════════════════════════════════════════════════════

def detect_keyword_stuffing(raw_text: str) -> list[str]:
    """
    Detects unnaturally repeated keywords in the raw resume text.

    Logic:
        - Tokenize and count word frequencies
        - Flag any non-stopword appearing more than THRESHOLD times
        - Compute density as count / total_words × 100

    Args:
        raw_text: Full raw extracted text from the resume.

    Returns:
        List of flag strings for stuffed keywords.
    """
    flags = []

    if not raw_text or len(raw_text.strip()) < 100:
        return flags

    words = re.findall(r"\b[a-zA-Z]{3,}\b", raw_text.lower())
    total_words = len(words)
    if total_words == 0:
        return flags

    freq = Counter(w for w in words if w not in _IGNORE_WORDS)
    stuffed = [
        (word, count)
        for word, count in freq.most_common(20)
        if count > KEYWORD_STUFFING_THRESHOLD
    ]

    for word, count in stuffed:
        density = (count / total_words) * 100
        if density > 3.0:  # Only flag if it's also a high density
            flags.append(
                f"Keyword Stuffing: '{word}' appears {count} times "
                f"({density:.1f}% of resume text). "
                "Possible keyword padding for ATS manipulation."
            )

    if stuffed:
        logger.debug("Keyword stuffing detected: %s", [(w, c) for w, c in stuffed])

    return flags


# ══════════════════════════════════════════════════════════════════════════════
#  3. detect_hidden_text(raw_text, parsed_data)
# ══════════════════════════════════════════════════════════════════════════════

def detect_hidden_text(raw_text: str, parsed_data: dict) -> list[str]:
    """
    Detects possible hidden/white text patterns using heuristics.

    Heuristics:
        1. Raw word count is significantly larger than parsed content word count
           (hidden text inflates raw but not structured parsed output)
        2. Sudden repetitive keyword blocks detected via regex
        3. Unusually high ratio of punctuation-free keyword lists

    Args:
        raw_text:    Full raw text extracted from the resume file.
        parsed_data: Structured parsed resume dict.

    Returns:
        List of flag strings for hidden text suspicion.
    """
    flags = []

    if not raw_text:
        return flags

    raw_words = re.findall(r"\b\w+\b", raw_text)
    raw_count = len(raw_words)

    # Build parsed corpus word count
    parsed_parts = []
    for field in ("name", "email", "phone", "location", "skills",
                  "experience", "projects", "education"):
        val = parsed_data.get(field, "")
        if isinstance(val, list):
            parsed_parts.extend(str(v) for v in val)
        elif isinstance(val, str):
            parsed_parts.append(val)
    parsed_words = re.findall(r"\b\w+\b", " ".join(parsed_parts))
    parsed_count = len(parsed_words)

    # Heuristic 1: Raw >> Parsed (≥ 2.5x ratio)
    if parsed_count > 0:
        ratio = raw_count / parsed_count
        if ratio >= HIDDEN_TEXT_RATIO_THRESHOLD:
            flags.append(
                f"Hidden Text Suspected: Raw extracted text is {ratio:.1f}× larger than parsed content. "
                "This may indicate hidden white text or invisible keyword blocks in the original file."
            )
            logger.debug("Hidden text ratio: %.2f", ratio)

    # Heuristic 2: Repetitive keyword blocks (same word repeated 5+ times in a row)
    repetitive_blocks = re.findall(r"\b(\w{4,})\b(?:\s+\1\b){4,}", raw_text, re.IGNORECASE)
    if repetitive_blocks:
        flags.append(
            f"Hidden Text Pattern: Repetitive keyword blocks detected "
            f"({', '.join(set(w.lower() for w in repetitive_blocks[:3]))}). "
            "These may be invisible ATS spam keywords."
        )

    # Heuristic 3: Very long lines with no punctuation (raw keyword dumps)
    lines = raw_text.split("\n")
    suspicious_lines = [
        line for line in lines
        if len(line.split()) > 20
        and not re.search(r"[,.;:\-–()]", line)
        and len(line.strip()) > 0
    ]
    if len(suspicious_lines) >= 3:
        flags.append(
            f"Possible Keyword Dump: {len(suspicious_lines)} line(s) found with 20+ words "
            "and no punctuation. May be a hidden keyword section for ATS manipulation."
        )

    return flags


# ══════════════════════════════════════════════════════════════════════════════
#  4. check_temporal_consistency(experience_data)
# ══════════════════════════════════════════════════════════════════════════════

def check_temporal_consistency(experience_data) -> list[str]:
    """
    Analyzes work experience date ranges for inconsistencies.

    Checks:
        1. Overlapping job roles (two full-time jobs at same time)
        2. Impossible timelines (end date before start date)
        3. Suspiciously long tenures (>15 years in a single role)
        4. Future dates in past experience

    Args:
        experience_data: String or list from parsed_json["experience"].

    Returns:
        List of flag strings for temporal issues.
    """
    flags = []

    if isinstance(experience_data, list):
        exp_text = " ".join(str(e) for e in experience_data)
    elif isinstance(experience_data, str):
        exp_text = experience_data
    else:
        return flags

    if not exp_text.strip():
        return flags

    # Extract date ranges: "Jan 2020 – Dec 2022" or "2020 – 2022" or "2020 - Present"
    range_pattern = re.compile(
        r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}|\d{4})"
        r"\s*[-–—]\s*"
        r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}|\d{4}|Present|Current|Now)",
        re.IGNORECASE,
    )

    ranges = range_pattern.findall(exp_text)
    parsed_ranges = []

    import datetime
    current_year = datetime.datetime.now().year

    for start_str, end_str in ranges:
        start = _parse_year_month(start_str)
        if end_str.lower() in ("present", "current", "now"):
            end = (current_year, 12)
        else:
            end = _parse_year_month(end_str)

        if start and end:
            s_months = _to_months(*start)
            e_months = _to_months(*end)

            # Check: end before start
            if e_months < s_months:
                flags.append(
                    f"Timeline Error: End date '{end_str}' is before start date '{start_str}'. "
                    "This is an impossible date range."
                )
                continue

            # Check: future start date
            if start[0] > current_year:
                flags.append(
                    f"Future Date Detected: Start date '{start_str}' is in the future. "
                    "Experience entries should not have future start dates."
                )
                continue

            # Check: suspiciously long tenure (> 15 years)
            tenure_years = (e_months - s_months) / 12
            if tenure_years > 15:
                flags.append(
                    f"Suspicious Tenure: A role spans {tenure_years:.0f} years "
                    f"({start_str} – {end_str}). Verify this is accurate."
                )

            parsed_ranges.append((s_months, e_months, f"{start_str}–{end_str}"))

    # Check overlapping ranges (potential simultaneous full-time roles)
    overlaps = []
    for i in range(len(parsed_ranges)):
        for j in range(i + 1, len(parsed_ranges)):
            s1, e1, label1 = parsed_ranges[i]
            s2, e2, label2 = parsed_ranges[j]
            # Overlap = one starts before the other ends
            overlap_months = min(e1, e2) - max(s1, s2)
            if overlap_months > 3:  # allow 3 months grace (transitions / part-time)
                overlaps.append((label1, label2, overlap_months))

    for label1, label2, months in overlaps[:3]:  # Report max 3 overlaps
        flags.append(
            f"Overlapping Employment: '{label1}' and '{label2}' overlap by "
            f"~{months} months. Simultaneous full-time roles may need clarification."
        )

    if parsed_ranges:
        logger.debug("Temporal check: %d ranges found, %d overlaps.", len(parsed_ranges), len(overlaps))

    return flags


# ══════════════════════════════════════════════════════════════════════════════
#  5. compute_fraud_risk(flags)
# ══════════════════════════════════════════════════════════════════════════════

# Severity weights by flag category keyword
_FLAG_SEVERITY = {
    "Hidden Text":       3,   # High severity
    "Timeline Error":    3,
    "Keyword Stuffing":  2,   # Medium severity
    "Overlapping":       2,
    "Future Date":       2,
    "Skill Inconsistency": 1, # Low severity (common in entry-level)
    "Suspicious Tenure": 1,
    "Keyword Dump":      2,
}

def compute_fraud_risk(flags: list[str]) -> tuple[str, int]:
    """
    Calculates a fraud risk level and numeric score from the detected flags.

    Each flag is weighted based on its category severity.
    Score is normalized to 0-100.

    Args:
        flags: List of flag strings from all detection functions.

    Returns:
        Tuple of (risk_level: str, risk_score: int)
        risk_level: "Low" | "Medium" | "High"
        risk_score: 0-100
    """
    if not flags:
        return "Low", 0

    total_weight = 0
    for flag in flags:
        matched = False
        for keyword, weight in _FLAG_SEVERITY.items():
            if keyword.lower() in flag.lower():
                total_weight += weight
                matched = True
                break
        if not matched:
            total_weight += 1  # Default weight for unknown flag type

    # Map weight to score (cap at 100)
    raw_score = min(total_weight * 15, 100)

    if raw_score <= 20:
        risk_level = "Low"
    elif raw_score <= 55:
        risk_level = "Medium"
    else:
        risk_level = "High"

    logger.info("Fraud risk: %s (score=%d, flags=%d)", risk_level, raw_score, len(flags))
    return risk_level, raw_score


# ══════════════════════════════════════════════════════════════════════════════
#  5b. Optional LLM Validation
# ══════════════════════════════════════════════════════════════════════════════

def validate_with_llm(
    raw_text: str,
    flags: list[str],
    api_key: str,
    provider: str = "gemini",
) -> Optional[str]:
    """
    Optional: Uses the configured LLM to validate suspicious patterns.

    Only called if api_key is provided AND flags are already detected.
    Prompt is concise to minimize token usage.

    Args:
        raw_text: Raw resume text (truncated to 3000 chars to save tokens).
        flags:    Already detected rule-based flags.
        api_key:  Gemini or OpenAI API key.
        provider: "gemini" or "openai".

    Returns:
        LLM verdict string, or None if skipped/failed.
    """
    if not api_key or not flags:
        return None

    try:
        from modules.llm_analysis import call_llm

        flag_summary = "\n".join(f"- {f}" for f in flags)
        prompt = f"""You are an expert resume fraud analyst.

The following automated flags were detected in a candidate's resume:
{flag_summary}

Here is an excerpt from the resume (first 3000 characters):
\"\"\"
{raw_text[:3000]}
\"\"\"

Based ONLY on the above, answer in 2-3 sentences:
1. Do these flags suggest genuine fraud/exaggeration, or are they likely false positives?
2. What should the recruiter look out for in the interview?

Be concise and professional. Do not hallucinate details not in the resume."""

        verdict = call_llm(prompt, provider=provider, api_key=api_key)
        logger.info("LLM fraud validation complete: %d chars.", len(verdict))
        return verdict.strip()

    except Exception as e:
        logger.warning("LLM fraud validation skipped: %s", e)
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  6. generate_fraud_report(parsed_json, raw_text)
# ══════════════════════════════════════════════════════════════════════════════

def generate_fraud_report(
    parsed_json: dict,
    raw_text: str,
    api_key: str = "",
    provider: str = "gemini",
    use_llm: bool = False,
) -> dict:
    """
    Full fraud detection pipeline. Runs all checks and produces a report.

    Args:
        parsed_json: Structured parsed resume dict.
        raw_text:    Raw extracted resume text.
        api_key:     (Optional) API key for LLM validation.
        provider:    LLM provider ("gemini" or "openai").
        use_llm:     Whether to invoke optional LLM validation.

    Returns:
        {
            "status":       "success" | "error",
            "fraud_risk":   "Low" | "Medium" | "High",
            "risk_score":   int (0-100),
            "flags":        [...all flag strings],
            "flag_details": {category: [flags]},
            "llm_verdict":  str | None,
            "summary":      str,
        }
    """
    result = {
        "status":       "error",
        "fraud_risk":   "Unknown",
        "risk_score":   0,
        "flags":        [],
        "flag_details": {},
        "llm_verdict":  None,
        "summary":      "",
    }

    try:
        flag_details: dict[str, list[str]] = {}

        # ── Run all checks ────────────────────────────────────────────────
        skill_flags = check_skill_inconsistency(parsed_json)
        flag_details["Skill Inconsistency"] = skill_flags

        stuffing_flags = detect_keyword_stuffing(raw_text)
        flag_details["Keyword Stuffing"] = stuffing_flags

        hidden_flags = detect_hidden_text(raw_text, parsed_json)
        flag_details["Hidden Text"] = hidden_flags

        temporal_flags = check_temporal_consistency(parsed_json.get("experience", ""))
        flag_details["Temporal Consistency"] = temporal_flags

        # Aggregate all flags
        all_flags = skill_flags + stuffing_flags + hidden_flags + temporal_flags

        # ── Compute risk ──────────────────────────────────────────────────
        risk_level, risk_score = compute_fraud_risk(all_flags)

        # ── Optional LLM validation (only on Medium/High risk) ────────────
        llm_verdict = None
        if use_llm and api_key and (risk_level in ("Medium", "High")) and all_flags:
            llm_verdict = validate_with_llm(raw_text, all_flags, api_key, provider)

        # ── Build summary ─────────────────────────────────────────────────
        if not all_flags:
            summary = "✅ No fraud indicators detected. Resume appears consistent and legitimate."
        elif risk_level == "Low":
            summary = (
                f"🟡 {len(all_flags)} minor indicator(s) found. "
                "Likely not fraud — but worth a quick review during the interview."
            )
        elif risk_level == "Medium":
            summary = (
                f"🟠 {len(all_flags)} flag(s) detected. "
                "Some inconsistencies need clarification from the candidate."
            )
        else:
            summary = (
                f"🔴 {len(all_flags)} serious flag(s) detected. "
                "This resume shows multiple signs of manipulation. Proceed with caution."
            )

        result.update({
            "status":       "success",
            "fraud_risk":   risk_level,
            "risk_score":   risk_score,
            "flags":        all_flags,
            "flag_details": flag_details,
            "llm_verdict":  llm_verdict,
            "summary":      summary,
        })

    except Exception as e:
        result["summary"] = f"Fraud detection error: {str(e)}"
        logger.exception("Error in generate_fraud_report.")

    return result
