"""
HireMind AI - LLM Analysis Module
====================================================
Uses LLM intelligence (Google Gemini / OpenAI GPT-4) to produce
deep qualitative analysis of a candidate's resume against a
Job Description.

Pipeline:
    resume_text + jd_text → build_analysis_prompt → call_llm →
    parse_llm_output → structured JSON report
"""

import json
import re
import logging
from typing import Optional

# --- Logging ---
logger = logging.getLogger("hiremind.llm_analysis")

# ── Supported Providers ────────────────────────────────────────────────────────
PROVIDER_GEMINI = "gemini"
PROVIDER_OPENAI = "openai"

# ── Default Models ─────────────────────────────────────────────────────────────
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


# ══════════════════════════════════════════════════════════════════════════════
#  1. build_analysis_prompt(resume_text, jd_text, structured_data)
# ══════════════════════════════════════════════════════════════════════════════

def build_analysis_prompt(
    resume_text: str,
    jd_text: str,
    structured_data: Optional[dict] = None,
) -> str:
    """
    Constructs a carefully engineered system+user prompt that instructs
    the LLM to perform deep qualitative analysis.

    The prompt is designed to:
        - Go beyond keyword matching (semantic reasoning)
        - Justify conclusions with evidence from the resume
        - Detect implicit/soft skills from project descriptions
        - Evaluate cultural fit based on role context
        - Return a STRICT JSON schema

    Args:
        resume_text:     Cleaned resume text.
        jd_text:         Job description text.
        structured_data: Optional parsed resume JSON (skills, experience, etc.)

    Returns:
        Formatted prompt string.
    """
    # Build structured context if available
    structured_context = ""
    if structured_data:
        skills = structured_data.get("skills", [])
        name = structured_data.get("name", "Unknown")
        if skills:
            structured_context = f"""
--- PARSED CANDIDATE DATA ---
Candidate Name: {name}
Extracted Skills: {', '.join(skills)}
Education: {structured_data.get('education', 'Not available')}
Experience: {structured_data.get('experience', 'Not available')}
Projects: {structured_data.get('projects', 'Not available')}
"""

    prompt = f"""You are an elite HR intelligence analyst with 20 years of experience in technical recruitment, talent assessment, and organizational psychology. You have deep expertise in evaluating candidates across engineering, data science, AI/ML, and software development roles.

Your task is to perform a DEEP, THOROUGH analysis comparing a candidate's resume against a job description. You must go far beyond surface-level keyword matching.

--- INSTRUCTIONS ---

1. MATCH ANALYSIS:
   - Evaluate how well the candidate's background aligns with the role
   - Consider BOTH explicit skills AND implicit competencies derived from their projects/experience
   - Identify transferable skills even if not directly mentioned
   - Provide a match percentage (0-100) with detailed justification

2. STRENGTHS:
   - Identify 3-5 specific, evidence-backed strengths
   - Reference specific projects, tools, or achievements from the resume
   - Explain WHY each strength is relevant to THIS specific role

3. WEAKNESSES / GAPS:
   - Identify 2-4 areas where the candidate falls short
   - Be constructive, not dismissive
   - Distinguish between "missing skill" vs "skill that can be learned quickly"

4. SOFT SKILLS ANALYSIS:
   - Infer soft skills from project descriptions and experience
   - Look for evidence of: Leadership, Communication, Teamwork, Problem-solving, Adaptability, Initiative
   - Do NOT assume soft skills without evidence

5. CULTURAL FIT:
   - Assess based on the type of organization implied by the JD
   - Startup → look for adaptability, self-starter traits, breadth
   - Corporate → look for specialization, process adherence, scale experience
   - Remote → look for self-discipline, async communication

6. RECOMMENDATIONS:
   - Provide 3-5 specific, actionable recommendations
   - Include learning resources or certifications if relevant
   - Suggest interview questions that probe identified gaps

--- RESUME ---
{resume_text}

{structured_context}
--- JOB DESCRIPTION ---
{jd_text}

--- OUTPUT FORMAT ---
You MUST respond with ONLY a valid JSON object. No additional text, no markdown formatting, no code blocks. Just the raw JSON.

Use this exact schema:
{{
    "match_percentage": <int 0-100>,
    "match_summary": "<2-3 sentence summary explaining the overall match quality>",
    "strengths": [
        "<strength 1 with evidence>",
        "<strength 2 with evidence>",
        "<strength 3 with evidence>"
    ],
    "weaknesses": [
        "<weakness 1 with context>",
        "<weakness 2 with context>"
    ],
    "matched_skills": ["<skill1>", "<skill2>"],
    "missing_skills": ["<skill1>", "<skill2>"],
    "soft_skills_analysis": "<paragraph analyzing leadership, communication, teamwork, problem-solving with EVIDENCE from resume>",
    "cultural_fit": "<paragraph analyzing cultural alignment based on JD context>",
    "recommendations": [
        "<recommendation 1>",
        "<recommendation 2>",
        "<recommendation 3>"
    ],
    "interview_questions": [
        "<suggested question 1 targeting a gap>",
        "<suggested question 2 targeting depth>"
    ]
}}
"""

    return prompt


# ══════════════════════════════════════════════════════════════════════════════
#  2. call_llm(prompt, provider, api_key, model)
# ══════════════════════════════════════════════════════════════════════════════

def call_llm(
    prompt: str,
    provider: str = PROVIDER_GEMINI,
    api_key: str = "",
    model: Optional[str] = None,
) -> str:
    """
    Sends the analysis prompt to the selected LLM provider and returns
    the raw response text.

    Supports:
        - Google Gemini (gemini-1.5-flash / gemini-1.5-pro)
        - OpenAI (gpt-4o-mini / gpt-4o / gpt-4)

    Args:
        prompt:   Fully constructed analysis prompt.
        provider: "gemini" or "openai".
        api_key:  API key for the chosen provider.
        model:    Override model name (optional).

    Returns:
        Raw text response from the LLM.

    Raises:
        ValueError: If provider is unsupported or API key is missing.
    """
    if not api_key:
        raise ValueError(f"API key is required for provider '{provider}'.")

    if provider == PROVIDER_GEMINI:
        return _call_gemini(prompt, api_key, model or DEFAULT_GEMINI_MODEL)
    elif provider == PROVIDER_OPENAI:
        return _call_openai(prompt, api_key, model or DEFAULT_OPENAI_MODEL)
    else:
        raise ValueError(f"Unsupported LLM provider: '{provider}'. Use 'gemini' or 'openai'.")


def _call_gemini(prompt: str, api_key: str, model: str) -> str:
    """Calls Google Gemini API using the new google.genai SDK with basic retry logic."""
    from google import genai
    from google.genai import types
    import time

    client = genai.Client(api_key=api_key)

    config = types.GenerateContentConfig(
        temperature=0.3,
        top_p=0.85,
        max_output_tokens=4096,
    )

    max_retries = 2
    models_to_try = [model, "gemini-flash-latest"]
    
    last_err = ""
    for current_model in models_to_try:
        for attempt in range(max_retries):
            try:
                logger.info("Calling Gemini: %s (Attempt %d)", current_model, attempt + 1)
                response = client.models.generate_content(
                    model=current_model,
                    contents=prompt,
                    config=config,
                )

                if not response or not response.text:
                    raise RuntimeError("Gemini returned an empty response.")

                logger.info("Gemini response received: %d chars.", len(response.text))
                return response.text

            except Exception as e:
                last_err = str(e)
                # If it's a 404 (Not Found), immediately try the next model in the list
                if "404" in last_err or "NOT_FOUND" in last_err:
                    logger.warning("Model %s not found. trying next...", current_model)
                    break 
                
                # If it's a 429 (Quota), wait and retry the same model
                if "429" in last_err or "RESOURCE_EXHAUSTED" in last_err:
                    if attempt < max_retries - 1:
                        logger.warning("Quota hit for %s. Retrying in 5 seconds...", current_model)
                        time.sleep(5)
                        continue
                
                # For any other error, or if retries failed, move to next model
                logger.error("Error with %s: %s", current_model, last_err)
                break
    
    # If we fall through all models and attempts
    if "429" in last_err:
        raise RuntimeError("Gemini API Quota Exceeded. Please wait 1 minute and try again.")
    if "404" in last_err:
        raise RuntimeError("No compatible Gemini model found for your API key. Please check Google AI Studio.")
    raise RuntimeError(f"Gemini API Error: {last_err}")


def _call_openai(prompt: str, api_key: str, model: str) -> str:
    """Calls OpenAI API."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    logger.info("Calling OpenAI model: %s", model)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an elite HR intelligence analyst. "
                    "Always respond with valid JSON only."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=4096,
    )

    text = response.choices[0].message.content
    if not text:
        raise RuntimeError("OpenAI returned an empty response.")

    logger.info("OpenAI response received: %d chars.", len(text))
    return text


# ══════════════════════════════════════════════════════════════════════════════
#  3. parse_llm_output(response)
# ══════════════════════════════════════════════════════════════════════════════

# Expected fields in the output schema
_EXPECTED_FIELDS = {
    "match_percentage":     int,
    "match_summary":        str,
    "strengths":            list,
    "weaknesses":           list,
    "matched_skills":       list,
    "missing_skills":       list,
    "soft_skills_analysis": str,
    "cultural_fit":         str,
    "recommendations":      list,
    "interview_questions":  list,
}

# Default values for missing fields
_DEFAULTS = {
    "match_percentage":     0,
    "match_summary":        "Analysis could not be completed.",
    "strengths":            [],
    "weaknesses":           [],
    "matched_skills":       [],
    "missing_skills":       [],
    "soft_skills_analysis": "Not available.",
    "cultural_fit":         "Not available.",
    "recommendations":      [],
    "interview_questions":  [],
}


def parse_llm_output(raw_response: str) -> dict:
    """
    Parses the raw LLM response into a validated, structured JSON dict.

    Handles:
        - Markdown code block wrappers (```json ... ```)
        - Missing fields (fills with safe defaults)
        - Type validation for each field
        - Completely malformed responses (returns error structure)

    Args:
        raw_response: Raw text from the LLM.

    Returns:
        Validated dict matching the expected schema.
    """
    # Step 1: Strip markdown code block wrappers if present
    cleaned = raw_response.strip()
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    cleaned = cleaned.strip()

    # Step 2: Try to extract JSON from the response
    parsed = None

    # Direct parse attempt
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fallback: Find JSON object in the text
    if parsed is None:
        json_match = re.search(r"\{[\s\S]*\}", cleaned)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

    # If all parsing failed, return error structure
    if parsed is None or not isinstance(parsed, dict):
        logger.error("Failed to parse LLM response as JSON.")
        error_result = dict(_DEFAULTS)
        error_result["match_summary"] = (
            "The AI analysis could not be parsed. "
            "Raw response preview: " + raw_response[:200]
        )
        return error_result

    # Step 3: Validate and fill missing fields
    result = {}
    for field, expected_type in _EXPECTED_FIELDS.items():
        value = parsed.get(field, _DEFAULTS[field])

        # Type coercion
        if expected_type == int and not isinstance(value, int):
            try:
                value = int(value)
            except (ValueError, TypeError):
                value = _DEFAULTS[field]

        if expected_type == list and not isinstance(value, list):
            value = [value] if value else _DEFAULTS[field]

        if expected_type == str and not isinstance(value, str):
            value = str(value) if value else _DEFAULTS[field]

        result[field] = value

    # Clamp match_percentage to 0-100
    result["match_percentage"] = max(0, min(100, result["match_percentage"]))

    logger.info(
        "LLM output parsed: match=%d%%, strengths=%d, weaknesses=%d",
        result["match_percentage"], len(result["strengths"]), len(result["weaknesses"]),
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  4. generate_analysis_report (Main Pipeline Orchestrator)
# ══════════════════════════════════════════════════════════════════════════════

def generate_analysis_report(
    resume_text: str,
    jd_text: str,
    provider: str = PROVIDER_GEMINI,
    api_key: str = "",
    model: Optional[str] = None,
    structured_data: Optional[dict] = None,
) -> dict:
    """
    Full LLM analysis pipeline: build prompt → call LLM → parse → validate.

    Args:
        resume_text:     Cleaned resume text.
        jd_text:         Job description text.
        provider:        "gemini" or "openai".
        api_key:         API key for the chosen provider.
        model:           Override model name (optional).
        structured_data: Optional parsed resume JSON for enhanced context.

    Returns:
        dict with keys:
            - "status":  "success" or "error"
            - "report":  The validated analysis JSON (all expected fields)
            - "message": Human-readable status
    """
    result = {
        "status":  "error",
        "report":  dict(_DEFAULTS),
        "message": "",
    }

    try:
        # Validate inputs
        if not resume_text or not resume_text.strip():
            result["message"] = "Resume text is empty."
            return result

        if not jd_text or not jd_text.strip():
            result["message"] = "Job description is empty."
            return result

        if not api_key:
            result["message"] = (
                f"No API key provided for {provider}. "
                "Please enter your API key in the sidebar."
            )
            return result

        # Step 1 — Build prompt
        prompt = build_analysis_prompt(resume_text, jd_text, structured_data)
        logger.info("Analysis prompt built: %d chars.", len(prompt))

        # Step 2 — Call LLM
        raw_response = call_llm(prompt, provider, api_key, model)

        # Step 3 — Parse and validate
        report = parse_llm_output(raw_response)

        result["status"] = "success"
        result["report"] = report
        result["message"] = (
            f"AI analysis complete — {report['match_percentage']}% match "
            f"({len(report['strengths'])} strengths, "
            f"{len(report['weaknesses'])} gaps identified) ✅"
        )

    except ValueError as e:
        result["message"] = f"Configuration error: {e}"
        logger.error("LLM config error: %s", e)

    except Exception as e:
        result["message"] = f"LLM analysis failed: {e}"
        logger.exception("Error in generate_analysis_report.")

    return result
