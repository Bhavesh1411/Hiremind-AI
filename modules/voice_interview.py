"""
HireMind AI - Stage 2 Voice Interview Engine
=============================================
Backend engine for the voice-based interview stage.

Transcription:  AssemblyAI REST API (direct upload → poll — no SDK quirks)
TTS:            gTTS
Evaluation:     Gemini LLM (background thread, non-blocking)
Report:         Stage 1 (60%) + Stage 2 (40%) weighted combined report
"""

import re
import io
import json
import time
import logging
import tempfile
import os
import threading
from typing import List, Dict, Any

logger = logging.getLogger("hiremind.voice_interview")

# ── gTTS (Text-to-Speech) ─────────────────────────────────────────────────────
_GTTS_AVAILABLE = True
try:
    from gtts import gTTS
except ImportError:
    _GTTS_AVAILABLE = False

# ── requests (used for AssemblyAI REST calls) ─────────────────────────────────
try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

from modules.llm_analysis import call_llm, PROVIDER_GEMINI

# AssemblyAI REST endpoints
_AAI_UPLOAD_URL    = "https://api.assemblyai.com/v2/upload"
_AAI_TRANSCRIPT_URL = "https://api.assemblyai.com/v2/transcript"


# ════════════════════════════════════════════════════════════════════════════════
#  QUESTION GENERATION  (Gemini)
# ════════════════════════════════════════════════════════════════════════════════

def generate_voice_questions(
    resume_data: dict,
    jd_text: str,
    api_key: str,
) -> List[Dict[str, str]]:
    """
    Generate exactly 3 personalized voice interview questions using Gemini.

    Returns: [{"id": str, "type": str, "question": str}]
    """
    skills   = resume_data.get("skills", [])
    projects = resume_data.get("projects", [])
    name     = resume_data.get("name", "the candidate")

    skills_str   = ", ".join(skills[:8])   if skills   else "general software engineering"
    projects_str = ", ".join(projects[:3]) if projects else "None listed"

    prompt = f"""
You are an expert senior technical interviewer conducting a real-world voice interview.

Candidate Profile:
  Name: {name}
  Skills: {skills_str}
  Projects: {projects_str}

Job Description Summary:
{jd_text[:600]}

Generate EXACTLY 3 interview questions. Each must be medium-to-hard difficulty and directly
relevant to either the candidate's resume or the job description.

Question types (one of each, in this exact order):
1. project_based  – Ask specifically about one of their listed projects
2. skill_based    – Deep technical question about one of their key skills
3. scenario_based – Real-world scenario they would encounter in this job role

RETURN ONLY a valid JSON array. No markdown. No explanation:
[
  {{"id": "v1", "type": "project_based",  "question": "<full question>"}},
  {{"id": "v2", "type": "skill_based",    "question": "<full question>"}},
  {{"id": "v3", "type": "scenario_based", "question": "<full question>"}}
]
"""
    try:
        response  = call_llm(prompt, PROVIDER_GEMINI, api_key)
        clean_res = re.sub(r"^```(?:json)?\s*\n?", "", response.strip())
        clean_res = re.sub(r"\n?```\s*$", "", clean_res).strip()
        data = json.loads(clean_res)
        if isinstance(data, list) and len(data) >= 3:
            return data[:3]
        raise ValueError("Unexpected response structure")
    except Exception as e:
        logger.error("Voice question generation failed: %s", e)
        return [
            {"id": "v1", "type": "project_based",
             "question": "Tell me about a project you built from scratch. What were the key technical decisions?"},
            {"id": "v2", "type": "skill_based",
             "question": "Explain how you would design a scalable REST API and what technologies you would use."},
            {"id": "v3", "type": "scenario_based",
             "question": "You discover a critical bug in production 30 minutes before a major release. Walk me through your response."},
        ]


# ════════════════════════════════════════════════════════════════════════════════
#  TEXT → SPEECH  (gTTS)
# ════════════════════════════════════════════════════════════════════════════════

def text_to_speech(text: str) -> bytes | None:
    """Convert question text to MP3 bytes. Returns None if gTTS is unavailable."""
    if not _GTTS_AVAILABLE:
        return None
    try:
        tts = gTTS(text=text, lang="en", slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        logger.error("gTTS failed: %s", e)
        return None


# ════════════════════════════════════════════════════════════════════════════════
#  TRANSCRIPTION  — AssemblyAI via Direct REST API
#  Flow: Upload audio file → Request transcript → Poll until complete
# ════════════════════════════════════════════════════════════════════════════════

def transcribe_audio(audio_bytes: bytes, assembly_api_key: str = "") -> str:
    """
    Convert audio bytes → text using AssemblyAI REST API directly.

    Uses the three-step AssemblyAI flow:
      1. POST /v2/upload     — upload raw audio bytes
      2. POST /v2/transcript — request transcription of the upload URL
      3. GET  /v2/transcript/{id} — poll until status == 'completed'

    Returns the transcript text, or "" on failure.
    """
    if not audio_bytes:
        logger.warning("transcribe_audio: empty audio bytes — skipping.")
        return ""

    if not _REQUESTS_AVAILABLE:
        logger.error("requests library not available.")
        return ""

    # Resolve key: param > session > env
    api_key = assembly_api_key or os.getenv("ASSEMBLYAI_API_KEY", "")
    if not api_key:
        logger.warning("No ASSEMBLYAI_API_KEY found.")
        return ""

    headers = {"authorization": api_key}

    # ── STEP 1: Upload audio ──────────────────────────────────────────────────
    try:
        upload_resp = _requests.post(
            _AAI_UPLOAD_URL,
            headers=headers,
            data=audio_bytes,
            timeout=60,
        )
        upload_resp.raise_for_status()
        upload_url = upload_resp.json().get("upload_url")
        if not upload_url:
            raise ValueError("No upload_url in AssemblyAI upload response.")
        logger.info("AssemblyAI: audio uploaded successfully.")
    except Exception as e:
        logger.error("AssemblyAI upload failed: %s", e)
        return ""

    # ── STEP 2: Request transcription ─────────────────────────────────────────
    try:
        transcript_resp = _requests.post(
            _AAI_TRANSCRIPT_URL,
            json={
                "audio_url": upload_url,
                "language_code": "en",
                "speech_models": ["universal-3-pro"],
            },
            headers={**headers, "content-type": "application/json"},
            timeout=30,
        )

        transcript_resp.raise_for_status()
        transcript_id = transcript_resp.json().get("id")
        if not transcript_id:
            raise ValueError("No transcript id in AssemblyAI response.")
        logger.info("AssemblyAI: transcript job created (id=%s).", transcript_id)
    except Exception as e:
        logger.error("AssemblyAI transcript request failed: %s", e)
        return ""

    # ── STEP 3: Poll until complete (max 120 s) ───────────────────────────────
    poll_url = f"{_AAI_TRANSCRIPT_URL}/{transcript_id}"
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            poll_resp = _requests.get(poll_url, headers=headers, timeout=15)
            poll_resp.raise_for_status()
            data   = poll_resp.json()
            status = data.get("status")

            if status == "completed":
                text = (data.get("text") or "").strip()
                logger.info("AssemblyAI: transcription complete (%d chars).", len(text))
                return text

            if status == "error":
                logger.error("AssemblyAI transcript error: %s", data.get("error"))
                return ""

            # status == "queued" or "processing" — keep polling
            time.sleep(3)

        except Exception as e:
            logger.error("AssemblyAI polling error: %s", e)
            return ""

    logger.error("AssemblyAI transcription timed out after 120 s.")
    return ""


# ════════════════════════════════════════════════════════════════════════════════
#  BACKGROUND EVALUATION  (Gemini — non-blocking)
# ════════════════════════════════════════════════════════════════════════════════

def evaluate_voice_answer_async(
    question: Dict[str, str],
    transcript: str,
    api_key: str,
    result_store: list,
) -> None:
    """
    Evaluate a voice answer using Gemini in a background thread.

    The result is appended to `result_store` so the caller can retrieve it later.
    Stores {"score": int, "reasoning": str}.
    """
    q_type = question.get("type", "general")
    q_text = question.get("question", "")

    prompt = f"""
You are an expert senior technical interviewer evaluating a voice interview response.

Question Type: {q_type}
Question: {q_text}

Candidate's spoken answer (transcript):
\"\"\"{transcript}\"\"\"

Evaluate strictly based on:
  1. Correctness and accuracy of the answer
  2. Depth of explanation and technical understanding
  3. Relevance to the actual question asked
  4. Communication clarity

Return ONLY valid JSON. No markdown. No explanation:
{{"score": <integer 0-10>, "reasoning": "<2-3 sentence professional evaluation>"}}
"""
    try:
        response  = call_llm(prompt, PROVIDER_GEMINI, api_key)
        clean_res = re.sub(r"^```(?:json)?\s*\n?", "", response.strip())
        clean_res = re.sub(r"\n?```\s*$", "", clean_res).strip()
        data = json.loads(clean_res)
        if "marks" in data and "score" not in data:
            data["score"] = data.pop("marks")
        result_store.append({
            "score":     int(data.get("score", 0)),
            "reasoning": data.get("reasoning", "No reasoning provided."),
        })
    except Exception as e:
        logger.error("Background voice evaluation failed: %s", e)
        result_store.append({"score": 0, "reasoning": "Evaluation could not be completed."})


def evaluate_voice_answer(
    question: Dict[str, str],
    transcript: str,
    api_key: str,
) -> Dict[str, Any]:
    """
    Synchronous wrapper — kept for backward compat with the combined report.
    Used only at report generation time, not during the live interview UI.
    """
    result_store: list = []
    evaluate_voice_answer_async(question, transcript, api_key, result_store)
    return result_store[0] if result_store else {"score": 0, "reasoning": "Evaluation failed."}


# ════════════════════════════════════════════════════════════════════════════════
#  COMBINED REPORT  (Stage 1 × 60% + Stage 2 × 40%)
# ════════════════════════════════════════════════════════════════════════════════

def generate_combined_report(
    stage1_answers: List[dict],
    stage2_results: List[dict],
    voice_questions: List[dict],
    api_key: str = "",
) -> dict:
    """
    Generate the unified final report combining both interview stages.
    Weights: Stage 1 = 60%, Stage 2 = 40%.
    """
    # ── Stage 1 ───────────────────────────────────────────────────────────────
    s1_total = sum(a.get("score", 0) for a in stage1_answers)
    s1_max   = len(stage1_answers) * 10
    s1_pct   = (s1_total / s1_max * 100) if s1_max > 0 else 0

    s1_strengths  = [a.get("question_text", "")[:60] for a in stage1_answers if a.get("score", 0) >= 7]
    s1_weaknesses = [a.get("question_text", "")[:60] for a in stage1_answers if a.get("score", 0) <= 4]

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    s2_scores = [r.get("score", 0) for r in stage2_results]
    s2_total  = sum(s2_scores)
    s2_max    = len(stage2_results) * 10
    s2_pct    = (s2_total / s2_max * 100) if s2_max > 0 else 0
    s2_avg    = (s2_total / len(stage2_results)) if stage2_results else 0

    # ── Combined ──────────────────────────────────────────────────────────────
    combined_pct = round(s1_pct * 0.60 + s2_pct * 0.40, 1)

    if combined_pct >= 75:
        verdict, verdict_color, verdict_icon = "Strong Hire", "#10b981", "🟢"
    elif combined_pct >= 55:
        verdict, verdict_color, verdict_icon = "Hire",        "#f59e0b", "🟡"
    else:
        verdict, verdict_color, verdict_icon = "No Hire",     "#ef4444", "🔴"

    # ── Gemini deep analysis ──────────────────────────────────────────────────
    ai_analysis = {}
    if api_key and stage2_results:
        try:
            s2_summary = "\n".join([
                f"Q: {voice_questions[i].get('question', '')}\n"
                f"Score: {stage2_results[i].get('score', 0)}/10\n"
                f"Evaluation: {stage2_results[i].get('reasoning', '')}"
                for i in range(len(stage2_results))
            ])
            s1_summary = "\n".join([
                f"Q: {a.get('question_text', '')[:80]}\n"
                f"Score: {a.get('score', 0)}/10\n"
                f"Evaluation: {a.get('evaluation', '')}"
                for a in stage1_answers
            ])
            prompt = f"""
You are the final hiring authority reviewing a two-stage AI interview.

STAGE 1 (Text/Coding - 60% weight):
{s1_summary}

STAGE 2 (Voice - 40% weight):
{s2_summary}

Combined weighted score: {combined_pct}%

Return ONLY JSON:
{{
  "overall_summary": "<2-3 sentences>",
  "key_strengths": ["<s1>", "<s2>", "<s3>"],
  "key_weaknesses": ["<w1>", "<w2>"],
  "skill_gaps": ["<gap1>", "<gap2>"],
  "hiring_recommendation": "<detailed paragraph>",
  "development_areas": ["<area1>", "<area2>"]
}}
"""
            response  = call_llm(prompt, PROVIDER_GEMINI, api_key)
            clean_res = re.sub(r"^```(?:json)?\s*\n?", "", response.strip())
            clean_res = re.sub(r"\n?```\s*$", "", clean_res).strip()
            ai_analysis = json.loads(clean_res)
        except Exception as e:
            logger.error("Combined AI analysis failed: %s", e)

    return {
        "stage1": {
            "answers":    stage1_answers,
            "total":      s1_total,
            "max":        s1_max,
            "percentage": round(s1_pct, 1),
            "strengths":  s1_strengths,
            "weaknesses": s1_weaknesses,
        },
        "stage2": {
            "results":    stage2_results,
            "questions":  voice_questions,
            "total":      s2_total,
            "max":        s2_max,
            "percentage": round(s2_pct, 1),
            "avg_score":  round(s2_avg, 1),
        },
        "combined": {
            "percentage":    combined_pct,
            "verdict":       verdict,
            "verdict_color": verdict_color,
            "verdict_icon":  verdict_icon,
        },
        "ai_analysis": ai_analysis,
    }
