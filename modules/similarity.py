"""
HireMind AI - Semantic Similarity Matching Module
====================================================
Given a Job Description, retrieves and ranks the most relevant
resume chunks from the FAISS vector store using cosine similarity,
then applies hard-skill filtering and produces a ranked candidate list.

Pipeline:
    JD text → clean → extract skills → embed → FAISS search →
    aggregate by candidate → skill filter → rank → output
"""

import re
import logging
from typing import Optional
from collections import defaultdict

import numpy as np

from modules.embeddings import (
    _get_model,
    load_vector_store,
    EMBEDDING_DIM,
    INDEX_PATH,
    METADATA_PATH,
)
from modules.text_processing import extract_skills, clean_text_advanced

# --- Logging ---
logger = logging.getLogger("hiremind.similarity")


# ══════════════════════════════════════════════════════════════════════════════
#  1. process_job_description(jd_text)
# ══════════════════════════════════════════════════════════════════════════════

def process_job_description(jd_text: str) -> dict:
    """
    Cleans and structures a raw Job Description for matching.

    Extracts:
        - Cleaned text for embedding
        - Required skills (hard skills) for filtering

    Args:
        jd_text: Raw job description string.

    Returns:
        dict with keys:
            - "cleaned_text": Cleaned JD text
            - "skills":       List of extracted skill keywords
            - "word_count":   Word count of JD
    """
    cleaned = clean_text_advanced(jd_text)

    # Extract skills using the same taxonomy as resume parsing
    skills = extract_skills(cleaned)

    result = {
        "cleaned_text": cleaned,
        "skills":       skills,
        "word_count":   len(cleaned.split()),
    }

    logger.info(
        "JD processed: %d words, %d skills extracted.",
        result["word_count"], len(skills),
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  2. generate_jd_embedding(jd_text)
# ══════════════════════════════════════════════════════════════════════════════

def generate_jd_embedding(jd_text: str) -> np.ndarray:
    """
    Generates a single embedding vector for the Job Description.

    Uses the SAME model (all-MiniLM-L6-v2) and L2-normalization
    as the resume embeddings, ensuring consistent vector space.

    Args:
        jd_text: Cleaned job description text.

    Returns:
        numpy array of shape (1, 384), dtype float32.
    """
    model = _get_model()

    embedding = model.encode(
        [jd_text],
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,   # Must match resume embedding normalization
    )

    embedding = embedding.astype(np.float32)
    logger.info("JD embedding generated: shape=%s", embedding.shape)
    return embedding


# ══════════════════════════════════════════════════════════════════════════════
#  3. load_faiss_index()
# ══════════════════════════════════════════════════════════════════════════════

def load_faiss_index() -> tuple:
    """
    Loads the persisted FAISS index and chunk metadata.

    Wrapper around embeddings.load_vector_store() with error context.

    Returns:
        Tuple of (faiss_index, metadata_list).

    Raises:
        FileNotFoundError: If no index has been built yet.
    """
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            "No FAISS index found. Please upload at least one resume first."
        )
    index, metadata = load_vector_store()
    logger.info("Index loaded: %d vectors, %d metadata entries.", index.ntotal, len(metadata))
    return index, metadata


# ══════════════════════════════════════════════════════════════════════════════
#  4. search_similar_resumes(jd_embedding, k)
# ══════════════════════════════════════════════════════════════════════════════

def search_similar_resumes(
    jd_embedding: np.ndarray,
    index,
    metadata: list[dict],
    k: int = 20,
) -> list[dict]:
    """
    Performs a Top-K nearest neighbor search on the FAISS index.

    Since we use IndexFlatIP with L2-normalized vectors, the returned
    scores ARE cosine similarity values (range: -1 to 1, higher = better).

    Args:
        jd_embedding: JD vector of shape (1, dim).
        index:        FAISS index object.
        metadata:     List of chunk metadata dicts.
        k:            Number of nearest neighbors to retrieve.

    Returns:
        List of dicts, each with:
            - "chunk_text":  The matched chunk text
            - "score":       Cosine similarity score (0-1)
            - "section":     Which resume section the chunk came from
            - "source":      "resume" or "job_description"
            - "name":        Candidate name
            - "chunk_id":    Original chunk ID
    """
    # Clamp k to the number of available vectors
    actual_k = min(k, index.ntotal)
    if actual_k == 0:
        logger.warning("FAISS index is empty.")
        return []

    # FAISS search
    scores, indices = index.search(jd_embedding, actual_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue  # FAISS returns -1 for missing results

        chunk_meta = metadata[idx]
        results.append({
            "chunk_text": chunk_meta.get("text", ""),
            "score":      float(max(0.0, score)),  # Clamp negative scores to 0
            "section":    chunk_meta.get("section", "unknown"),
            "source":     chunk_meta.get("source", "unknown"),
            "name":       chunk_meta.get("name", "Unknown"),
            "chunk_id":   chunk_meta.get("chunk_id", -1),
        })

    logger.info("FAISS search returned %d results (k=%d).", len(results), actual_k)
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  5. aggregate_by_candidate(search_results)
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_by_candidate(search_results: list[dict]) -> dict:
    """
    Groups chunk-level search results by candidate name and computes
    an aggregate similarity score.

    Scoring strategy:
        - Mean of top chunk scores (to reward depth of match)
        - Max score (to capture the best single match)
        - Combined score = 0.6 * max + 0.4 * mean (balanced)

    Args:
        search_results: List of chunk-level match dicts from FAISS search.

    Returns:
        dict mapping candidate name → {
            "max_score", "mean_score", "combined_score",
            "matched_sections", "top_chunks"
        }
    """
    candidates = defaultdict(list)

    for result in search_results:
        if result["source"] == "resume":
            candidates[result["name"]].append(result)

    aggregated = {}
    for name, chunks in candidates.items():
        scores = [c["score"] for c in chunks]
        sections = list(set(c["section"] for c in chunks))

        max_score  = max(scores)
        mean_score = sum(scores) / len(scores)
        combined   = 0.6 * max_score + 0.4 * mean_score

        # Deduplication + Threshold Selection
        # 1. Sort by score descending
        sorted_chunks = sorted(chunks, key=lambda x: x["score"], reverse=True)
        
        unique_chunks = []
        seen_content_hashes = set()
        
        for c in sorted_chunks:
            # Apply dynamic threshold: only show meaningful matches
            if c["score"] < 0.40:
                continue
                
            # Content deduplication: use a simplified normalisation to detect near-duplicates
            # (common with overlapping chunks)
            content_norm = re.sub(r"[^a-zA-Z0-9]", "", c["chunk_text"].lower())[:150]
            if content_norm in seen_content_hashes:
                continue
                
            seen_content_hashes.add(content_norm)
            unique_chunks.append(c)
            
            # Optional: limit to top 10 to keep dashboard readable, but remove hardcoded 3
            if len(unique_chunks) >= 10:
                break

        aggregated[name] = {
            "max_score":        round(max_score, 4),
            "mean_score":       round(mean_score, 4),
            "combined_score":   round(combined, 4),
            "matched_sections": sections,
            "num_chunks_hit":   len(chunks),
            "top_chunks":       unique_chunks,
        }

    logger.info("Aggregated results for %d candidates.", len(aggregated))
    return aggregated


# ══════════════════════════════════════════════════════════════════════════════
#  6. filter_by_hard_skills(candidate_data, jd_skills, candidate_skills)
# ══════════════════════════════════════════════════════════════════════════════

def filter_by_hard_skills(
    candidate_skills: list[str],
    jd_skills: list[str],
) -> dict:
    """
    Compares a candidate's extracted skills against JD requirements.

    Uses case-insensitive matching to find overlaps and gaps.

    Args:
        candidate_skills: List of skills from resume parsing.
        jd_skills:        List of skills from JD parsing.

    Returns:
        dict with:
            - "matched_skills": Skills the candidate HAS
            - "missing_skills": Skills the candidate LACKS
            - "match_ratio":    Fraction of JD skills matched (0-1)
            - "skill_score":    Weighted score for ranking boost
    """
    if not jd_skills:
        return {
            "matched_skills": [],
            "missing_skills": [],
            "match_ratio":    0.0,
            "skill_score":    0.0,
        }

    candidate_lower = {s.lower() for s in candidate_skills}
    jd_lower_map    = {s.lower(): s for s in jd_skills}

    matched = []
    missing = []

    for skill_lower, skill_original in jd_lower_map.items():
        if skill_lower in candidate_lower:
            matched.append(skill_original)
        else:
            missing.append(skill_original)

    match_ratio = len(matched) / len(jd_skills) if jd_skills else 0.0

    return {
        "matched_skills": sorted(matched),
        "missing_skills": sorted(missing),
        "match_ratio":    round(match_ratio, 4),
        "skill_score":    round(match_ratio * 100, 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  7. rank_candidates(aggregated, skill_analysis)
# ══════════════════════════════════════════════════════════════════════════════

def rank_candidates(
    aggregated: dict,
    skill_analyses: dict,
) -> list[dict]:
    """
    Produces the final ranked list of candidates.

    Final score = (semantic_similarity * 0.65) + (skill_match_ratio * 0.35)

    This weighting ensures:
        - Candidates who are semantically relevant rank high
        - But those missing critical skills get penalised

    Args:
        aggregated:     dict from aggregate_by_candidate().
        skill_analyses: dict mapping candidate name → filter_by_hard_skills() output.

    Returns:
        Sorted list of candidate result dicts (highest score first).
    """
    ranked = []

    for name, agg_data in aggregated.items():
        skill_data = skill_analyses.get(name, {})

        semantic_score = agg_data["combined_score"]
        skill_ratio    = skill_data.get("match_ratio", 0.0)

        # Weighted final score
        final_score = (semantic_score * 0.65) + (skill_ratio * 0.35)

        ranked.append({
            "name":              name,
            "final_score":       round(final_score * 100, 2),
            "semantic_score":    round(semantic_score * 100, 2),
            "skill_score":       round(skill_ratio * 100, 2),
            "matched_skills":    skill_data.get("matched_skills", []),
            "missing_skills":    skill_data.get("missing_skills", []),
            "matched_sections":  agg_data["matched_sections"],
            "num_chunks_hit":    agg_data["num_chunks_hit"],
            "top_chunks":        agg_data["top_chunks"],
        })

    # Sort by final_score descending
    ranked.sort(key=lambda x: x["final_score"], reverse=True)

    logger.info(
        "Ranked %d candidates. Top: %s (%.1f%%)",
        len(ranked),
        ranked[0]["name"] if ranked else "N/A",
        ranked[0]["final_score"] if ranked else 0,
    )
    return ranked


# ══════════════════════════════════════════════════════════════════════════════
#  8. match_resume_to_jd (Main Pipeline Orchestrator)
# ══════════════════════════════════════════════════════════════════════════════

def match_resume_to_jd(
    jd_text: str,
    candidate_skills: Optional[list[str]] = None,
    k: int = 20,
) -> dict:
    """
    Full matching pipeline: JD → embed → search → filter → rank.

    Args:
        jd_text:          Raw or cleaned Job Description text.
        candidate_skills: Skills list from parsed resume (for skill filtering).
                          If None, skill filtering is skipped.
        k:                Number of top chunks to retrieve from FAISS.

    Returns:
        dict with keys:
            - "status":       "success" or "error"
            - "jd_skills":    Skills extracted from JD
            - "ranked":       Sorted list of candidate results
            - "total_found":  Number of candidates found
            - "message":      Human-readable status
    """
    result = {
        "status":      "error",
        "jd_skills":   [],
        "ranked":      [],
        "total_found": 0,
        "message":     "",
    }

    try:
        # Step 1 — Process JD
        jd_data = process_job_description(jd_text)
        result["jd_skills"] = jd_data["skills"]

        if not jd_data["cleaned_text"].strip():
            result["message"] = "Job description is empty after cleaning."
            return result

        # Step 2 — Generate JD embedding
        jd_embedding = generate_jd_embedding(jd_data["cleaned_text"])

        # Step 3 — Load FAISS index
        index, metadata = load_faiss_index()

        # Step 4 — FAISS search
        search_results = search_similar_resumes(jd_embedding, index, metadata, k)
        if not search_results:
            result["message"] = "No matching resumes found in the vector store."
            return result

        # Step 5 — Aggregate by candidate
        aggregated = aggregate_by_candidate(search_results)

        # Step 6 — Skill filtering for each candidate
        skill_analyses = {}
        for name in aggregated:
            # Use provided skills if this is the uploaded candidate,
            # otherwise try to find skills in the chunk text
            if candidate_skills is not None:
                skills_for_candidate = candidate_skills
            else:
                # Fallback: extract skills from top chunks
                combined_text = " ".join(
                    c["chunk_text"] for c in aggregated[name]["top_chunks"]
                )
                skills_for_candidate = extract_skills(combined_text)

            skill_analyses[name] = filter_by_hard_skills(
                skills_for_candidate, jd_data["skills"]
            )

        # Step 7 — Rank
        ranked = rank_candidates(aggregated, skill_analyses)

        result["status"]      = "success"
        result["ranked"]      = ranked
        result["total_found"] = len(ranked)
        result["message"]     = f"Found {len(ranked)} candidate(s) matching the JD ✅"

    except FileNotFoundError as e:
        result["message"] = f"⚠️ {e}"
        logger.warning("No FAISS index: %s", e)

    except Exception as e:
        result["message"] = f"❌ Matching failed: {e}"
        logger.exception("Error in match_resume_to_jd.")

    return result
