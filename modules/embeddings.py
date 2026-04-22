"""
HireMind AI - Embeddings & Vector Database Module
====================================================
Converts structured resume/JD data into dense vector embeddings
using Sentence Transformers, stores them in a FAISS index for
semantic retrieval.

Pipeline:
    structured_json → prepare_text_chunks → chunk_text →
    generate_embeddings → create_faiss_index → save_vector_store
"""

import json
import logging
import pickle
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Logging ---
logger = logging.getLogger("hiremind.embeddings")

# ── Storage Paths ──────────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).resolve().parent.parent
VECTOR_STORE_DIR = BASE_DIR / "data" / "vector_store"
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH    = VECTOR_STORE_DIR / "faiss.index"
METADATA_PATH = VECTOR_STORE_DIR / "metadata.pkl"

# ── Model Config ───────────────────────────────────────────────────────────────
MODEL_NAME     = "all-MiniLM-L6-v2"
EMBEDDING_DIM  = 384   # Dimension of all-MiniLM-L6-v2 output

# Chunking Config
CHUNK_SIZE     = 400
CHUNK_OVERLAP  = 80

import streamlit as st

# ── Cached Model & Resource Loading ──────────────────────────────────────────
@st.cache_resource
def _get_model() -> SentenceTransformer:
    """Load the Sentence Transformer model once and cache it via Streamlit."""
    logger.info("Loading Sentence Transformer model: %s ...", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    logger.info("Model loaded. Embedding dimension: %d", EMBEDDING_DIM)
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  1. prepare_text_chunks(structured_json)
# ══════════════════════════════════════════════════════════════════════════════

def prepare_text_chunks(structured_json: dict, source_label: str = "resume") -> list[dict]:
    """
    Converts a structured resume/JD JSON into meaningful text segments
    suitable for embedding.

    Each section gets a descriptive prefix so the embedding model understands
    the context of the text.

    Args:
        structured_json: Parsed resume/JD dict from text_processing module.
        source_label:    "resume" or "job_description" — for metadata tagging.

    Returns:
        List of dicts, each with:
            - "text":   the prepared text string
            - "section": which resume section it came from
            - "source":  "resume" or "job_description"
            - "name":    candidate name (if available)
    """
    chunks = []
    name = structured_json.get("name", "")

    # Define which fields to process and their display prefixes
    field_map = {
        "summary":        "Professional Summary",
        "skills":         "Technical & Professional Skills",
        "experience":     "Work Experience",
        "education":      "Education & Qualifications",
        "projects":       "Projects & Portfolio",
        "certifications": "Certifications & Licenses",
        "achievements":   "Achievements & Awards",
    }

    for field_key, prefix in field_map.items():
        content = structured_json.get(field_key, "")

        # Handle skills list → join into string
        if isinstance(content, list):
            content = ", ".join(str(item) for item in content)

        content = str(content).strip()
        if not content:
            continue

        # Build a contextual text block (ONLY actual content, no prefixes)
        text_block = content

        chunks.append({
            "text":    text_block,
            "section": field_key,
            "source":  source_label,
            "name":    name,
        })
    logger.info(
        "Prepared %d text segments from %s (%s).",
        len(chunks), source_label, name,
    )
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
#  2. chunk_text(segments)
# ══════════════════════════════════════════════════════════════════════════════

def chunk_text(segments: list[dict]) -> list[dict]:
    """
    Splits longer text segments into smaller, overlapping chunks
    using LangChain's RecursiveCharacterTextSplitter.

    Short segments (< CHUNK_SIZE) are kept as-is.
    Long segments are split with overlap to preserve context at boundaries.

    Args:
        segments: List of dicts from prepare_text_chunks().

    Returns:
        List of dicts with the same metadata, but with text split into
        chunks. Each chunk dict has an additional "chunk_id" field.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
    )

    all_chunks = []
    chunk_counter = 0

    for segment in segments:
        text = segment["text"]

        if len(text) <= CHUNK_SIZE:
            # Small enough — keep as one chunk
            all_chunks.append({
                **segment,
                "chunk_id": chunk_counter,
            })
            chunk_counter += 1
        else:
            # Split into smaller pieces
            sub_chunks = splitter.split_text(text)
            for sub in sub_chunks:
                all_chunks.append({
                    "text":     sub,
                    "section":  segment["section"],
                    "source":   segment["source"],
                    "name":     segment["name"],
                    "chunk_id": chunk_counter,
                })
                chunk_counter += 1

    logger.info(
        "Chunking complete: %d segments → %d chunks (size=%d, overlap=%d).",
        len(segments), len(all_chunks), CHUNK_SIZE, CHUNK_OVERLAP,
    )
    return all_chunks


# ══════════════════════════════════════════════════════════════════════════════
#  3. generate_embeddings(chunks)
# ══════════════════════════════════════════════════════════════════════════════

def generate_embeddings(chunks: list[dict]) -> np.ndarray:
    """
    Generates dense vector embeddings for each text chunk.

    Args:
        chunks: List of chunk dicts (must contain "text" key).

    Returns:
        numpy array of shape (n_chunks, EMBEDDING_DIM), dtype float32.
    """
    model = _get_model()
    texts = [chunk["text"] for chunk in chunks]

    logger.info("Generating embeddings for %d chunks...", len(texts))
    embeddings = model.encode(
        texts,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalize for cosine similarity
    )

    # Ensure correct dtype for FAISS
    embeddings = embeddings.astype(np.float32)

    logger.info(
        "Embeddings generated: shape=%s, dtype=%s",
        embeddings.shape, embeddings.dtype,
    )
    return embeddings


# ══════════════════════════════════════════════════════════════════════════════
#  4. create_faiss_index(embeddings)
# ══════════════════════════════════════════════════════════════════════════════

def create_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Creates a FAISS index from embeddings.

    Uses IndexFlatIP (Inner Product) because embeddings are L2-normalized,
    so inner product == cosine similarity. This gives us direct similarity
    scores without additional computation.

    Args:
        embeddings: numpy array of shape (n, EMBEDDING_DIM).

    Returns:
        Populated FAISS index.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner Product (cosine sim for normalized vecs)
    index.add(embeddings)

    logger.info(
        "FAISS index created: %d vectors, dimension=%d.",
        index.ntotal, dim,
    )
    return index


# ══════════════════════════════════════════════════════════════════════════════
#  5. save_vector_store(index, metadata)
# ══════════════════════════════════════════════════════════════════════════════

def save_vector_store(
    index: faiss.IndexFlatIP,
    metadata: list[dict],
    index_path: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
) -> dict:
    """
    Persists the FAISS index and chunk metadata to disk.

    Args:
        index:         FAISS index object.
        metadata:      List of chunk dicts (text, section, source, name, chunk_id).
        index_path:    Where to save the .index file. Defaults to data/vector_store/faiss.index.
        metadata_path: Where to save metadata. Defaults to data/vector_store/metadata.pkl.

    Returns:
        dict with saved file paths and stats.
    """
    idx_path  = index_path or INDEX_PATH
    meta_path = metadata_path or METADATA_PATH

    # Save FAISS index
    faiss.write_index(index, str(idx_path))
    logger.info("FAISS index saved → %s", idx_path)

    # Save metadata (using pickle for speed; JSON alternative below)
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)
    logger.info("Metadata saved → %s (%d chunks)", meta_path, len(metadata))

    # Also save a human-readable JSON version for inspection
    json_path = meta_path.with_suffix(".json")
    json_safe = [
        {k: v for k, v in chunk.items() if k != "embedding"}
        for chunk in metadata
    ]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_safe, f, indent=2, ensure_ascii=False)

    return {
        "index_path":    idx_path,
        "metadata_path": meta_path,
        "json_path":     json_path,
        "total_vectors": index.ntotal,
        "dimension":     EMBEDDING_DIM,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  6. load_vector_store()
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_vector_store(
    index_path: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
) -> tuple:
    """
    Loads a previously saved FAISS index and metadata from disk.

    Args:
        index_path:    Path to .index file.
        metadata_path: Path to .pkl metadata file.

    Returns:
        Tuple of (faiss_index, metadata_list).

    Raises:
        FileNotFoundError: If index or metadata files don't exist.
    """
    idx_path  = index_path or INDEX_PATH
    meta_path = metadata_path or METADATA_PATH

    if not idx_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {idx_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    index = faiss.read_index(str(idx_path))
    logger.info("FAISS index loaded: %d vectors.", index.ntotal)

    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)
    logger.info("Metadata loaded: %d chunks.", len(metadata))

    return index, metadata


# ══════════════════════════════════════════════════════════════════════════════
#  7. process_embeddings (Main Pipeline Orchestrator)
# ══════════════════════════════════════════════════════════════════════════════

def process_embeddings(
    structured_json: dict,
    source_label: str = "resume",
    persist: bool = True,
) -> dict:
    """
    Full embedding pipeline: structured JSON → vectors → FAISS index.

    Steps:
        1. Prepare text segments from structured data
        2. Chunk long segments with overlap
        3. Generate embeddings via Sentence Transformer
        4. Build FAISS index
        5. Save to disk (optional)

    Args:
        structured_json: Parsed resume/JD from text_processing module.
        source_label:    "resume" or "job_description".
        persist:         Whether to save the index to disk.

    Returns:
        dict with keys:
            - "status"        : "success" or "error"
            - "total_chunks"  : Number of chunks created
            - "total_vectors" : Number of vectors in index
            - "dimension"     : Embedding dimension
            - "index"         : FAISS index object (in-memory reference)
            - "metadata"      : List of chunk metadata dicts
            - "message"       : Human-readable status
            - "storage"       : File paths (if persisted)
    """
    result = {
        "status":        "error",
        "total_chunks":  0,
        "total_vectors": 0,
        "dimension":     EMBEDDING_DIM,
        "index":         None,
        "metadata":      [],
        "message":       "",
        "storage":       None,
    }

    try:
        # Step 1 — Prepare text segments
        segments = prepare_text_chunks(structured_json, source_label)
        if not segments:
            result["message"] = "No text content found in structured data."
            return result

        # Step 2 — Chunk into smaller pieces
        chunks = chunk_text(segments)
        result["total_chunks"] = len(chunks)

        # Step 3 — Generate embeddings
        embeddings = generate_embeddings(chunks)

        # Step 4 — Build FAISS index
        index = create_faiss_index(embeddings)
        result["index"] = index
        result["metadata"] = chunks
        result["total_vectors"] = index.ntotal

        # Step 5 — Persist to disk (OVERWRITE for Freshness)
        if persist:
            # We strictly overwrite the store for every new analysis to ensure
            # "Top Matching Segments" and other data are NEVER stale or dummy.
            storage_info = save_vector_store(index, chunks)
            result["storage"] = storage_info
            result["total_vectors"] = index.ntotal
            result["metadata"] = chunks
            result["index"] = index
            logger.info("Vector store refreshed (overwritten) for analysis.")

        result["status"] = "success"
        result["message"] = (
            f"Embedded {len(chunks)} chunks into {result['total_vectors']} vectors "
            f"(dim={EMBEDDING_DIM}) ✅"
        )

    except Exception as e:
        result["message"] = f"Embedding pipeline failed: {e}"
        logger.exception("Error in process_embeddings.")

    return result
