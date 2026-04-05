"""
HireMind AI - Data Ingestion Module
====================================
Handles file upload, text extraction, cleaning, and local storage
for candidate resumes (PDF, DOCX, TXT).

Pipeline:
    upload → save_raw → extract_text → clean_text → save_processed
"""

import os
import re
import logging
import shutil
from pathlib import Path
from datetime import datetime

# --- PDF Extraction ---
import PyPDF2
from pdfminer.high_level import extract_text as pdfminer_extract

# --- DOCX Extraction ---
from docx import Document

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("hiremind.ingestion")

# ── Storage Paths ──────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent.parent   # project root
RAW_DIR        = BASE_DIR / "data" / "resumes" / "raw"
PROCESSED_DIR  = BASE_DIR / "data" / "resumes" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. save_uploaded_file ──────────────────────────────────────────────────────
def save_uploaded_file(uploaded_file) -> Path:
    """
    Saves a Streamlit UploadedFile object to the raw/ directory.

    Args:
        uploaded_file: Streamlit UploadedFile object.

    Returns:
        Path to the saved file.
    """
    # Build a timestamped filename to avoid collisions
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem        = Path(uploaded_file.name).stem
    suffix      = Path(uploaded_file.name).suffix.lower()
    safe_stem   = re.sub(r"[^\w\-]", "_", stem)          # sanitise filename
    filename    = f"{safe_stem}_{timestamp}{suffix}"
    dest_path   = RAW_DIR / filename

    with open(dest_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    logger.info("Raw file saved → %s", dest_path)
    return dest_path


# ── 2a. PDF extraction helpers ────────────────────────────────────────────────
def _extract_pdf_pypdf2(file_path: Path) -> str:
    """Primary PDF parser using PyPDF2."""
    text_parts = []
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def _extract_pdf_pdfminer(file_path: Path) -> str:
    """Fallback PDF parser using pdfminer.six (handles complex layouts better)."""
    return pdfminer_extract(str(file_path))


def _extract_pdf_ocr_placeholder(file_path: Path) -> str:
    """
    [PLACEHOLDER] OCR fallback for scanned / image-only PDFs.
    Future implementation:
        - Convert PDF pages to images via pdf2image / Pillow
        - Run Tesseract on each image: pytesseract.image_to_string(img)
        - Concatenate results

    Requires:
        pip install pdf2image pytesseract
        Tesseract binary installed on PATH.
    """
    logger.warning(
        "OCR fallback triggered for '%s'. Tesseract not yet implemented.", file_path.name
    )
    return ""


# ── 2b. extract_text ──────────────────────────────────────────────────────────
def extract_text(file_path: Path) -> str:
    """
    Detects file type and routes to the appropriate extractor.

    Strategy:
        PDF  → PyPDF2 (primary) → pdfminer.six (fallback) → OCR (placeholder)
        DOCX → python-docx
        TXT  → plain read with UTF-8 coercion

    Args:
        file_path: Path to the stored raw file.

    Returns:
        Extracted raw text string.

    Raises:
        ValueError: For unsupported file extensions.
    """
    suffix = file_path.suffix.lower()
    logger.info("Extracting text from '%s' (type: %s)", file_path.name, suffix)

    # ── PDF ──────────────────────────────────────────────────────────────────
    if suffix == ".pdf":
        text = ""

        # Primary: PyPDF2
        try:
            text = _extract_pdf_pypdf2(file_path)
            if text.strip():
                logger.info("PyPDF2 extraction succeeded.")
                return text
            else:
                logger.warning("PyPDF2 returned empty text, trying pdfminer fallback.")
        except Exception as e:
            logger.warning("PyPDF2 failed: %s. Falling back to pdfminer.", e)

        # Fallback: pdfminer.six
        try:
            text = _extract_pdf_pdfminer(file_path)
            if text.strip():
                logger.info("pdfminer fallback succeeded.")
                return text
            else:
                logger.warning("pdfminer returned empty text; scanned PDF suspected.")
        except Exception as e:
            logger.warning("pdfminer failed: %s. Triggering OCR placeholder.", e)

        # Last resort: OCR placeholder
        text = _extract_pdf_ocr_placeholder(file_path)
        return text

    # ── DOCX ─────────────────────────────────────────────────────────────────
    elif suffix == ".docx":
        try:
            doc = Document(str(file_path))
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            text = "\n".join(paragraphs)
            logger.info("python-docx extraction succeeded.")
            return text
        except Exception as e:
            logger.error("DOCX extraction failed: %s", e)
            raise

    # ── TXT ──────────────────────────────────────────────────────────────────
    elif suffix == ".txt":
        try:
            raw_bytes = file_path.read_bytes()
            # Coerce to UTF-8; replace undecodable bytes
            text = raw_bytes.decode("utf-8", errors="replace")
            logger.info("TXT file read successfully.")
            return text
        except Exception as e:
            logger.error("TXT read failed: %s", e)
            raise

    else:
        raise ValueError(f"Unsupported file type: '{suffix}'. Expected pdf, docx, or txt.")


# ── 3. clean_text ─────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Normalises and cleans extracted resume text for downstream processing.

    Operations:
        - Normalise unicode characters to ASCII where possible
        - Remove non-printable / control characters (except newlines/tabs)
        - Collapse sequences of whitespace/blank lines
        - Strip leading/trailing whitespace

    Args:
        text: Raw extracted text.

    Returns:
        Cleaned, normalised text string.
    """
    if not text:
        return ""

    # Step 1: Encode to UTF-8 and decode back (ensures consistent encoding)
    text = text.encode("utf-8", errors="ignore").decode("utf-8")

    # Step 2: Remove non-printable control characters (keep \n and \t)
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", " ", text)

    # Step 3: Normalise unicode dashes, quotes, and bullets to ASCII equivalents
    replacements = {
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2018": "'",  # left single quote
        "\u2019": "'",  # right single quote
        "\u201C": '"',  # left double quote
        "\u201D": '"',  # right double quote
        "\u2022": "*",  # bullet
        "\u00A0": " ",  # non-breaking space
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)

    # Step 4: Collapse multiple spaces into one
    text = re.sub(r"[ \t]+", " ", text)

    # Step 5: Collapse more than 2 consecutive newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Step 6: Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in text.splitlines()]
    text  = "\n".join(lines)

    # Step 7: Final strip
    text = text.strip()

    logger.info("Text cleaning complete. Length: %d chars.", len(text))
    return text


# ── 4. save_processed_text ────────────────────────────────────────────────────
def save_processed_text(raw_file_path: Path, cleaned_text: str) -> Path:
    """
    Saves cleaned text to the processed/ directory as a .txt file.

    The output filename mirrors the raw file's stem for easy tracing.

    Args:
        raw_file_path: Path of the original raw file (for naming convention).
        cleaned_text: The cleaned text to persist.

    Returns:
        Path to the saved processed text file.
    """
    output_filename = raw_file_path.stem + ".txt"
    output_path     = PROCESSED_DIR / output_filename

    output_path.write_text(cleaned_text, encoding="utf-8")
    logger.info("Processed text saved → %s", output_path)
    return output_path


# ── 5. process_resume (Main Pipeline) ─────────────────────────────────────────
def process_resume(uploaded_file) -> dict:
    """
    Full ingestion pipeline for a candidate's resume.

    Steps:
        1. Save uploaded file to data/resumes/raw/
        2. Extract raw text based on file type
        3. Clean and normalise the extracted text
        4. Persist cleaned text to data/resumes/processed/

    Args:
        uploaded_file: Streamlit UploadedFile object.

    Returns:
        dict with keys:
            - "status"        : "success" | "error"
            - "raw_path"      : Path to saved raw file
            - "processed_path": Path to saved processed .txt
            - "cleaned_text"  : The cleaned text string
            - "char_count"    : Length of extracted text
            - "word_count"    : Approx word count
            - "message"       : Human-readable status message
    """
    result = {
        "status"         : "error",
        "raw_path"       : None,
        "processed_path" : None,
        "cleaned_text"   : "",
        "char_count"     : 0,
        "word_count"     : 0,
        "message"        : "",
    }

    try:
        # Step 1 — Save raw file
        raw_path = save_uploaded_file(uploaded_file)
        result["raw_path"] = raw_path

        # Step 2 — Extract text
        raw_text = extract_text(raw_path)
        if not raw_text.strip():
            result["message"] = (
                "⚠️ No text could be extracted. "
                "The file may be image-only or password-protected."
            )
            return result

        # Step 3 — Clean text
        cleaned_text = clean_text(raw_text)

        # Step 4 — Save processed text
        processed_path = save_processed_text(raw_path, cleaned_text)

        # Populate success result
        result.update({
            "status"         : "success",
            "processed_path" : processed_path,
            "cleaned_text"   : cleaned_text,
            "char_count"     : len(cleaned_text),
            "word_count"     : len(cleaned_text.split()),
            "message"        : "Resume uploaded and processed successfully ✅",
        })

    except ValueError as ve:
        result["message"] = f"❌ Unsupported file type: {ve}"
        logger.error("ValueError in process_resume: %s", ve)

    except Exception as e:
        result["message"] = f"❌ Processing failed: {e}"
        logger.exception("Unexpected error in process_resume.")

    return result
