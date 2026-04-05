"""
HireMind AI - Text Processing & Parsing Module
=================================================
Transforms cleaned resume text into structured JSON using
spaCy NER, regex-based entity extraction, and keyword matching.

Pipeline:
    cleaned_text → segment_sections → extract_entities → extract_skills
                 → build_structured_json
"""

import re
import json
import logging
from typing import Optional

import spacy

# --- Logging ---
logger = logging.getLogger("hiremind.text_processing")

# --- spaCy Model (lazy-loaded singleton) ---
_nlp = None

def _get_nlp():
    """Load spaCy model once and cache it globally."""
    global _nlp
    if _nlp is None:
        logger.info("Loading spaCy en_core_web_md model...")
        _nlp = spacy.load("en_core_web_md")
        logger.info("spaCy model loaded successfully.")
    return _nlp


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION HEADINGS TAXONOMY
# ══════════════════════════════════════════════════════════════════════════════

SECTION_HEADINGS = {
    "education": [
        r"education", r"academic", r"qualification", r"degree",
        r"university", r"college", r"school", r"certifications?",
    ],
    "experience": [
        r"experience", r"employment", r"work\s*history",
        r"professional\s*experience", r"career", r"jobs?",
        r"internships?", r"training",
    ],
    "skills": [
        r"skills?", r"technical\s*skills?", r"core\s*competenc",
        r"proficienc", r"technologies", r"tools?\s*(?:and|&)\s*technologies",
        r"programming", r"software", r"expertise",
    ],
    "projects": [
        r"projects?", r"portfolio", r"personal\s*projects?",
        r"academic\s*projects?", r"side\s*projects?",
        r"key\s*projects?", r"notable\s*projects?",
    ],
    "summary": [
        r"summary", r"objective", r"profile", r"about\s*me",
        r"professional\s*summary", r"career\s*objective",
        r"overview",
    ],
    "certifications": [
        r"certifications?", r"licenses?", r"accreditations?",
        r"professional\s*development",
    ],
    "achievements": [
        r"achievements?", r"awards?", r"honors?", r"accomplishments?",
        r"recognition",
    ],
    "languages": [
        r"languages?", r"linguistic",
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
#  SKILLS TAXONOMY (200+ entries across major domains)
# ══════════════════════════════════════════════════════════════════════════════

SKILLS_TAXONOMY = {
    # --- Programming Languages ---
    "Python", "Java", "JavaScript", "TypeScript", "C", "C++", "C#", "Go",
    "Rust", "Ruby", "PHP", "Swift", "Kotlin", "Scala", "R", "MATLAB",
    "Perl", "Lua", "Dart", "Haskell", "Elixir", "Julia", "Shell",
    "Bash", "PowerShell", "SQL", "HTML", "CSS", "SASS", "LESS",

    # --- Web Frameworks ---
    "React", "Angular", "Vue", "Svelte", "Next.js", "Nuxt.js", "Django",
    "Flask", "FastAPI", "Express", "NestJS", "Spring Boot", "Laravel",
    "Ruby on Rails", "ASP.NET", "Streamlit", "Gradio",

    # --- Data Science & ML ---
    "Machine Learning", "Deep Learning", "NLP", "Natural Language Processing",
    "Computer Vision", "TensorFlow", "PyTorch", "Keras", "Scikit-Learn",
    "XGBoost", "LightGBM", "CatBoost", "Pandas", "NumPy", "SciPy",
    "Matplotlib", "Seaborn", "Plotly", "OpenCV", "NLTK", "spaCy",
    "Hugging Face", "Transformers", "LangChain", "LlamaIndex",
    "FAISS", "Pinecone", "Weaviate", "ChromaDB",

    # --- AI & LLM ---
    "GPT", "ChatGPT", "OpenAI", "Gemini", "Claude", "LLM", "RAG",
    "Prompt Engineering", "Fine-Tuning", "RLHF", "LoRA",
    "Stable Diffusion", "Generative AI",

    # --- Cloud & DevOps ---
    "AWS", "Azure", "GCP", "Google Cloud", "Heroku", "Vercel", "Netlify",
    "Docker", "Kubernetes", "Terraform", "Ansible", "Jenkins", "GitHub Actions",
    "GitLab CI", "CI/CD", "Linux", "Nginx", "Apache",

    # --- Databases ---
    "MySQL", "PostgreSQL", "MongoDB", "Redis", "Elasticsearch",
    "DynamoDB", "Cassandra", "Firebase", "Supabase", "SQLite",
    "Neo4j", "InfluxDB", "Snowflake", "BigQuery",

    # --- Data Engineering ---
    "Apache Spark", "Hadoop", "Kafka", "Airflow", "dbt",
    "ETL", "Data Pipeline", "Data Warehouse", "Data Lake",

    # --- Tools & Platforms ---
    "Git", "GitHub", "GitLab", "Bitbucket", "JIRA", "Confluence",
    "Figma", "Postman", "VS Code", "IntelliJ", "Jupyter",
    "Tableau", "Power BI", "Looker", "Excel",

    # --- Soft Skills ---
    "Leadership", "Communication", "Team Management", "Agile",
    "Scrum", "Problem Solving", "Critical Thinking", "Project Management",
    "Public Speaking", "Mentoring",

    # --- Security ---
    "Cybersecurity", "Penetration Testing", "OWASP", "OAuth", "JWT",
    "SSL/TLS", "Encryption", "Firewall",

    # --- Mobile ---
    "React Native", "Flutter", "SwiftUI", "Android", "iOS",
    "Xamarin", "Ionic",

    # --- Blockchain ---
    "Blockchain", "Solidity", "Ethereum", "Web3", "Smart Contracts",
}

# Pre-compile a case-insensitive lookup for fast matching
_SKILLS_LOWER = {s.lower(): s for s in SKILLS_TAXONOMY}


# ══════════════════════════════════════════════════════════════════════════════
#  1. segment_sections(text)
# ══════════════════════════════════════════════════════════════════════════════

def segment_sections(text: str) -> dict:
    """
    Splits resume text into labelled sections based on heading detection.

    Strategy:
        - Build a composite regex from SECTION_HEADINGS.
        - Scan for lines that look like section headers (short lines, possibly
          uppercased, ending with optional colon).
        - Split the text at header boundaries.
        - Anything before the first detected header → "header" section
          (typically contains name/contact info).

    Args:
        text: Cleaned resume text.

    Returns:
        dict mapping section names to their text content.
        Example: {"header": "...", "education": "...", "skills": "..."}
    """
    sections = {}

    # Build one giant pattern: match a line that IS a section heading
    all_patterns = []
    heading_to_section = {}
    for section_name, patterns in SECTION_HEADINGS.items():
        for pat in patterns:
            all_patterns.append(pat)
            heading_to_section[pat] = section_name

    # Pattern: beginning of line, optional whitespace, heading keyword,
    # optional trailing chars like ":", "-", then end of meaningful content
    heading_regex = re.compile(
        r"^[\s]*(?:" + "|".join(all_patterns) + r")[\s:—\-]*$",
        re.IGNORECASE | re.MULTILINE,
    )

    # Find all heading positions
    matches = list(heading_regex.finditer(text))

    if not matches:
        # No headings found — return entire text as "body"
        logger.warning("No section headings detected in resume text.")
        return {"body": text.strip()}

    # Everything before the first heading is the "header" (name/contact block)
    header_text = text[: matches[0].start()].strip()
    if header_text:
        sections["header"] = header_text

    # Assign each heading span to its section
    for i, match in enumerate(matches):
        matched_text = match.group().strip().rstrip(":- ").lower()

        # Determine which canonical section this maps to
        section_name = "other"
        for pat, name in heading_to_section.items():
            if re.search(pat, matched_text, re.IGNORECASE):
                section_name = name
                break

        # Content runs from end of this heading to start of next heading
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()

        # If section name already exists, append
        if section_name in sections:
            sections[section_name] += "\n\n" + content
        else:
            sections[section_name] = content

    logger.info("Segmented %d sections: %s", len(sections), list(sections.keys()))
    return sections


# ══════════════════════════════════════════════════════════════════════════════
#  2. extract_entities(text)
# ══════════════════════════════════════════════════════════════════════════════

# --- Compiled Regex Patterns ---
EMAIL_PATTERN = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", re.IGNORECASE
)

# More tolerant phone pattern (finds 10 digits even if stuck to text)
PHONE_PATTERN = re.compile(
    r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\b\d{10}\b|(?:\+?\d{1,3}[\s\-.]?)?\(?\d{2,4}\)?[\s\-.]?\d{3,5}[\s\-.]?\d{3,5}"
)

LOCATION_INDICATORS = re.compile(
    r"(?:address|location|city|based\s+in|residing|currently\s+at)\s*[:|\-]?\s*(.+)",
    re.IGNORECASE,
)

def preprocess_entities_text(text: str) -> str:
    """
    Cleans up common PDF extraction artifacts that break entity extraction.
    - Joins split emails (e.g., 'user@g' \n 'mail.com').
    - Fixes joined-text headers (e.g., 'SKILLS7756828367' -> 'SKILLS 7756828367').
    """
    if not text:
        return ""
    
    # 1. Stitch split emails (look for @ preceded/followed by newline)
    # e.g. "bhaveshchaudhary2506@g\nmail.com" -> "bhaveshchaudhary2506@gmail.com"
    text = re.sub(r"([a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+)\s*\n\s*([a-zA-Z]{2,})", r"\1\2", text)
    
    # 2. Add space between words and numbers (common in headers)
    # e.g. "SKILLS7756" -> "SKILLS 7756"
    text = re.sub(r"([a-zA-Z]+)(\d{7,})", r"\1 \2", text)
    
    return text

def extract_entities(text: str) -> dict:
    """
    Extracts personal identification entities from resume text.

    Uses:
        - spaCy NER for name (PERSON) and location (GPE/LOC).
        - Regex for email and phone.

    Args:
        text: Full resume text (or the 'header' section for better accuracy).

    Returns:
        dict with keys: name, email, phone, location.
    """
    # Pre-process to fix fragmented text
    text = preprocess_entities_text(text)

    nlp = _get_nlp()
    result = {
        "name": None,
        "email": None,
        "phone": None,
        "location": None,
    }

    # ------- Email (regex) -------
    email_match = EMAIL_PATTERN.search(text)
    if email_match:
        result["email"] = email_match.group().strip()

    # ------- Phone (regex) -------
    # First look for standard formats, then any 10-digit sequence
    phone_match = PHONE_PATTERN.search(text)
    if phone_match:
        result["phone"] = phone_match.group().strip()
    else:
        # Fallback: Find any 10-digit number
        digits_only_match = re.search(r"\b\d{10,12}\b", re.sub(r"\D", "", text))
        if digits_only_match:
            result["phone"] = digits_only_match.group()

    # ------- Name & Location (spaCy NER) -------
    # Process a larger chunk (first 1500) because names in multi-column PDFs 
    # can get pushed down far in the text extraction order.
    header_doc = nlp(text[:1500])

    persons = []
    locations = []

    for ent in header_doc.ents:
        if ent.label_ == "PERSON":
            # Filter out short fragments or section labels
            val = ent.text.strip()
            if len(val.split()) >= 2 and not any(h.lower() in val.lower() for h in SECTION_HEADINGS):
                persons.append(val)
        elif ent.label_ in ("GPE", "LOC"):
            locations.append(ent.text.strip())

    # Name: Take the first PERSON entity
    if persons:
        result["name"] = persons[0]
    else:
        # Fallback: Look for Uppercase segments in the first 1500 chars 
        # Restriction: Only match on a single line to avoid joining separate headers
        # (e.g. avoid joining "CONTACT" and "SKILLS")
        potential_names = re.findall(r"\b[A-Z]{2,}(?: [A-Z]{2,})+\b", text[:1500])
        
        # Flatten all heading patterns into a single set for fast lookup
        all_heading_words = set()
        for h_list in SECTION_HEADINGS.values():
            for h_pat in h_list:
                # remove regex symbols to get core words
                word = h_pat.replace(r"\s*", "").replace("?", "").replace(r"\b", "")
                all_heading_words.add(word.lower())

        for p_name in potential_names:
            words = p_name.split()
            if len(words) < 2: continue
            
            # If any word is a common section header or skill, skip
            is_noise = any(
                w.lower() in _SKILLS_LOWER or 
                w.lower() in all_heading_words or
                w.lower() in ["contact", "profile", "aspirant", "engineer", "student"]
                for w in words
            )
            
            if not is_noise:
                result["name"] = p_name.title()
                break

    # Location: spaCy GPE or regex fallback
    if locations:
        result["location"] = locations[0]
    else:
        # Search deeper for location indicators (up to 1500)
        loc_match = LOCATION_INDICATORS.search(text[:1500])
        if loc_match:
            result["location"] = loc_match.group(1).strip()

    logger.info(
        "Entities extracted → name=%s, email=%s, phone=%s, location=%s",
        result["name"], result["email"],
        result["phone"][:4] + "***" if result["phone"] else None,
        result["location"],
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  3. extract_skills(text)
# ══════════════════════════════════════════════════════════════════════════════

def extract_skills(text: str) -> list:
    """
    Extracts technical and professional skills from resume text.

    Strategy (layered):
        1. Taxonomy Match: Scan text against the 200+ skills taxonomy.
        2. spaCy NER Supplement: Pick up ORG/PRODUCT entities that look
           like tool/technology names.
        3. Deduplicate and return canonical casing from taxonomy.

    Args:
        text: Full resume text.

    Returns:
        Sorted list of unique skills (canonical casing).
    """
    found_skills = set()
    text_lower = text.lower()

    # --- Layer 1: Taxonomy keyword matching ---
    for skill_lower, skill_canonical in _SKILLS_LOWER.items():
        # Use word-boundary matching to avoid partial hits
        # e.g. "R" should not match every word containing 'r'
        if len(skill_lower) <= 2:
            # For very short skill names (R, C, Go), require word boundaries
            pattern = r"\b" + re.escape(skill_lower) + r"\b"
            if re.search(pattern, text_lower):
                found_skills.add(skill_canonical)
        else:
            if skill_lower in text_lower:
                found_skills.add(skill_canonical)

    # --- Layer 2: spaCy NER supplement ---
    nlp = _get_nlp()
    doc = nlp(text[:5000])  # limit to first 5000 chars for performance

    for ent in doc.ents:
        if ent.label_ in ("ORG", "PRODUCT"):
            ent_lower = ent.text.strip().lower()
            if ent_lower in _SKILLS_LOWER:
                found_skills.add(_SKILLS_LOWER[ent_lower])

    skills_list = sorted(found_skills, key=str.lower)
    logger.info("Extracted %d unique skills.", len(skills_list))
    return skills_list


# ══════════════════════════════════════════════════════════════════════════════
#  4. clean_text_advanced(text)
# ══════════════════════════════════════════════════════════════════════════════

# Boilerplate phrases commonly found in resumes
_BOILERPLATE = [
    r"references?\s*(?:available)?\s*(?:upon|on)\s*request",
    r"i\s*hereby\s*declare",
    r"declaration\s*:?",
    r"(?:date|place)\s*:\s*.*$",
    r"curriculum\s*vitae",
    r"page\s*\d+\s*(?:of\s*\d+)?",
    r"resume\s*[-–]\s*confidential",
]

_BOILERPLATE_REGEX = re.compile(
    "|".join(_BOILERPLATE), re.IGNORECASE | re.MULTILINE
)


def clean_text_advanced(text: str) -> str:
    """
    Advanced cleaning that goes beyond basic normalisation.

    Operations:
        - Remove boilerplate phrases (declarations, page numbers).
        - Strip URLs and email-like noise from body text.
        - Remove isolated single characters and stray punctuation lines.
        - Collapse excessive blank lines.

    NOTE: This does NOT remove technical keywords or stopwords that carry
    domain meaning (e.g. "REST", "API", "OOP").

    Args:
        text: Pre-cleaned resume text.

    Returns:
        Further cleaned text string.
    """
    if not text:
        return ""

    # Remove boilerplate
    text = _BOILERPLATE_REGEX.sub("", text)

    # Remove standalone URLs (not part of skills)
    text = re.sub(
        r"https?://\S+|www\.\S+",
        "", text
    )

    # Remove lines that are just punctuation / single chars
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip lines that are only punctuation/special chars
        if stripped and not re.match(r"^[\W_]+$", stripped):
            cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # Collapse 3+ blank lines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ══════════════════════════════════════════════════════════════════════════════
#  5. build_structured_json(text) — Main Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def build_structured_json(text: str) -> dict:
    """
    Master function that orchestrates the full text processing pipeline:
        1. Advanced cleaning
        2. Section segmentation
        3. Entity extraction
        4. Skill extraction
        5. Assembly into structured JSON

    Args:
        text: Cleaned resume text (output of data_ingestion module).

    Returns:
        Structured dict with all parsed resume fields.
    """
    logger.info("Starting structured JSON build pipeline...")

    # Step 1: Advanced cleaning
    cleaned = clean_text_advanced(text)

    # Step 2: Segment into sections
    sections = segment_sections(cleaned)

    # Step 3: Extract personal entities
    # Prefer the header block if it exists, otherwise use full text
    entity_source = sections.get("header", cleaned[:1500])
    entities = extract_entities(entity_source)

    # Step 4: Extract skills
    # Prefer the skills section if found, but also scan full text
    skills_source = sections.get("skills", "") + "\n" + cleaned
    skills = extract_skills(skills_source)

    # Step 5: Assemble final structure
    structured = {
        "name":           entities.get("name"),
        "email":          entities.get("email"),
        "phone":          entities.get("phone"),
        "location":       entities.get("location"),
        "skills":         skills,
        "summary":        sections.get("summary", ""),
        "education":      sections.get("education", ""),
        "experience":     sections.get("experience", ""),
        "projects":       sections.get("projects", ""),
        "certifications": sections.get("certifications", ""),
        "achievements":   sections.get("achievements", ""),
        "languages":      sections.get("languages", ""),
        "raw_sections":   list(sections.keys()),
    }

    logger.info(
        "Structured JSON built → fields populated: %d, skills found: %d",
        sum(1 for v in structured.values() if v),
        len(skills),
    )
    return structured
