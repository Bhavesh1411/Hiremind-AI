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
#  SKILL KNOWLEDGE GRAPH
# ══════════════════════════════════════════════════════════════════════════════
# Maps skill domains → prerequisite chains, related skills, learning resources

SKILL_KNOWLEDGE_GRAPH = {
    # ── AI / ML ────────────────────────────────────────────────────────────
    "machine learning": {
        "prerequisites": ["python", "statistics", "linear algebra"],
        "courses":       ["ML Specialization (Coursera - Andrew Ng)", "Hands-On ML with Scikit-Learn (Book)"],
        "projects":      ["Build a spam classifier", "Train a housing price predictor"],
        "certifications": ["Google ML Engineer Certificate", "AWS ML Specialty"],
        "domain":        "AI/ML",
    },
    "deep learning": {
        "prerequisites": ["machine learning", "python", "calculus"],
        "courses":       ["Deep Learning Specialization (Coursera)", "Fast.ai Practical DL"],
        "projects":      ["Image classifier with CNN", "Text generator with RNN"],
        "certifications": ["TensorFlow Developer Certificate"],
        "domain":        "AI/ML",
    },
    "natural language processing": {
        "prerequisites": ["python", "machine learning", "statistics"],
        "courses":       ["NLP with Transformers (Hugging Face)", "CS224N (Stanford)"],
        "projects":      ["Sentiment analysis chatbot", "Document summarizer"],
        "certifications": ["DeepLearning.AI NLP Specialization"],
        "domain":        "AI/ML",
    },
    "nlp": {
        "prerequisites": ["python", "machine learning"],
        "courses":       ["Hugging Face NLP Course", "spaCy Advanced NLP"],
        "projects":      ["Named entity recognizer", "Resume parser"],
        "certifications": ["DeepLearning.AI NLP Specialization"],
        "domain":        "AI/ML",
    },
    "computer vision": {
        "prerequisites": ["deep learning", "python", "linear algebra"],
        "courses":       ["CS231N (Stanford)", "OpenCV Bootcamp"],
        "projects":      ["Object detection system", "Face recognition app"],
        "certifications": ["TensorFlow Developer Certificate"],
        "domain":        "AI/ML",
    },
    "tensorflow": {
        "prerequisites": ["python", "deep learning"],
        "courses":       ["TensorFlow in Practice (Coursera)", "TF Official Tutorials"],
        "projects":      ["Image classification API", "Time series forecaster"],
        "certifications": ["TensorFlow Developer Certificate"],
        "domain":        "AI/ML",
    },
    "pytorch": {
        "prerequisites": ["python", "deep learning"],
        "courses":       ["PyTorch for Deep Learning (Udacity)", "Fast.ai"],
        "projects":      ["GAN for image generation", "Custom object detector"],
        "certifications": [],
        "domain":        "AI/ML",
    },
    "data science": {
        "prerequisites": ["python", "statistics", "sql"],
        "courses":       ["IBM Data Science Professional (Coursera)", "DataCamp tracks"],
        "projects":      ["EDA on a public dataset", "Predictive analytics dashboard"],
        "certifications": ["IBM Data Science Professional Certificate"],
        "domain":        "Data",
    },

    # ── Data Engineering ───────────────────────────────────────────────────
    "sql": {
        "prerequisites": [],
        "courses":       ["SQL for Data Science (Coursera)", "Mode Analytics SQL Tutorial"],
        "projects":      ["Design a normalized database", "Write complex analytical queries"],
        "certifications": ["Oracle SQL Certified Associate"],
        "domain":        "Data",
    },
    "spark": {
        "prerequisites": ["python", "sql"],
        "courses":       ["Big Data with PySpark (Udemy)", "Databricks Academy"],
        "projects":      ["ETL pipeline with PySpark", "Real-time streaming analytics"],
        "certifications": ["Databricks Certified Associate"],
        "domain":        "Data",
    },
    "hadoop": {
        "prerequisites": ["linux", "sql"],
        "courses":       ["Hadoop Platform and Application Framework (Coursera)"],
        "projects":      ["MapReduce word count", "HDFS data lake setup"],
        "certifications": ["Cloudera CCA Data Analyst"],
        "domain":        "Data",
    },
    "power bi": {
        "prerequisites": ["sql", "excel"],
        "courses":       ["Microsoft Power BI Data Analyst (Coursera)", "PL-300 Prep"],
        "projects":      ["Sales analytics dashboard", "HR metrics tracker"],
        "certifications": ["Microsoft PL-300 Certification"],
        "domain":        "Data",
    },
    "tableau": {
        "prerequisites": ["sql"],
        "courses":       ["Tableau Desktop Specialist Prep", "Tableau Public Portfolio"],
        "projects":      ["Interactive COVID dashboard", "Customer segmentation viz"],
        "certifications": ["Tableau Desktop Specialist"],
        "domain":        "Data",
    },

    # ── Web Development ────────────────────────────────────────────────────
    "react": {
        "prerequisites": ["javascript", "html", "css"],
        "courses":       ["React Complete Guide (Udemy - Schwarzmüller)", "React Docs Beta"],
        "projects":      ["Todo app with state management", "E-commerce storefront"],
        "certifications": ["Meta Front-End Developer (Coursera)"],
        "domain":        "Web",
    },
    "angular": {
        "prerequisites": ["typescript", "javascript", "html"],
        "courses":       ["Angular Complete Guide (Udemy)", "Angular Official Tour of Heroes"],
        "projects":      ["Project management app", "Admin dashboard"],
        "certifications": [],
        "domain":        "Web",
    },
    "node.js": {
        "prerequisites": ["javascript"],
        "courses":       ["Node.js Complete Guide (Udemy)", "The Odin Project Backend"],
        "projects":      ["REST API for a blog", "Real-time chat server"],
        "certifications": ["OpenJS Node.js Application Developer"],
        "domain":        "Web",
    },
    "django": {
        "prerequisites": ["python"],
        "courses":       ["Django for Beginners (Book)", "Django REST Framework Tutorial"],
        "projects":      ["Blog with user auth", "REST API for task manager"],
        "certifications": [],
        "domain":        "Web",
    },
    "flask": {
        "prerequisites": ["python"],
        "courses":       ["Flask Mega Tutorial (Miguel Grinberg)", "Explore Flask (Book)"],
        "projects":      ["URL shortener", "Microservice with Flask"],
        "certifications": [],
        "domain":        "Web",
    },
    "javascript": {
        "prerequisites": [],
        "courses":       ["JavaScript.info", "Eloquent JavaScript (Book)"],
        "projects":      ["Interactive quiz app", "Browser-based game"],
        "certifications": [],
        "domain":        "Web",
    },
    "typescript": {
        "prerequisites": ["javascript"],
        "courses":       ["TypeScript Handbook (Official)", "Total TypeScript (Matt Pocock)"],
        "projects":      ["Type-safe API client", "CLI tool in TS"],
        "certifications": [],
        "domain":        "Web",
    },
    "html": {
        "prerequisites": [],
        "courses":       ["MDN Web Docs HTML Guide", "freeCodeCamp Responsive Web"],
        "projects":      ["Portfolio website", "Semantic landing page"],
        "certifications": [],
        "domain":        "Web",
    },
    "css": {
        "prerequisites": [],
        "courses":       ["CSS for JS Devs (Josh Comeau)", "MDN CSS Guide"],
        "projects":      ["Responsive dashboard layout", "CSS art challenge"],
        "certifications": [],
        "domain":        "Web",
    },

    # ── Cloud & DevOps ─────────────────────────────────────────────────────
    "aws": {
        "prerequisites": ["linux", "networking"],
        "courses":       ["AWS Solutions Architect Prep (A Cloud Guru)", "AWS Skill Builder"],
        "projects":      ["Deploy a 3-tier app on AWS", "Serverless API with Lambda"],
        "certifications": ["AWS Solutions Architect Associate", "AWS Cloud Practitioner"],
        "domain":        "Cloud",
    },
    "azure": {
        "prerequisites": ["networking"],
        "courses":       ["AZ-900 Fundamentals (Microsoft Learn)", "AZ-204 Developer"],
        "projects":      ["Deploy app via Azure DevOps", "Azure Functions chatbot"],
        "certifications": ["AZ-900 Azure Fundamentals", "AZ-104 Administrator"],
        "domain":        "Cloud",
    },
    "gcp": {
        "prerequisites": ["linux"],
        "courses":       ["Google Cloud Fundamentals (Coursera)", "GCP Skill Badges"],
        "projects":      ["BigQuery analytics pipeline", "Cloud Run microservice"],
        "certifications": ["Google Cloud Associate Cloud Engineer"],
        "domain":        "Cloud",
    },
    "docker": {
        "prerequisites": ["linux"],
        "courses":       ["Docker Mastery (Udemy - Bret Fisher)", "Docker Official Getting Started"],
        "projects":      ["Containerize a web app", "Multi-container setup with Compose"],
        "certifications": ["Docker Certified Associate"],
        "domain":        "DevOps",
    },
    "kubernetes": {
        "prerequisites": ["docker", "linux", "networking"],
        "courses":       ["CKA Prep (KodeKloud)", "Kubernetes the Hard Way (Kelsey Hightower)"],
        "projects":      ["Deploy microservices on K8s", "Auto-scaling demo cluster"],
        "certifications": ["CKA - Certified Kubernetes Administrator"],
        "domain":        "DevOps",
    },
    "ci/cd": {
        "prerequisites": ["git"],
        "courses":       ["GitHub Actions Tutorial", "Jenkins Pipeline Fundamentals"],
        "projects":      ["Automated test + deploy pipeline", "Multi-branch CI workflow"],
        "certifications": ["GitHub Actions Certification"],
        "domain":        "DevOps",
    },

    # ── Programming Languages ──────────────────────────────────────────────
    "python": {
        "prerequisites": [],
        "courses":       ["Python for Everybody (Coursera)", "Automate the Boring Stuff"],
        "projects":      ["Web scraper", "CLI task manager"],
        "certifications": ["PCEP Python Certified Entry Programmer"],
        "domain":        "Programming",
    },
    "java": {
        "prerequisites": [],
        "courses":       ["Java Programming MOOC (Helsinki)", "Head First Java (Book)"],
        "projects":      ["Banking system simulator", "REST API with Spring Boot"],
        "certifications": ["Oracle Java SE Certified"],
        "domain":        "Programming",
    },
    "c++": {
        "prerequisites": [],
        "courses":       ["C++ Primer (Book)", "LearnCpp.com"],
        "projects":      ["Memory manager", "Simple game engine"],
        "certifications": [],
        "domain":        "Programming",
    },
    "go": {
        "prerequisites": [],
        "courses":       ["Go by Example", "Let's Go (Book - Alex Edwards)"],
        "projects":      ["CLI tool", "HTTP server from scratch"],
        "certifications": [],
        "domain":        "Programming",
    },
    "rust": {
        "prerequisites": [],
        "courses":       ["The Rust Book (Official)", "Rustlings exercises"],
        "projects":      ["CLI grep clone", "Simple web server"],
        "certifications": [],
        "domain":        "Programming",
    },

    # ── Other Tools ────────────────────────────────────────────────────────
    "git": {
        "prerequisites": [],
        "courses":       ["Git & GitHub Crash Course (Traversy Media)", "Pro Git (Book)"],
        "projects":      ["Open source contribution", "Branching strategy demo"],
        "certifications": [],
        "domain":        "Tools",
    },
    "linux": {
        "prerequisites": [],
        "courses":       ["Linux Command Line Basics (Udacity)", "The Linux Foundation Intro"],
        "projects":      ["Shell scripting automation", "Server setup from scratch"],
        "certifications": ["CompTIA Linux+", "LFCS"],
        "domain":        "Tools",
    },
    "excel": {
        "prerequisites": [],
        "courses":       ["Excel Skills for Business (Coursera)", "ExcelJet Formulas"],
        "projects":      ["Financial model spreadsheet", "Automated reporting template"],
        "certifications": ["Microsoft MO-200 Excel Associate"],
        "domain":        "Tools",
    },
}

# ── Role Mapping: skill clusters → alternative job titles ──────────────────
ROLE_SKILL_MAP = {
    "Data Analyst": {"sql", "python", "excel", "power bi", "tableau", "statistics"},
    "Data Scientist": {"python", "machine learning", "statistics", "sql", "data science"},
    "ML Engineer": {"python", "machine learning", "deep learning", "tensorflow", "pytorch", "docker"},
    "Backend Developer": {"python", "java", "node.js", "sql", "django", "flask", "docker"},
    "Frontend Developer": {"javascript", "react", "angular", "html", "css", "typescript"},
    "Full Stack Developer": {"javascript", "react", "node.js", "python", "sql", "html", "css", "docker"},
    "DevOps Engineer": {"docker", "kubernetes", "aws", "linux", "ci/cd", "git"},
    "Cloud Engineer": {"aws", "azure", "gcp", "docker", "kubernetes", "linux"},
    "NLP Engineer": {"python", "nlp", "natural language processing", "deep learning", "pytorch"},
    "Computer Vision Engineer": {"python", "computer vision", "deep learning", "tensorflow", "pytorch"},
    "Business Analyst": {"sql", "excel", "power bi", "tableau"},
    "AI Research Scientist": {"python", "deep learning", "machine learning", "pytorch", "statistics"},
}


# ══════════════════════════════════════════════════════════════════════════════
#  1. identify_skill_gaps(candidate_skills, jd_skills)
# ══════════════════════════════════════════════════════════════════════════════

def identify_skill_gaps(
    candidate_skills: list[str],
    jd_skills: list[str],
) -> dict:
    """
    Compare candidate skills against JD requirements.

    Uses case-insensitive exact matching plus substring matching
    (e.g., "machine learning" matches "ml" or "Machine Learning").

    Args:
        candidate_skills: Skills from parsed resume.
        jd_skills:        Skills from job description.

    Returns:
        {
            "matched_skills":  [...],
            "missing_skills":  [...],
            "extra_skills":    [...],  # candidate has but JD doesn't need
            "match_rate":      float (0-1),
        }
    """
    cand_lower = {s.lower().strip() for s in candidate_skills if s.strip()}
    jd_lower   = {s.lower().strip() for s in jd_skills if s.strip()}

    # Fuzzy matching: also check substring containment
    matched = set()
    for jd_skill in jd_lower:
        for cand_skill in cand_lower:
            if jd_skill == cand_skill or jd_skill in cand_skill or cand_skill in jd_skill:
                matched.add(jd_skill)
                break

    missing = jd_lower - matched
    extra   = cand_lower - {s for s in cand_lower for m in matched if m == s or m in s or s in m}

    match_rate = len(matched) / len(jd_lower) if jd_lower else 1.0

    logger.debug("Skill gaps: %d matched, %d missing, rate=%.2f", len(matched), len(missing), match_rate)

    return {
        "matched_skills": sorted(matched),
        "missing_skills": sorted(missing),
        "extra_skills":   sorted(extra),
        "match_rate":     round(match_rate, 3),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  2. map_skills_to_learning_paths(missing_skills)
# ══════════════════════════════════════════════════════════════════════════════

def map_skills_to_learning_paths(missing_skills: list[str]) -> list[dict]:
    """
    Maps each missing skill to its knowledge graph entry for learning paths.

    If a skill isn't in the graph, generates a sensible generic entry.

    Args:
        missing_skills: List of missing skill strings (lowercase).

    Returns:
        List of dicts with: skill, courses, projects, certifications, prerequisites.
    """
    paths = []
    for skill in missing_skills:
        skill_key = skill.lower().strip()
        if skill_key in SKILL_KNOWLEDGE_GRAPH:
            entry = SKILL_KNOWLEDGE_GRAPH[skill_key]
            paths.append({
                "skill":          skill,
                "domain":         entry.get("domain", "General"),
                "prerequisites":  entry.get("prerequisites", []),
                "courses":        entry.get("courses", []),
                "projects":       entry.get("projects", []),
                "certifications": entry.get("certifications", []),
            })
        else:
            # Generic recommendation for unknown skills
            paths.append({
                "skill":          skill,
                "domain":         "General",
                "prerequisites":  [],
                "courses":        [f"Search Udemy/Coursera for '{skill}' courses"],
                "projects":       [f"Build a portfolio project using {skill}"],
                "certifications": [f"Check vendor certifications for {skill}"],
            })

    return paths


# ══════════════════════════════════════════════════════════════════════════════
#  3. generate_learning_recommendations(skill_paths)
# ══════════════════════════════════════════════════════════════════════════════

def generate_learning_recommendations(skill_paths: list[dict]) -> list[dict]:
    """
    Transforms skill paths into user-friendly recommendation cards.

    Prioritizes: prerequisites first → courses → projects → certifications.

    Args:
        skill_paths: Output of map_skills_to_learning_paths().

    Returns:
        List of recommendation dicts with: skill, priority, resources.
    """
    recs = []
    for i, path in enumerate(skill_paths):
        resources = []

        prereqs = path.get("prerequisites", [])
        if prereqs:
            resources.append(f"📚 Prerequisites: {', '.join(prereqs)}")

        for course in path.get("courses", [])[:2]:
            resources.append(f"🎓 Course: {course}")

        for project in path.get("projects", [])[:1]:
            resources.append(f"🛠️ Project: {project}")

        for cert in path.get("certifications", [])[:1]:
            resources.append(f"📜 Certification: {cert}")

        recs.append({
            "skill":     path["skill"],
            "domain":    path.get("domain", "General"),
            "priority":  i + 1,
            "resources": resources,
        })

    return recs


# ══════════════════════════════════════════════════════════════════════════════
#  4. suggest_alternative_roles(candidate_skills, jd_skills)
# ══════════════════════════════════════════════════════════════════════════════

def suggest_alternative_roles(
    candidate_skills: list[str],
    jd_skills: list[str],
) -> list[dict]:
    """
    Suggests alternative job roles the candidate is better suited for.

    Logic:
        - For each role in ROLE_SKILL_MAP, compute overlap with candidate skills
        - Rank by overlap percentage
        - Return top 4 roles (excluding roles the JD already targets)

    Args:
        candidate_skills: Skills from parsed resume.
        jd_skills:        Skills from job description.

    Returns:
        List of dicts: role, match_pct, matched_skills, missing_count.
    """
    cand_lower = {s.lower().strip() for s in candidate_skills}

    role_scores = []
    for role, required_skills in ROLE_SKILL_MAP.items():
        matched = set()
        for req in required_skills:
            for cand in cand_lower:
                if req == cand or req in cand or cand in req:
                    matched.add(req)
                    break

        match_pct = len(matched) / len(required_skills) if required_skills else 0
        missing_count = len(required_skills) - len(matched)

        role_scores.append({
            "role":           role,
            "match_pct":      round(match_pct * 100, 1),
            "matched_skills": sorted(matched),
            "missing_count":  missing_count,
        })

    # Sort by match percentage descending
    role_scores.sort(key=lambda x: x["match_pct"], reverse=True)

    # Return top 4 with at least 30% match
    return [r for r in role_scores if r["match_pct"] >= 30][:4]


# ══════════════════════════════════════════════════════════════════════════════
#  5. generate_improvement_summary(gap_data, recs, alt_roles)
# ══════════════════════════════════════════════════════════════════════════════

def generate_improvement_summary(
    gap_data: dict,
    recs: list[dict],
    alt_roles: list[dict],
) -> str:
    """
    Generates a concise, human-readable improvement summary.

    Args:
        gap_data:  Output of identify_skill_gaps().
        recs:      Output of generate_learning_recommendations().
        alt_roles: Output of suggest_alternative_roles().

    Returns:
        Multi-line summary string.
    """
    match_rate = gap_data.get("match_rate", 0)
    matched    = gap_data.get("matched_skills", [])
    missing    = gap_data.get("missing_skills", [])

    lines = []

    # Opening assessment
    if match_rate >= 0.8:
        lines.append(
            f"🟢 **Strong Match** — The candidate already has {len(matched)} of the required skills "
            f"({match_rate*100:.0f}% coverage). Only {len(missing)} gap(s) to close."
        )
    elif match_rate >= 0.5:
        lines.append(
            f"🟡 **Moderate Match** — {len(matched)} skills align with the role "
            f"({match_rate*100:.0f}% coverage), but {len(missing)} key skill(s) are missing."
        )
    else:
        lines.append(
            f"🔴 **Significant Gap** — Only {match_rate*100:.0f}% skill coverage. "
            f"The candidate needs to develop {len(missing)} additional skill(s) for this role."
        )

    # Top priority skills
    if missing:
        top_missing = ", ".join(f"**{s}**" for s in missing[:5])
        lines.append(f"\n**Priority Skills to Learn:** {top_missing}")

    # Learning path summary
    if recs:
        domains = set(r.get("domain", "General") for r in recs)
        lines.append(
            f"\n**Recommended Focus Areas:** {', '.join(sorted(domains))}"
        )

    # Alternative roles
    if alt_roles and match_rate < 0.7:
        top_roles = ", ".join(f"*{r['role']}* ({r['match_pct']}%)" for r in alt_roles[:3])
        lines.append(
            f"\n**Alternative Roles to Consider:** {top_roles}"
        )

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  5b. Optional LLM Enhancement
# ══════════════════════════════════════════════════════════════════════════════

def enhance_summary_with_llm(
    summary: str,
    candidate_skills: list[str],
    missing_skills: list[str],
    api_key: str = "",
    provider: str = "gemini",
) -> Optional[str]:
    """
    Optional: Uses LLM to create a more personalized improvement summary.

    Args:
        summary:          Rule-based summary string.
        candidate_skills: What the candidate has.
        missing_skills:   What the candidate lacks.
        api_key:          API key for the LLM.
        provider:         "gemini" or "openai".

    Returns:
        Enhanced summary string, or None if skipped/failed.
    """
    if not api_key or not missing_skills:
        return None

    try:
        from modules.llm_analysis import call_llm

        prompt = f"""You are an expert career coach. A candidate has these skills: {', '.join(candidate_skills[:15])}.

They are missing these skills for their target role: {', '.join(missing_skills[:10])}.

Here is an automated assessment:
{summary}

Write a SHORT (4-5 sentences), encouraging, and actionable career improvement plan.
Include: what to learn first, a realistic timeline (weeks/months), and one specific project to build.
Be practical, not generic. Do NOT use bullet points — write in paragraph form."""

        result = call_llm(prompt, provider=provider, api_key=api_key)
        logger.info("LLM career summary generated: %d chars.", len(result))
        return result.strip()

    except Exception as e:
        logger.warning("LLM career enhancement skipped: %s", e)
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  6. build_recommendation_output() — Full Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def build_recommendation_output(
    candidate_skills: list[str],
    jd_skills: list[str],
    api_key: str = "",
    provider: str = "gemini",
    use_llm: bool = False,
) -> dict:
    """
    Full recommendation pipeline orchestrator.

    Args:
        candidate_skills: From parsed resume.
        jd_skills:        From JD or ATS analysis.
        api_key:          Optional API key for LLM enhancement.
        provider:         "gemini" or "openai".
        use_llm:          Whether to use LLM for personalized summary.

    Returns:
        {
            "status":                "success" | "error",
            "missing_skills":        [...],
            "matched_skills":        [...],
            "extra_skills":          [...],
            "match_rate":            float,
            "recommended_learning":  [...],
            "alternative_roles":     [...],
            "improvement_summary":   str,
            "llm_career_plan":       str | None,
        }
    """
    result = {
        "status":               "error",
        "missing_skills":       [],
        "matched_skills":       [],
        "extra_skills":         [],
        "match_rate":           0,
        "recommended_learning": [],
        "alternative_roles":    [],
        "improvement_summary":  "",
        "llm_career_plan":      None,
    }

    try:
        # Step 1: Skill Gap Analysis
        gap_data = identify_skill_gaps(candidate_skills, jd_skills)

        # Step 2: Map missing skills to learning paths
        skill_paths = map_skills_to_learning_paths(gap_data["missing_skills"])

        # Step 3: Generate learning recommendations
        recs = generate_learning_recommendations(skill_paths)

        # Step 4: Suggest alternative roles
        alt_roles = suggest_alternative_roles(candidate_skills, jd_skills)

        # Step 5: Generate improvement summary
        summary = generate_improvement_summary(gap_data, recs, alt_roles)

        # Step 5b: Optional LLM enhancement
        llm_plan = None
        if use_llm and api_key and gap_data["missing_skills"]:
            llm_plan = enhance_summary_with_llm(
                summary, candidate_skills,
                gap_data["missing_skills"],
                api_key, provider,
            )

        result.update({
            "status":               "success",
            "missing_skills":       gap_data["missing_skills"],
            "matched_skills":       gap_data["matched_skills"],
            "extra_skills":         gap_data["extra_skills"],
            "match_rate":           gap_data["match_rate"],
            "recommended_learning": recs,
            "alternative_roles":    alt_roles,
            "improvement_summary":  summary,
            "llm_career_plan":      llm_plan,
        })

        logger.info(
            "Recommendations built: %d missing, %d recs, %d alt roles.",
            len(gap_data["missing_skills"]), len(recs), len(alt_roles),
        )

    except Exception as e:
        result["improvement_summary"] = f"Recommendation engine error: {str(e)}"
        logger.exception("Error in build_recommendation_output.")

    return result
