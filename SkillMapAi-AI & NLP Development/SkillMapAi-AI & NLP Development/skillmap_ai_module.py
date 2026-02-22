"""
SkillMap AI Module
==================
AI & NLP component for the SkillMap AI hackathon project.
Uses sentence-transformers for skill–job similarity and identifies skill gaps.
"""

from pathlib import Path
import re
import ast
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Optional imports for CV parsing (graceful fallback if not installed)
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from pdfminer.high_level import extract_text as pdf_extract_text
except ImportError:
    pdf_extract_text = None

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None


# ---------------------------------------------------------------------------
# Configuration & state
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_JOBS_PATH = BASE_DIR / "jobs.csv"

# Column names we accept (user spec: "Job Title", "Required Skills")
# Also support alternate names from processed datasets
TITLE_COLUMNS = ("Job Title", "job_title", "title")
SKILLS_COLUMNS = ("Required Skills", "job_skill_set", "required_skills", "skills")

_model = None
_jobs_df = None
_job_embeddings = None
_all_skills_vocab = None


def _get_model():
    """Load the pre-trained sentence embedding model (lazy singleton).
    Model: 'sentence-transformers/all-MiniLM-L6-v2' produces 384-dim vectors
    suitable for semantic similarity between skill phrases.
    """
    global _model
    if _model is None:
        if SentenceTransformer is None:
            raise ImportError("Install sentence-transformers: pip install sentence-transformers")
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model


def _normalize_skills(skill_list):
    """Convert to list of lowercased, stripped skill strings."""
    if isinstance(skill_list, str):
        # Try Python list literal first (e.g. "['Python', 'SQL']")
        s = skill_list.strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            try:
                skill_list = ast.literal_eval(s)
            except (ValueError, SyntaxError):
                skill_list = [x.strip() for x in s.split(",")]
        else:
            skill_list = [x.strip() for x in skill_list.split(",")]
    return [str(s).lower().strip() for s in skill_list if str(s).strip()]


def _detect_csv_columns(df):
    """Detect job title and skills column names from the dataframe."""
    title_col = None
    skills_col = None
    for c in df.columns:
        if c in TITLE_COLUMNS:
            title_col = c
        if c in SKILLS_COLUMNS:
            skills_col = c
    if title_col is None:
        title_col = df.columns[0]
    if skills_col is None:
        for c in df.columns:
            if "skill" in c.lower() or "required" in c.lower():
                skills_col = c
                break
        if skills_col is None:
            skills_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    return title_col, skills_col


def load_jobs_dataset(csv_path=None):
    """
    Load job dataset from CSV and precompute job embeddings.
    Expects at least 'Job Title' and 'Required Skills' (or equivalent) columns.
    Required Skills can be comma-separated or a string representation of a list.
    Each job's embedding is the mean of its skill embeddings (one vector per job).
    """
    global _jobs_df, _job_embeddings, _all_skills_vocab
    path = Path(csv_path) if csv_path else DEFAULT_JOBS_PATH
    if not path.exists():
        raise FileNotFoundError(f"Jobs CSV not found: {path}")

    df = pd.read_csv(path)
    title_col, skills_col = _detect_csv_columns(df)
    df = df.rename(columns={title_col: "_title", skills_col: "_skills"})
    df = df[["_title", "_skills"]].dropna(how="all")

    # Normalize skills per row into list of strings
    df["_skills_list"] = df["_skills"].apply(_normalize_skills)
    # Drop rows with no skills
    df = df[df["_skills_list"].map(len) > 0].reset_index(drop=True)

    # Build global vocabulary of all skills (for CV keyword matching)
    _all_skills_vocab = []
    for slist in df["_skills_list"]:
        _all_skills_vocab.extend(slist)
    _all_skills_vocab = sorted(set(_all_skills_vocab))

    model = _get_model()
    job_embeddings_list = []

    for idx, row in df.iterrows():
        skills = row["_skills_list"]
        # Encode each skill phrase; model.encode returns (n_skills, 384)
        emb = model.encode(skills, convert_to_numpy=True)
        # Average over skills to get one vector per job
        job_vec = np.mean(emb, axis=0, keepdims=True)
        job_embeddings_list.append(job_vec)

    _job_embeddings = np.vstack(job_embeddings_list)
    _jobs_df = df
    return df


def get_jobs_df():
    """Return the loaded jobs dataframe (after load_jobs_dataset)."""
    if _jobs_df is None:
        load_jobs_dataset()
    return _jobs_df


def get_all_skills_vocabulary():
    """Return the set of all skills from the job dataset (for CV keyword extraction)."""
    if _all_skills_vocab is None:
        load_jobs_dataset()
    return _all_skills_vocab


# ---------------------------------------------------------------------------
# User-facing API
# ---------------------------------------------------------------------------

def extract_skills_from_text(text, skills_list=None):
    """
    Extract skills from raw text using keyword matching.
    - text: CV or profile text (string).
    - skills_list: Optional list of known skills to look for. If None, uses the
      full vocabulary from the loaded job dataset.
    Returns: list of skills (lowercased) that appear in the text.
    """
    if not text or not str(text).strip():
        return []
    if skills_list is None:
        skills_list = get_all_skills_vocabulary()
    text_lower = text.lower()
    found = []
    for skill in skills_list:
        # Word-boundary aware: avoid "python" matching inside "cpython"
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            found.append(skill)
    return list(dict.fromkeys(found))


def process_cv(file_path):
    """
    Extract text from a PDF or Word CV, then detect skills via keyword matching.
    - file_path: path to .pdf or .docx file.
    Returns: dict with 'text' (extracted raw text) and 'skills' (list of detected skills).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"CV file not found: {path}")

    suffix = path.suffix.lower()
    text = ""

    if suffix == ".pdf":
        if pdf_extract_text is None:
            raise ImportError("Install pdfminer.six: pip install pdfminer.six")
        text = pdf_extract_text(str(path)) or ""
    elif suffix in (".docx", ".doc"):
        if DocxDocument is None:
            raise ImportError("Install python-docx: pip install python-docx")
        doc = DocxDocument(str(path))
        text = "\n".join(p.text for p in doc.paragraphs)
    else:
        raise ValueError(f"Unsupported CV format: {suffix}. Use .pdf or .docx")

    skills = extract_skills_from_text(text)
    return {"text": text.strip(), "skills": skills}


def recommend_jobs(user_skills, top_k=5, jobs_csv_path=None):
    """
    Recommend top jobs by similarity and list missing skills for each.
    - user_skills: list of skill strings (e.g. from manual input or process_cv).
    - top_k: number of top jobs to return.
    - jobs_csv_path: optional path to jobs CSV (default: jobs.csv in module dir).
    Returns: list of dicts, each with job_title, similarity_score, required_skills,
             user_skills_matched, missing_skills.
    """
    if _jobs_df is None or _job_embeddings is None:
        load_jobs_dataset(jobs_csv_path)

    user_skills = _normalize_skills(user_skills)
    if not user_skills:
        return []

    model = _get_model()
    user_vec = model.encode(user_skills, convert_to_numpy=True)
    user_embedding = np.mean(user_vec, axis=0, keepdims=True)

    # Cosine similarity: (1, n_jobs)
    sims = cosine_similarity(user_embedding, _job_embeddings)[0]
    top_indices = np.argsort(sims)[::-1][:top_k]

    results = []
    for idx in top_indices:
        row = _jobs_df.iloc[idx]
        job_title = row["_title"]
        required = list(row["_skills_list"])
        user_set = set(user_skills)
        required_set = set(required)
        missing = list(required_set - user_set)
        matched = list(required_set & user_set)
        results.append({
            "job_title": job_title,
            "similarity_score": float(sims[idx]),
            "required_skills": required,
            "user_skills_matched": matched,
            "missing_skills": missing,
        })
    return results


def build_skill_gap_roadmap(user_skills, top_n_jobs=5, jobs_csv_path=None):
    """
    Build a skill-gap roadmap from the top N recommended jobs.
    Aggregates missing skills across those jobs and returns a prioritized view.
    """
    recs = recommend_jobs(user_skills, top_k=top_n_jobs, jobs_csv_path=jobs_csv_path)
    # Count how many of the top jobs require each missing skill
    skill_counts = {}
    for r in recs:
        for s in r["missing_skills"]:
            skill_counts[s] = skill_counts.get(s, 0) + 1
    # Sort by frequency (most often missing across top jobs first)
    roadmap = sorted(skill_counts.items(), key=lambda x: -x[1])
    return {
        "top_jobs": recs,
        "skill_gap_roadmap": [{"skill": s, "count_in_top_jobs": c} for s, c in roadmap],
    }


# ---------------------------------------------------------------------------
# Main: test the module independently
# ---------------------------------------------------------------------------

def main():
    """Test the module with example skills and optional sample CV."""
    print("SkillMap AI – Module test\n" + "=" * 40)

    # Ensure jobs dataset exists
    jobs_path = DEFAULT_JOBS_PATH
    if not jobs_path.exists():
        print(f"Warning: {jobs_path} not found. Creating a minimal sample jobs.csv for demo.")
        sample_df = pd.DataFrame({
            "Job Title": [
                "Python Developer",
                "Data Scientist",
                "ML Engineer",
                "Backend Engineer",
                "Full Stack Developer",
            ],
            "Required Skills": [
                "python, sql, git, rest api",
                "python, sql, machine learning, statistics",
                "python, machine learning, tensorflow, pytorch",
                "python, sql, docker, kubernetes, rest api",
                "python, javascript, react, sql, rest api",
            ],
        })
        sample_df.to_csv(jobs_path, index=False)
        print(f"Created {jobs_path}\n")

    # Load dataset and precompute embeddings
    try:
        load_jobs_dataset()
        print("Jobs dataset loaded and embeddings precomputed.")
    except Exception as e:
        print(f"Failed to load jobs: {e}")
        return

    # Test 1: Manual skills
    example_skills = ["python", "sql", "git"]
    print(f"\n1) Input skills: {example_skills}")
    recs = recommend_jobs(example_skills, top_k=3)
    print("   Top 3 recommendations:")
    for r in recs:
        print(f"   - {r['job_title']} (score: {r['similarity_score']:.3f})")
        print(f"     Missing: {r['missing_skills']}")

    # Test 2: Skill-gap roadmap
    print("\n2) Skill-gap roadmap (top 5 jobs):")
    roadmap = build_skill_gap_roadmap(example_skills, top_n_jobs=5)
    for item in roadmap["skill_gap_roadmap"][:8]:
        print(f"   - {item['skill']} (missing in {item['count_in_top_jobs']} of top jobs)")

    # Test 3: Extract skills from text (simulated CV snippet)
    sample_text = "I have experience with Python, SQL and REST APIs. Used Git for version control."
    extracted = extract_skills_from_text(sample_text)
    print(f"\n3) Extracted skills from sample text: {extracted}")

    # Test 4: Optional CV file (if user provides one)
    sample_cv = BASE_DIR / "sample_cv.pdf"
    if not sample_cv.exists():
        sample_cv = BASE_DIR / "sample_cv.docx"
    if sample_cv.exists():
        try:
            cv_result = process_cv(sample_cv)
            print(f"\n4) CV processed: {len(cv_result['skills'])} skills found.")
            print(f"   Skills: {cv_result['skills']}")
        except Exception as e:
            print(f"\n4) CV processing skipped: {e}")
    else:
        print("\n4) No sample_cv.pdf or sample_cv.docx found; skipping CV test.")

    print("\n" + "=" * 40 + "\nDone.")


if __name__ == "__main__":
    main()
