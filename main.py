from fastapi import FastAPI, UploadFile, File, Form
import os
import shutil
import pdfplumber
import glob
from sentence_transformers import SentenceTransformer, util
from docx import Document
from sklearn.feature_extraction.text import CountVectorizer
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- Load Embedding Model ----------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- Trusted Resume Dataset ----------------
TRUSTED_RESUME_EMBEDDINGS = []

def load_trusted_resumes(dataset_path="dataset"):
    global TRUSTED_RESUME_EMBEDDINGS
    TRUSTED_RESUME_EMBEDDINGS = []

    for file in os.listdir(dataset_path):
        file_path = os.path.join(dataset_path, file)

        if file.endswith(".pdf") or file.endswith(".docx"):
            try:
                print(f"Processing: {file_path}")
                text = extract_resume_text(file_path)
                embedding = embedding_model.encode([text])[0]
                TRUSTED_RESUME_EMBEDDINGS.append(embedding)
            except Exception as e:
                print(f"Error processing {file}: {e}")

    print(f"Loaded {len(TRUSTED_RESUME_EMBEDDINGS)} trusted resumes.")


# Load semantic model
semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

app = FastAPI()

UPLOAD_DIR = "uploads"
DATASET_DIR = "../dataset/trusted_resumes"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ---------------- Resume Parsing ----------------

def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text


def extract_text_from_docx(file_path: str) -> str:
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])


def extract_resume_text(file_path: str) -> str:
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        return extract_text_from_docx(file_path)
    else:
        return ""


# ---------------- Keyword Scoring ----------------
# ---------------- Skill Gap Suggestions ----------------

def generate_skill_feedback(keyword_results):
    matched = keyword_results["matched_skills"]
    missing = keyword_results["missing_skills"]
    score = keyword_results["keyword_score"]

    if score >= 80:
        message = "Your resume matches most of the required technical skills."
    elif score >= 60:
        message = "Your resume has a decent skill match, but some important skills are missing."
    else:
        message = "Your resume is missing several key skills required for this role."

    suggestions = []
    for skill in missing[:5]:
        suggestions.append(f"Consider learning or adding projects using {skill}.")

    return {
        "skill_feedback": message,
        "skill_suggestions": suggestions
    }


# ---------------- Skill Dictionary ----------------

SWE_SKILLS = {
    "python", "java", "c", "cpp", "c++", "javascript", "typescript",
    "react", "angular", "vue", "node", "nodejs", "express",
    "django", "flask", "spring", "springboot",
    "sql", "mysql", "postgresql", "mongodb",
    "redis", "elasticsearch",
    "html", "css", "sass",
    "git", "github", "gitlab",
    "docker", "kubernetes",
    "aws", "gcp", "azure",
    "linux", "bash",
    "rest", "api", "apis",
    "microservices",
    "system design",
    "data structures", "algorithms",
    "oop", "object oriented",
    "machine learning", "deep learning",
    "tensorflow", "pytorch",
    "ci", "cd", "ci/cd",
    "testing", "unit testing",
    "agile", "scrum"
}


# ---------------- Keyword Scoring ----------------

def extract_skills(text: str):
    text = text.lower()
    found_skills = set()

    for skill in SWE_SKILLS:
        if skill in text:
            found_skills.add(skill)

    return found_skills


def keyword_match_score(resume_text: str, jd_text: str):
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)

    matched = list(resume_skills.intersection(jd_skills))
    missing = list(jd_skills - resume_skills)

    if len(jd_skills) == 0:
        score = 0
    else:
        score = int(len(matched) / len(jd_skills) * 100)

    return {
        "keyword_score": score,
        "matched_skills": sorted(matched),
        "missing_skills": sorted(missing)
    }
# ---------------- Semantic Scoring ----------------

def semantic_similarity_score(resume_text: str, jd_text: str):
    resume_embedding = semantic_model.encode(resume_text, convert_to_tensor=True)
    jd_embedding = semantic_model.encode(jd_text, convert_to_tensor=True)

    similarity = util.cos_sim(resume_embedding, jd_embedding).item()

    # Convert to percentage
    semantic_score = int(similarity * 100)

    return {
        "semantic_score": semantic_score
    }
# ---------------- Profile Alignment ----------------

def load_trusted_resume_texts():
    texts = []

    for file_path in glob.glob(os.path.join(DATASET_DIR, "*")):
        if file_path.endswith(".pdf") or file_path.endswith(".docx"):
            try:
                text = extract_resume_text(file_path)
                if text.strip():
                    texts.append(text)
            except:
                continue

    return texts


def profile_alignment_score(resume_text: str):
    if not TRUSTED_RESUME_EMBEDDINGS:
        return {"profile_alignment_score": 50}

    resume_embedding = embedding_model.encode([resume_text])[0]

    similarities = cosine_similarity(
        [resume_embedding],
        TRUSTED_RESUME_EMBEDDINGS
    )[0]

    best_match = max(similarities)

    score = int(best_match * 100)

    return {
        "profile_alignment_score": score
    }

# ---------------- Experience Analysis ----------------

def analyze_experience(resume_text: str, jd_text: str):
    resume_lower = resume_text.lower()
    jd_lower = jd_text.lower()

    # Detect resume experience level
    if "intern" in resume_lower:
        resume_level = "Intern/Fresher"
        resume_score = 40
    elif "experience" in resume_lower or "years" in resume_lower:
        resume_level = "Experienced"
        resume_score = 80
    else:
        resume_level = "Fresher"
        resume_score = 30

    # Detect JD expectation
    if "2+" in jd_lower or "3+" in jd_lower or "years of experience" in jd_lower:
        jd_requirement = "Experienced"
    elif "intern" in jd_lower:
        jd_requirement = "Intern"
    else:
        jd_requirement = "Entry-level"

    # Fit logic
    if resume_level == "Experienced" and jd_requirement == "Experienced":
        fit = "Strong"
        feedback = "Your experience level aligns well with this role."
        fit_score = 90
    elif resume_level == "Intern/Fresher" and jd_requirement in ["Intern", "Entry-level"]:
        fit = "Good"
        feedback = "Your experience level is suitable for this role."
        fit_score = 75
    else:
        fit = "Low"
        feedback = "This role may expect more experience than your current profile."
        fit_score = 40

    return {
        "resume_experience_level": resume_level,
        "jd_experience_requirement": jd_requirement,
        "experience_fit": fit,
        "experience_fit_score": fit_score,
        "experience_feedback": feedback
    }

# ---------------- Relevant Text Extraction ----------------

def extract_relevant_text(resume_text: str):
    text = resume_text.lower()

    relevant_lines = []

    keywords = [
        "project", "experience", "intern", "developer",
        "engineer", "built", "developed", "api",
        "backend", "frontend", "system", "database"
    ]

    for line in text.split("\n"):
        if any(kw in line for kw in keywords):
            relevant_lines.append(line)

    return " ".join(relevant_lines)
# ---------------- Resume Quality Analysis ----------------

def analyze_resume_quality(resume_text: str):
    text = resume_text.lower()

    score = 100
    issues = []
    suggestions = []

    # Check for key sections
    if "skills" not in text:
        score -= 15
        issues.append("Skills section not clearly detected.")
        suggestions.append("Add a dedicated skills section with technical skills.")

    if "project" not in text:
        score -= 20
        issues.append("Projects section is missing or unclear.")
        suggestions.append("Add at least 2–3 technical projects.")

    if "experience" not in text and "intern" not in text:
        score -= 20
        issues.append("No experience or internship detected.")
        suggestions.append("Add internship, freelance, or project-based experience.")

    if "education" not in text:
        score -= 10
        issues.append("Education section not clearly detected.")
        suggestions.append("Add an education section with degree and institution.")

    # Length check
    word_count = len(text.split())
    if word_count < 200:
        score -= 15
        issues.append("Resume content is too short.")
        suggestions.append("Add more project details and technical descriptions.")

    # Minimum score cap
    score = max(score, 30)

    return {
        "resume_quality_score": score,
        "detected_issues": issues,
        "quality_suggestions": suggestions
    }
# ---------------- Final Report Generation ----------------

def generate_final_report(keyword_results,
                          semantic_results,
                          alignment_results,
                          skill_feedback,
                          project_analysis,
                          experience_analysis,
                          resume_quality,
                          final_score):

    strengths = []
    weaknesses = []
    roadmap = []

    # Strengths
    if keyword_results["keyword_score"] >= 70:
        strengths.append("Good match with required technical skills.")

    if project_analysis["project_depth_score"] >= 80:
        strengths.append("Projects show strong engineering depth.")

    if experience_analysis["experience_fit"] in ["Good", "Strong"]:
        strengths.append("Experience level aligns with the job role.")

    if resume_quality["resume_quality_score"] >= 85:
        strengths.append("Resume structure is ATS-friendly.")

    # Weaknesses
    missing_skills = keyword_results["missing_skills"]
    if missing_skills:
        weaknesses.append(
            f"Missing some required skills: {', '.join(missing_skills[:3])}."
        )

    if project_analysis["project_depth_score"] < 60:
        weaknesses.append("Projects lack system-level complexity.")

    if experience_analysis["experience_fit"] == "Low":
        weaknesses.append("Experience level may not match job expectations.")

    if resume_quality["resume_quality_score"] < 70:
        weaknesses.append("Resume structure or content needs improvement.")

    # Roadmap
    for suggestion in skill_feedback["skill_suggestions"][:3]:
        roadmap.append(suggestion)

    for suggestion in project_analysis["project_suggestions"][:2]:
        roadmap.append(suggestion)

    # Overall assessment
    if final_score >= 80:
        overall = "Strong profile with high alignment for this role."
    elif final_score >= 60:
        overall = "Moderate profile. With a few improvements, this resume can be highly competitive."
    else:
        overall = "Profile needs significant improvement to match this role."
    return {
        "overall_assessment": overall,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "improvement_roadmap": roadmap
    }


# ---------------- API Endpoints ----------------
# ---------------- Project Depth Analysis ----------------

ADVANCED_PROJECT_KEYWORDS = {
    "microservices", "docker", "kubernetes", "aws", "gcp", "azure",
    "redis", "caching", "ci/cd", "load balancer", "system design"
}

INTERMEDIATE_PROJECT_KEYWORDS = {
    "rest", "api", "authentication", "jwt", "database",
    "react", "node", "flask", "django"
}


def analyze_project_depth(resume_text: str):
    text = resume_text.lower()

    advanced_hits = sum(1 for kw in ADVANCED_PROJECT_KEYWORDS if kw in text)
    intermediate_hits = sum(1 for kw in INTERMEDIATE_PROJECT_KEYWORDS if kw in text)

    if advanced_hits >= 2:
        score = 90
        level = "Advanced"
        feedback = "Your projects demonstrate strong system-level or scalable engineering concepts."
    elif intermediate_hits >= 2:
        score = 65
        level = "Intermediate"
        feedback = "Your projects show practical engineering skills but could be improved with scalable or cloud-based features."
    else:
        score = 40
        level = "Basic"
        feedback = "Your projects appear to be basic or CRUD-style. Consider building more complex, scalable systems."

    suggestions = [
        "Build a cloud-deployed project using AWS or GCP.",
        "Add caching or message queues to an existing project.",
        "Implement CI/CD or containerization using Docker."
    ]

    return {
        "project_depth_score": score,
        "project_level": level,
        "project_feedback": feedback,
        "project_suggestions": suggestions
    }
@app.get("/")
def root():
    return {"message": "Resume Intelligence API is running"}


@app.post("/analyze/")
def analyze_resume(
    file: UploadFile = File(...),
    jd_text: str = Form(...)
):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    resume_text = extract_resume_text(file_path)

# Extract only relevant technical sections
    relevant_text = extract_relevant_text(resume_text)

    keyword_results = keyword_match_score(resume_text, jd_text)
    semantic_results = semantic_similarity_score(relevant_text, jd_text)
    alignment_results = profile_alignment_score(resume_text)

# Calculate final ATS score first
    final_score = int(
        0.3 * keyword_results["keyword_score"] +
        0.3 * semantic_results["semantic_score"] +
        0.4 * alignment_results["profile_alignment_score"]
)

    skill_feedback = generate_skill_feedback(keyword_results)
    project_analysis = analyze_project_depth(resume_text)
    experience_analysis = analyze_experience(resume_text, jd_text)
    resume_quality = analyze_resume_quality(resume_text)

# Now generate final report
    final_report = generate_final_report(
        keyword_results,
        semantic_results,
        alignment_results,
        skill_feedback,
        project_analysis,
        experience_analysis,
        resume_quality,
        final_score
)
    final_score = int(
        0.3 * keyword_results["keyword_score"] +
        0.3 * semantic_results["semantic_score"] +
        0.4 * alignment_results["profile_alignment_score"]
)

    return {
        "filename": file.filename,
        "keyword_analysis": keyword_results,
        "semantic_analysis": semantic_results,
        "profile_alignment": alignment_results,
        "final_ats_score": final_score,
        "skill_feedback": skill_feedback,
        "project_analysis": project_analysis,
        "experience_analysis": experience_analysis,
        "resume_quality": resume_quality,
        "final_report": final_report
}
# Load trusted resumes after all functions are defined
load_trusted_resumes("dataset/trusted_resumes")









