import streamlit as st
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Resume Skill Analyzer", page_icon="🤖")

SKILLS = [
    "python", "java", "c++", "javascript", "html", "css", "react",
    "node.js", "sql", "mysql", "mongodb", "machine learning", "deep learning",
    "artificial intelligence", "data science", "pandas", "numpy", "tensorflow",
    "pytorch", "git", "github", "docker", "aws", "flask", "django",
    "communication", "leadership", "problem solving"
]

def clean(text):
    return re.sub(r"\s+", " ", text.lower()).strip()

def find_skills(text):
    text = clean(text)
    found = []
    for skill in SKILLS:
        if skill in text:
            found.append(skill)
    return sorted(set(found))

def similarity(resume, job):
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform([clean(resume), clean(job)])
    score = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
    return round(score * 100, 2)

st.title("🤖 AI Resume Skill Analyzer")
st.write("Compare a resume with a job description using TF-IDF and cosine similarity.")

resume = st.text_area(
    "Paste Resume Text",
    height=220,
    placeholder="Example: Python, SQL, pandas, machine learning, Git..."
)

job = st.text_area(
    "Paste Job Description",
    height=220,
    placeholder="Example: Looking for a Python developer with SQL, Git and machine learning..."
)

if st.button("Analyze Resume"):
    if not resume.strip() or not job.strip():
        st.warning("Please enter both resume text and job description.")
    else:
        score = similarity(resume, job)
        resume_skills = set(find_skills(resume))
        job_skills = set(find_skills(job))
        matched = sorted(resume_skills & job_skills)
        missing = sorted(job_skills - resume_skills)

        st.subheader("AI Match Score")
        st.progress(min(int(score), 100))
        st.metric("Resume–Job Similarity", f"{score}%")

        st.subheader("Matched Skills")
        st.write(", ".join(matched) if matched else "No matching skills detected.")

        st.subheader("Suggested Skills to Improve")
        st.write(", ".join(missing) if missing else "No major listed skill gaps detected.")

        st.info("This is an educational prototype. The score is based on text similarity, not a hiring decision.")
