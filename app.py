import streamlit as st
import PyPDF2
import re
import nltk

from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK Resources
nltk.download("punkt")
nltk.download("punkt-tab")
nltk.download("stopwords")

# -------------------------
# PAGE CONFIG
# -------------------------

st.set_page_config(
    page_title="AI Resume Analyzer Pro",
    layout="wide"
)

st.title("🚀 AI Resume Analyzer Pro")

st.markdown("""
Upload your Resume PDF and compare it with a Job Description.

### Features
✅ ATS Score
✅ Skill Match Analysis
✅ Missing Skills
✅ Missing Keywords
✅ Resume Section Detection
✅ Contact Information Extraction
✅ ATS Suggestions
""")

# -------------------------
# SKILLS DATABASE
# -------------------------

def get_skills():

    return [
        "python",
        "java",
        "sql",
        "mysql",
        "postgresql",
        "html",
        "css",
        "javascript",
        "react",
        "django",
        "flask",
        "streamlit",
        "numpy",
        "pandas",
        "tensorflow",
        "machine learning",
        "data science",
        "git",
        "github",
        "docker",
        "aws",
        "api",
        "fastapi",
        "tableau",
        "powerbi",
        "excel"
    ]

# -------------------------
# PDF TEXT EXTRACTION
# -------------------------

def extract_text_from_pdf(uploaded_file):

    try:

        pdf_reader = PyPDF2.PdfReader(uploaded_file)

        text = ""

        for page in pdf_reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text

        return text

    except Exception as e:

        st.error(f"PDF Error: {e}")
        return ""

# -------------------------
# CONTACT INFO
# -------------------------

def extract_contact_info(text):

    email = re.findall(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        text
    )

    phone = re.findall(
        r"\+?\d[\d\s\-]{8,15}",
        text
    )

    return email, phone

# -------------------------
# CLEAN TEXT
# -------------------------

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    words = word_tokenize(text)

    stop_words = set(
        stopwords.words("english")
    )

    words = [
        word
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# -------------------------
# SIMILARITY
# -------------------------

def calculate_similarity(
    resume_text,
    job_description
):

    resume_processed = clean_text(
        resume_text
    )

    jd_processed = clean_text(
        job_description
    )

    vectorizer = TfidfVectorizer()

    tfidf = vectorizer.fit_transform(
        [
            resume_processed,
            jd_processed
        ]
    )

    similarity = cosine_similarity(
        tfidf[0:1],
        tfidf[1:2]
    )[0][0]

    return round(
        similarity * 100,
        2
    )

# -------------------------
# SKILL MATCH
# -------------------------

def get_skill_match(
    resume_text,
    job_description
):

    skills = get_skills()

    resume_text = resume_text.lower()
    job_description = job_description.lower()

    resume_skills = [
        skill
        for skill in skills
        if skill in resume_text
    ]

    jd_skills = [
        skill
        for skill in skills
        if skill in job_description
    ]

    matched = list(
        set(resume_skills)
        &
        set(jd_skills)
    )

    missing = list(
        set(jd_skills)
        -
        set(resume_skills)
    )

    return matched, missing

# -------------------------
# ATS SCORE
# -------------------------

def calculate_ats_score(
    similarity,
    matched,
    missing
):

    total_skills = (
        len(matched)
        +
        len(missing)
    )

    if total_skills == 0:
        skill_score = 0

    else:
        skill_score = (
            len(matched)
            /
            total_skills
        ) * 100

    final_score = (
        similarity * 0.6
        +
        skill_score * 0.4
    )

    return round(final_score, 2)

# -------------------------
# MISSING KEYWORDS
# -------------------------

def get_missing_keywords(
    resume_text,
    job_description
):

    resume_words = set(
        clean_text(
            resume_text
        ).split()
    )

    jd_words = set(
        clean_text(
            job_description
        ).split()
    )

    missing = list(
        jd_words - resume_words
    )

    return missing[:20]

# -------------------------
# RESUME SECTIONS
# -------------------------

def check_resume_sections(
    resume_text
):

    text = resume_text.lower()

    sections = {

        "Education":
        "education" in text,

        "Skills":
        "skills" in text,

        "Projects":
        "project" in text,

        "Experience":
        "experience" in text,

        "Certifications":
        "certification" in text

    }

    return sections

# -------------------------
# KEYWORDS
# -------------------------

def extract_keywords(text):

    text = clean_text(text)

    words = word_tokenize(text)

    freq = Counter(words)

    return freq.most_common(15)

# -------------------------
# UI
# -------------------------

uploaded_file = st.file_uploader(
    "Upload Resume PDF",
    type=["pdf"]
)

job_description = st.text_area(
    "Paste Job Description"
)

if st.button("Analyze Resume"):

    if uploaded_file is None:

        st.warning(
            "Please upload resume."
        )

    elif not job_description:

        st.warning(
            "Please enter Job Description."
        )

    else:

        resume_text = extract_text_from_pdf(
            uploaded_file
        )

        similarity = calculate_similarity(
            resume_text,
            job_description
        )

        matched_skills, missing_skills = get_skill_match(
            resume_text,
            job_description
        )

        ats_score = calculate_ats_score(
            similarity,
            matched_skills,
            missing_skills
        )

        missing_keywords = get_missing_keywords(
            resume_text,
            job_description
        )

        resume_sections = check_resume_sections(
            resume_text
        )

        emails, phones = extract_contact_info(
            resume_text
        )

        st.subheader("📊 ATS Score")

        st.progress(
            ats_score / 100
        )

        st.metric(
            "ATS Score",
            f"{ats_score}%"
        )

        st.write(
            f"Keyword Similarity: {similarity}%"
        )

        st.subheader("📧 Contact Information")

        st.write(
            "Emails:",
            emails
        )

        st.write(
            "Phones:",
            phones
        )

        st.subheader("✅ Matched Skills")

        for skill in matched_skills:
            st.success(skill)

        st.subheader("❌ Missing Skills")

        for skill in missing_skills:
            st.error(skill)

        st.subheader("🔑 Missing Keywords")

        for word in missing_keywords:
            st.write(word)

        st.subheader("📂 Resume Sections")

        for section, present in resume_sections.items():

            if present:
                st.success(
                    f"✅ {section}"
                )

            else:
                st.error(
                    f"❌ {section}"
                )

        st.subheader("📈 Top Resume Keywords")

        for word, count in extract_keywords(
            resume_text
        ):
            st.write(
                f"{word} ({count})"
            )

        st.subheader("💡 ATS Suggestions")

        if missing_skills:

            st.info(
                "Add missing skills from Job Description."
            )

        if ats_score < 60:

            st.warning(
                "Resume needs more relevant keywords."
            )

        elif ats_score < 80:

            st.info(
                "Good match. Add more keywords."
            )

        else:

            st.success(
                "Excellent ATS Match!"
            )
