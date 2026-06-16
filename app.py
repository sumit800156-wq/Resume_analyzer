import streamlit as st
import PyPDF2
import re
import nltk
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# -------------------------
# NLTK DOWNLOAD
# -------------------------

nltk.download("punkt")
nltk.download("stopwords")

# -------------------------
# PAGE CONFIG
# -------------------------

st.set_page_config(
    page_title="AI Resume Analyzer Pro",
    page_icon="🚀",
    layout="wide"
)

# -------------------------
# DARK THEME
# -------------------------

st.markdown("""
<style>

.stApp{
background-color:#0E1117;
color:white;
}

[data-testid="stSidebar"]{
background-color:#161B22;
}

.metric-card{
padding:15px;
border-radius:12px;
background:#1C2128;
}

.skill-box{
background:#198754;
padding:8px;
margin:5px;
border-radius:10px;
color:white;
}

.missing-box{
background:#dc3545;
padding:8px;
margin:5px;
border-radius:10px;
color:white;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# HEADER
# -------------------------

st.title("🚀 AI Resume Analyzer Pro")

st.markdown("""
### Smart ATS Resume Screening Dashboard

Analyze resumes using:

✅ ATS Score

✅ Skill Matching

✅ Missing Keywords

✅ Resume Sections

✅ Word Cloud

✅ Skill Analytics Dashboard
""")

# -------------------------
# SIDEBAR
# -------------------------

with st.sidebar:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/3135/3135715.png",
        width=120
    )

    st.title("Resume Analyzer")

    st.markdown("---")

    st.markdown("""
### Features

📊 ATS Score

🎯 Skill Match

📈 Analytics Dashboard

☁ Word Cloud

📥 Download Report

🤖 AI Suggestions
""")

    st.markdown("---")

    st.info(
        "Upload Resume PDF and compare it with Job Description."
    )
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
# TEXT CLEANING
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
# SKILL DATABASE
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
        "nodejs",
        "django",
        "flask",
        "streamlit",
        "machine learning",
        "deep learning",
        "tensorflow",
        "pytorch",
        "pandas",
        "numpy",
        "opencv",
        "data science",
        "power bi",
        "excel",
        "aws",
        "azure",
        "docker",
        "git",
        "github",
        "linux",
        "api",
        "rest api"
    ]


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

    job_processed = clean_text(
        job_description
    )

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(
        [
            resume_processed,
            job_processed
        ]
    )

    similarity = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:2]
    )[0][0]

    return round(
        similarity * 100,
        2
    )


# -------------------------
# SKILL MATCHING
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

    if total_skills > 0:

        skill_score = (
            len(matched)
            /
            total_skills
        ) * 100

    else:

        skill_score = 0

    ats_score = (
        similarity * 0.6
        +
        skill_score * 0.4
    )

    return (
        round(ats_score, 2),
        round(skill_score, 2)
    )


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
        jd_words
        -
        resume_words
    )

    return missing[:25]


# -------------------------
# TOP KEYWORDS
# -------------------------

def extract_keywords(text):

    text = clean_text(text)

    words = word_tokenize(text)

    freq = Counter(words)

    return freq.most_common(15)


# -------------------------
# RESUME SECTION CHECK
# -------------------------

def check_resume_sections(
    resume_text
):

    resume_text = resume_text.lower()

    sections = {

        "Education":
        "education" in resume_text,

        "Skills":
        "skills" in resume_text,

        "Projects":
        "project" in resume_text,

        "Experience":
        "experience" in resume_text,

        "Certifications":
        "certification" in resume_text,

        "Internship":
        "internship" in resume_text

    }

    return sections
# -------------------------
# FILE UPLOAD
# -------------------------

uploaded_file = st.file_uploader(
    "📄 Upload Resume PDF",
    type=["pdf"]
)

job_description = st.text_area(
    "📝 Paste Job Description",
    height=250
)

# -------------------------
# ANALYZE BUTTON
# -------------------------

if st.button("🚀 Analyze Resume"):

    if uploaded_file is None:

        st.warning(
            "Please upload resume PDF."
        )

    elif not job_description.strip():

        st.warning(
            "Please paste Job Description."
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

        ats_score, skill_score = calculate_ats_score(
            similarity,
            matched_skills,
            missing_skills
        )

        missing_keywords = get_missing_keywords(
            resume_text,
            job_description
        )

        sections = check_resume_sections(
            resume_text
        )

        # -------------------------
        # SCORE DASHBOARD
        # -------------------------

        st.markdown("---")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "🎯 ATS Score",
                f"{ats_score}%"
            )

        with c2:
            st.metric(
                "📄 Similarity",
                f"{similarity}%"
            )

        with c3:
            st.metric(
                "🛠 Skill Match",
                f"{skill_score}%"
            )

        st.progress(
            ats_score / 100
        )

        # -------------------------
        # PIE CHART
        # -------------------------

        st.subheader(
            "📊 Skill Analysis"
        )

        fig = px.pie(
            names=["Matched", "Missing"],
            values=[
                len(matched_skills),
                len(missing_skills)
            ],
            title="Skill Match Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # -------------------------
        # MATCHED SKILLS
        # -------------------------

        st.subheader(
            "✅ Matched Skills"
        )

        if matched_skills:

            for skill in matched_skills:

                st.success(skill)

        else:

            st.warning(
                "No matched skills found."
            )

        # -------------------------
        # MISSING SKILLS
        # -------------------------

        st.subheader(
            "❌ Missing Skills"
        )

        if missing_skills:

            for skill in missing_skills:

                st.error(skill)

        else:

            st.success(
                "No missing skills."
            )

        # -------------------------
        # MISSING KEYWORDS
        # -------------------------

        st.subheader(
            "🔍 Missing Keywords"
        )

        for keyword in missing_keywords:

            st.write(
                f"• {keyword}"
            )

        # -------------------------
        # TOP KEYWORDS
        # -------------------------

        st.subheader(
            "📈 Top Resume Keywords"
        )

        keywords = extract_keywords(
            resume_text
        )

        df = pd.DataFrame(
            keywords,
            columns=[
                "Keyword",
                "Count"
            ]
        )

        st.dataframe(
            df,
            use_container_width=True
        )

        # -------------------------
        # BAR CHART
        # -------------------------

        if len(df) > 0:

            fig2 = px.bar(
                df,
                x="Keyword",
                y="Count",
                title="Keyword Frequency"
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )

        # -------------------------
        # WORD CLOUD
        # -------------------------

        st.subheader(
            "☁ Resume Word Cloud"
        )

        cloud = WordCloud(
            width=1000,
            height=500,
            background_color="white"
        ).generate(
            resume_text
        )

        fig3, ax = plt.subplots(
            figsize=(10, 5)
        )

        ax.imshow(
            cloud,
            interpolation="bilinear"
        )

        ax.axis("off")

        st.pyplot(fig3)

        # -------------------------
        # SECTION CHECK
        # -------------------------

        st.subheader(
            "📋 Resume Sections"
        )

        for section, present in sections.items():

            if present:

                st.success(
                    f"✅ {section}"
                )

            else:

                st.error(
                    f"❌ {section}"
                )

        # -------------------------
        # ATS FEEDBACK
        # -------------------------

        st.subheader(
            "🤖 ATS Suggestions"
        )

        if ats_score < 60:

            st.error(
                "Resume needs major improvements."
            )

        elif ats_score < 80:

            st.warning(
                "Good resume. Add more JD keywords and skills."
            )

        else:

            st.success(
                "Excellent ATS Match!"
            )

        if missing_skills:

            st.info(
                "Add missing skills from Job Description."
            )

        if len(missing_keywords) > 0:

            st.info(
                "Include important missing keywords."
            )
