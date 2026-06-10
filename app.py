# ==========================================================
# Resume ATS Score Checker
# ----------------------------------------------------------
# Install Required Packages:
# pip install streamlit pdfplumber scikit-learn pandas numpy
#
# Run:
# streamlit run app.py
# ==========================================================

import streamlit as st
import pdfplumber
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------
# Page Config
# ----------------------------------------------------------
st.set_page_config(
    page_title="Resume ATS Score Checker",
    page_icon="📄",
    layout="wide"
)

# ----------------------------------------------------------
# Custom CSS (disabled for debugging — re-enable if needed)
# ----------------------------------------------------------
st.markdown("<!-- custom CSS disabled for debugging -->", unsafe_allow_html=True)

# ----------------------------------------------------------
# Functions
# ----------------------------------------------------------
def extract_text(pdf_file):
    """
    Extract text from uploaded PDF file.
    Returns extracted text string.
    """
    try:
        text = ""

        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()

                if page_text:
                    text += page_text + "\n"

        return text.strip()

    except Exception:
        return None


def calculate_score(resume_text, jd_text):
    """
    Calculate ATS Score using TF-IDF + Cosine Similarity.
    Returns percentage score.
    """
    documents = [resume_text, jd_text]

    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(documents)

    similarity = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]

    return round(similarity * 100, 2)


def get_missing_keywords(resume_text, jd_text):
    """
    Extract important JD keywords missing from resume.
    """

    common_skills = [
        "python",
        "sql",
        "excel",
        "aws",
        "react",
        "docker",
        "git",
        "java",
        "machine learning",
        "ml",
        "tensorflow",
        "pytorch",
        "power bi",
        "tableau",
        "mongodb",
        "mysql",
        "postgresql",
        "numpy",
        "pandas",
        "scikit-learn",
        "deep learning",
        "nlp",
        "data analysis",
        "data science",
        "javascript",
        "html",
        "css",
        "nodejs",
        "django",
        "flask",
        "streamlit",
        "linux",
        "kubernetes",
        "azure",
        "gcp"
    ]

    resume_text = resume_text.lower()
    jd_text = jd_text.lower()

    missing = []

    for skill in common_skills:
        if skill in jd_text and skill not in resume_text:
            missing.append(skill)

    return sorted(set(missing))


# ----------------------------------------------------------
# Header
# ----------------------------------------------------------
st.title("🚀 Resume ATS Score Checker")
st.markdown(
    "Upload your **Resume PDF** and **Job Description PDF** to check ATS compatibility."
)

st.divider()

# ----------------------------------------------------------
# Layout
# ----------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    resume_pdf = st.file_uploader(
        "📄 Upload Resume PDF",
        type=["pdf"]
    )

with col2:
    jd_pdf = st.file_uploader(
        "📋 Upload Job Description PDF",
        type=["pdf"]
    )

# ----------------------------------------------------------
# Analyze Button
# ----------------------------------------------------------
if st.button("🔍 Analyze ATS Score", use_container_width=True):

    if resume_pdf is None or jd_pdf is None:
        st.error("Please upload both Resume and Job Description PDFs.")
        st.stop()

    with st.spinner("Analyzing Resume... Please wait ⏳"):

        resume_text = extract_text(resume_pdf)
        jd_text = extract_text(jd_pdf)

        # Scanned PDF Handling
        if not resume_text:
            st.error("❌ Scanned PDF not supported (Resume).")
            st.stop()

        if not jd_text:
            st.error("❌ Scanned PDF not supported (Job Description).")
            st.stop()

        # ATS Score
        ats_score = calculate_score(resume_text, jd_text)

        st.subheader("📊 ATS Match Score")

        st.progress(min(int(ats_score), 100))

        if ats_score >= 80:
            st.success(f"Excellent Match: {ats_score}%")
        elif ats_score >= 60:
            st.success(f"Good Match: {ats_score}%")
        else:
            st.warning(f"Low Match: {ats_score}%")

        st.divider()

        # Missing Keywords
        missing_keywords = get_missing_keywords(
            resume_text,
            jd_text
        )

        st.subheader("⚠️ Missing Keywords")

        if missing_keywords:
            st.warning(", ".join(missing_keywords))
        else:
            st.success("No important keywords missing!")

        st.divider()

        # Resume Improvement Tips
        st.subheader("💡 ATS Optimization Tips")

        st.info("""
        ✔ Add missing keywords naturally into your resume

        ✔ Match job title wording

        ✔ Include relevant technical skills

        ✔ Add measurable achievements

        ✔ Keep formatting ATS-friendly

        ✔ Avoid images, tables, and graphics
        """)

# ----------------------------------------------------------
# Sidebar
# ----------------------------------------------------------
with st.sidebar:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/3135/3135715.png",
        width=120
    )

    st.title("👨‍💻 Developer")

    st.markdown("""
    **Sachin Kumar**
    
    AI/ML Engineer | Full Stack Developer
    
    Build AI, Data Science and Web Applications
    """)

    st.markdown("---")

    st.markdown(
        """
        🔗 **LinkedIn**
        
        https://www.linkedin.com/in/sachin-kumar-sde-ai-ml
        
        🔗 **GitHub**
        
        https://github.com/Sachin-SDE12
        """
    )

    st.markdown("---")

    st.success("ATS Checker v1.0")

# ----------------------------------------------------------
# Footer
# ----------------------------------------------------------
st.markdown(
    """
    <div class='footer'>
        Made with ❤️ using Python, Streamlit, PDFPlumber & Scikit-Learn
    </div>
    """,
    unsafe_allow_html=True
)