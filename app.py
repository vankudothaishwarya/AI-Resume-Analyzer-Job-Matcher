import streamlit as st
import re
import pytesseract
import fitz

from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from pypdf import PdfReader
from docx import Document
from langchain_ollama import OllamaLLM


# =========================================================
# TESSERACT OCR PATH
# =========================================================
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

# CSS
st.markdown("""
<style>
textarea {
    color: black !important;
    background-color: white !important;
}

textarea::placeholder {
    color: #555555 !important;
}
</style>
""", unsafe_allow_html=True)

st.title("📄 AI Resume Analyzer & Job Matcher")

# =========================================================
# CUSTOM CSS - UI DESIGN
# =========================================================
st.markdown("""
<style>
/* Main background */
.stApp {
    background: linear-gradient(135deg, #f8fbff 0%, #eef4ff 100%);
}

/* Main content */
.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Main title */
h1 {
    text-align: center;
    font-size: 42px !important;
    font-weight: 800 !important;
    color: #173b73 !important;
    margin-bottom: 8px !important;
}

/* Section headings */
h2, h3 {
    color: #173b73 !important;
    font-weight: 700 !important;
}

/* Description text */
.stApp p {
    color: #44546a;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #173b73 0%, #2459a6 100%);
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] li {
    color: white !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: white;
    border: 2px dashed #4f7fc7;
    border-radius: 16px;
    padding: 18px;
    box-shadow: 0 4px 15px rgba(23, 59, 115, 0.08);
}

/* Text area */
textarea {
    border-radius: 14px !important;
    border: 1px solid #b8c9e5 !important;
}

/* Input focus */
textarea:focus {
    border: 2px solid #3f73bd !important;
    box-shadow: 0 0 0 2px rgba(63, 115, 189, 0.12) !important;
}

/* Analyze button */
.stButton > button {
    width: 100%;
    border-radius: 12px;
    border: none;
    padding: 0.75rem 1rem;
    font-size: 18px;
    font-weight: 700;
    background: linear-gradient(90deg, #173b73, #3f73bd);
    color: white;
    transition: 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 18px rgba(23, 59, 115, 0.25);
}

/* Metric cards */
[data-testid="stMetric"] {
    background: white;
    border-radius: 16px;
    padding: 18px;
    border: 1px solid #dbe5f3;
    box-shadow: 0 5px 18px rgba(23, 59, 115, 0.08);
}

/* Metric value */
[data-testid="stMetricValue"] {
    color: #173b73 !important;
    font-weight: 800 !important;
}

/* Progress bar */
div[data-testid="stProgressBar"] > div > div {
    border-radius: 10px;
}

/* Alerts */
div[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Expander */
.streamlit-expanderHeader {
    background: white;
    border-radius: 12px;
    font-weight: 600;
}

/* Code / extracted text area */
.stTextArea textarea {
    background: #fbfdff !important;
}

/* Footer */
footer {
    visibility: hidden;
}

.app-footer {
    text-align: center;
    color: #6b7c93;
    font-size: 14px;
    padding-top: 15px;
}
</style>
""", unsafe_allow_html=True)

st.title("📄 AI Resume Analyzer & Job Matcher")

st.write(
    "Upload your resume and compare it with a job description using AI."
)


# =========================================================
# OLLAMA AI MODEL
# =========================================================
llm = OllamaLLM(model="llama3.2")


# =========================================================
# NORMAL PDF TEXT EXTRACTION
# =========================================================
def extract_pdf_text(file):

    try:
        file.seek(0)

        reader = PdfReader(file)

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        return text.strip()

    except Exception:
        return ""


# =========================================================
# OCR PDF EXTRACTION
# =========================================================
def extract_pdf_with_ocr(file):

    try:

        st.info("🔍 Scanned PDF detected. Running OCR...")

        # Read PDF bytes
        pdf_bytes = file.getvalue()

        if not pdf_bytes:
            st.error("❌ PDF file is empty.")
            return ""

        # Open PDF
        pdf_document = fitz.open(
            stream=pdf_bytes,
            filetype="pdf"
        )

        if len(pdf_document) == 0:
            st.error("❌ PDF contains no pages.")
            return ""

        extracted_text = ""

        # Process every page
        for page_number in range(len(pdf_document)):

            page = pdf_document[page_number]

            # Render page at high quality
            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(3, 3),
                alpha=False
            )

            # Convert to PIL image
            image = Image.frombytes(
                "RGB",
                [pixmap.width, pixmap.height],
                pixmap.samples
            )

            # Convert to grayscale
            image = ImageOps.grayscale(image)

            # Improve contrast
            image = ImageEnhance.Contrast(
                image
            ).enhance(2.0)

            # Sharpen image
            image = image.filter(
                ImageFilter.SHARPEN
            )

            # Try OCR with PSM 6
            text1 = pytesseract.image_to_string(
                image,
                config="--psm 6"
            )

            # Try OCR with PSM 11
            text2 = pytesseract.image_to_string(
                image,
                config="--psm 11"
            )

            # Select better result
            if len(text2.strip()) > len(text1.strip()):
                page_text = text2
            else:
                page_text = text1

            extracted_text += (
                f"\n--- Page {page_number + 1} ---\n"
            )

            extracted_text += page_text

        pdf_document.close()

        extracted_text = extracted_text.strip()

        # Check OCR result
        if len(extracted_text) < 20:

            st.error(
                "❌ OCR could not recognize readable text "
                "from this PDF."
            )

            return ""

        return extracted_text

    except Exception as e:

        st.error(
            f"❌ OCR Error: {e}"
        )

        return ""


# =========================================================
# DOCX TEXT EXTRACTION
# =========================================================
def extract_docx_text(file):

    try:

        document = Document(file)

        text = ""

        for paragraph in document.paragraphs:

            if paragraph.text.strip():

                text += paragraph.text + "\n"

        return text.strip()

    except Exception as e:

        st.error(
            f"❌ DOCX Error: {e}"
        )

        return ""


# =========================================================
# RESUME TEXT EXTRACTION
# =========================================================
def extract_resume_text(file):

    filename = file.name.lower()

    # -------------------------
    # DOCX
    # -------------------------
    if filename.endswith(".docx"):

        return extract_docx_text(file)

    # -------------------------
    # PDF
    # -------------------------
    elif filename.endswith(".pdf"):

        # First try normal text extraction
        text = extract_pdf_text(file)

        # If PDF is scanned/image PDF
        if len(text.strip()) < 50:

            text = extract_pdf_with_ocr(file)

        return text.strip()

    return ""


# =========================================================
# KEYWORD EXTRACTION
# =========================================================
def get_keywords(text):

    text = text.lower()

    words = re.findall(
        r"[a-zA-Z][a-zA-Z0-9+#.-]*",
        text
    )

    stop_words = {
        "the", "and", "for", "with",
        "this", "that", "from", "you",
        "your", "are", "was", "will",
        "have", "has", "our", "job",
        "role", "work", "using", "into",
        "their", "they", "about", "but",
        "not", "can", "all", "who",
        "how", "its"
    }

    keywords = {
        word
        for word in words
        if (
            word not in stop_words
            and len(word) > 2
        )
    }

    return keywords


# =========================================================
# MATCH SCORE
# =========================================================
def calculate_match(
    resume_text,
    job_text
):

    resume_keywords = get_keywords(
        resume_text
    )

    job_keywords = get_keywords(
        job_text
    )

    if not job_keywords:

        return 0, set(), set()

    matching = (
        resume_keywords.intersection(
            job_keywords
        )
    )

    missing = (
        job_keywords - resume_keywords
    )

    score = (
        len(matching) /
        len(job_keywords)
    ) * 100

    return (
        round(score, 2),
        matching,
        missing
    )


# =========================================================
# AI RESUME ANALYSIS
# =========================================================
def analyze_with_ai(
    resume_text,
    job_description
):

    prompt = f"""
You are an AI Resume Analyzer and Job Matching Assistant.

Analyze the candidate resume against the job description.

================ RESUME ================
{resume_text}

================ JOB DESCRIPTION ================
{job_description}

Give the analysis in this format:

1. Candidate Summary

2. Matching Skills

3. Missing Skills

4. Strengths

5. Weaknesses

6. Job Suitability

7. Improvement Suggestions

8. Final Recommendation

Also provide an overall suitability statement.

Keep the answer:
- Professional
- Clear
- Easy to understand
- Based only on the resume and job description
"""

    response = llm.invoke(prompt)

    return response


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("📌 Instructions")

st.sidebar.write(
    """
1. Upload your resume.
2. Enter the job description.
3. Click Analyze Resume.
4. View match score.
5. View matching skills.
6. View missing skills.
7. View AI analysis.
"""
)


# =========================================================
# UPLOAD RESUME
# =========================================================
st.header("1️⃣ Upload Resume")

uploaded_file = st.file_uploader(
    "Choose your Resume",
    type=["pdf", "docx"]
)


# =========================================================
# JOB DESCRIPTION
# =========================================================
st.header("2️⃣ Job Description")

job_description = st.text_area(
    "Paste Job Description",
    height=250,
    placeholder=(
        "Example: Python Developer with "
        "knowledge of Python, SQL, Machine "
        "Learning, Streamlit and LangChain..."
    )
)


# =========================================================
# ANALYZE BUTTON
# =========================================================
if st.button(
    "🔍 Analyze Resume",
    type="primary"
):

    if uploaded_file is None:

        st.error(
            "❌ Please upload your resume."
        )

    elif not job_description.strip():

        st.error(
            "❌ Please enter the job description."
        )

    else:

        with st.spinner(
            "🤖 AI is analyzing your resume..."
        ):

            # -----------------------------------------
            # EXTRACT RESUME TEXT
            # -----------------------------------------
            resume_text = extract_resume_text(
                uploaded_file
            )

            # -----------------------------------------
            # CHECK EXTRACTION
            # -----------------------------------------
            if not resume_text:

                st.error(
                    "❌ Could not extract text "
                    "from the resume."
                )

                st.info(
                    "Please upload a readable PDF "
                    "or DOCX resume."
                )

            else:

                st.success(
                    "✅ Resume text extracted successfully!"
                )

                # -----------------------------------------
                # MATCH SCORE
                # -----------------------------------------
                (
                    score,
                    matching,
                    missing
                ) = calculate_match(
                    resume_text,
                    job_description
                )

                # -----------------------------------------
                # SCORE
                # -----------------------------------------
                st.header(
                    "📊 Resume Match Score"
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Match Score",
                        f"{score}%"
                    )

                with col2:

                    st.metric(
                        "Matching Keywords",
                        len(matching)
                    )

                with col3:

                    st.metric(
                        "Missing Keywords",
                        len(missing)
                    )

                st.progress(
                    min(int(score), 100)
                )

                # -----------------------------------------
                # MATCHING SKILLS
                # -----------------------------------------
                st.subheader(
                    "✅ Matching Skills"
                )

                if matching:

                    st.write(
                        ", ".join(
                            sorted(matching)
                        )
                    )

                else:

                    st.warning(
                        "No matching skills found."
                    )

                # -----------------------------------------
                # MISSING SKILLS
                # -----------------------------------------
                st.subheader(
                    "❌ Missing Skills"
                )

                if missing:

                    st.write(
                        ", ".join(
                            sorted(missing)
                        )
                    )

                else:

                    st.success(
                        "🎉 No major missing skills!"
                    )

                # -----------------------------------------
                # AI ANALYSIS
                # -----------------------------------------
                st.header(
                    "🤖 AI Resume Analysis"
                )

                try:

                    ai_result = analyze_with_ai(
                        resume_text,
                        job_description
                    )

                    st.markdown(
                        ai_result
                    )

                except Exception as e:

                    st.error(
                        f"❌ AI Analysis Error: {e}"
                    )

                # -----------------------------------------
                # EXTRACTED TEXT
                # -----------------------------------------
                with st.expander(
                    "📄 View Extracted Resume Text"
                ):

                    st.text_area(
                        "Resume Content",
                        resume_text,
                        height=400
                    )


# =========================================================
# FOOTER
# =========================================================
st.markdown("---")

st.markdown(
    '<div class="app-footer">AI Resume Analyzer & Job Matcher | Python + Streamlit + LangChain + Ollama + OCR</div>',
    unsafe_allow_html=True
)