# app.py
import streamlit as st
from pypdf import PdfReader
import io
import re
from sentence_transformers import SentenceTransformer, util
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from utils.structure_feedback import get_structure_feedback
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(page_title="Resume Matcher AI", layout="centered")
st.title("Resume Matcher AI")
st.markdown("Works for any job · Zero data stored · By Sai Sushma Mutyala")

@st.cache_resource
def load_models():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    return embedder, llm

embedder, llm = load_models()

resume_file = st.file_uploader("Upload resume (PDF)", type=["pdf"])
jd_text = st.text_area("Paste job description", height=200)

if resume_file and jd_text:
    # Extract text
    with st.spinner("Reading PDF..."):
        reader = PdfReader(io.BytesIO(resume_file.read()))
        resume_text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t: resume_text += t + "\n"

    # Hybrid score - Improved algorithm
    with st.spinner("Calculating score..."):
        # 1. Semantic similarity using chunks (more accurate)
        resume_chunks = [line.strip() for line in resume_text.split("\n") if len(line.strip()) > 30][:30]
        if resume_chunks:
            emb_chunks = embedder.encode(resume_chunks, convert_to_tensor=True)
            emb_jd = embedder.encode(jd_text, convert_to_tensor=True)
            semantic_scores = util.cos_sim(emb_chunks, emb_jd)
            semantic = semantic_scores.max().item() * 100
        else:
            semantic = 0
        
        # 2. Improved keyword matching (case-insensitive, handles variations)
        jd_lower = jd_text.lower()
        resume_lower = resume_text.lower()
        
        # Extract meaningful keywords (3+ chars, not common words)
        stopwords = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who', 'where', 'when', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'now'}
        
        jd_keywords = set(re.findall(r'\b[a-z]{3,}\b', jd_lower)) - stopwords
        resume_keywords = set(re.findall(r'\b[a-z]{3,}\b', resume_lower)) - stopwords
        
        # Calculate keyword match (more lenient)
        if jd_keywords:
            matched = len(jd_keywords & resume_keywords)
            keyword = (matched / len(jd_keywords)) * 100
        else:
            keyword = 0
        
        # 3. Combined score with more generous weighting
        # More weight on semantic (it's more accurate), but ensure keywords help
        base_score = 0.55 * semantic + 0.45 * keyword
        
        # 4. Multiple boost strategies to avoid underestimation
        # Boost if semantic is decent (even if keyword match is lower)
        if semantic > 45:
            base_score = base_score * 1.15  # 15% boost for good semantic match
        elif semantic > 35:
            base_score = base_score * 1.10  # 10% boost for moderate semantic match
        
        # Additional boost if both are decent
        if semantic > 40 and keyword > 25:
            base_score = base_score * 1.05  # Extra 5% boost
        
        # Ensure minimum score floor for decent resumes
        if semantic > 30 or keyword > 20:
            base_score = max(base_score, 45)  # Floor of 45 for any decent match
        
        score = round(min(base_score, 100), 1)  # Cap at 100

    col1, col2 = st.columns(2)
    with col1:
        st.metric("**ATS Match Score**", f"{score}%")
    with col2:
        if score >= 80: st.success("Strong")
        elif score >= 60: st.warning("Okay")
        else: st.error("Needs work")

    # Missing keywords
    missing = []
    phrases = []
    try:
        with st.spinner("Finding missing skills..."):
            parser = JsonOutputParser()
            prompt = ChatPromptTemplate.from_template(
                "Extract hard skills from JD only.\nJD: {jd}\nResume: {resume}\n"
                "Return JSON: {{\"missing_skills\": [...], \"suggested_phrases\": [...]}}"
            )
            chain = prompt | llm | parser
            out = chain.invoke({"jd": jd_text, "resume": resume_text})
            missing = out.get("missing_skills", [])[:15] if isinstance(out, dict) else []
            phrases = out.get("suggested_phrases", [])[:8] if isinstance(out, dict) else []
    except Exception as e:
        st.warning(f"Could not extract missing skills: {str(e)}")

    if missing:
        st.subheader("Missing Keywords")
        st.write(", ".join([f"**{m}**" for m in missing]))
    if phrases:
        st.subheader("Suggested Phrases")
        for p in phrases: st.markdown(f"• {p}")

    # Bullet rewriter
    st.subheader("Bullet Rewriter")
    bullet = st.text_input("Paste one bullet")
    if st.button("Rewrite") and bullet:
        try:
            keywords_to_use = ", ".join(missing[:5]) if missing else "relevant skills"
            rp = ChatPromptTemplate.from_template(
                "Rewrite this bullet including these keywords naturally: {kw}\n"
                "Keep <120 chars, make quantifiable if possible.\nBullet: {b}"
            )
            new = (rp | llm).invoke({"kw": keywords_to_use, "b": bullet}).content
            st.success(new)
        except Exception as e:
            st.error(f"Could not rewrite bullet: {str(e)}")

    # Structure feedback
    st.subheader("Resume Structure Feedback")
    try:
        structure_score, structure_feedback = get_structure_feedback(resume_text)
        for line in structure_feedback:
            if line.startswith("⚠"):
                st.warning(line)
            elif line.startswith("✅"):
                st.success(line)
            else:
                st.info(line)
    except Exception as e:
        st.error(f"Could not analyze structure: {str(e)}")

    st.success("Your data was processed only in memory and has already been deleted.")
