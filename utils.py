import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import PyPDF2
import pdfplumber
import re
import nltk
from nltk.corpus import stopwords
from typing import Dict, Any, List
import sqlite3
from pathlib import Path
import time
import openai
import json
import streamlit as st

nltk.download('stopwords', quiet=True)

def init_db(conn):
    """Initializes the SQLite database and the parsed_resumes table."""
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS parsed_resumes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE,
                email TEXT,
                skills TEXT,
                experience TEXT,
                education TEXT,
                fit_score REAL DEFAULT NULL,
                file_bytes BLOB,
                raw_text TEXT,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    except sqlite3.Error as e:
        print(f"Database error during initialization: {e}")

def save_parsed_resume(conn, data: Dict[str, Any]):
    """Saves parsed resume data (or updates it) to the database."""
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO parsed_resumes (filename, email, skills, experience, education, fit_score, raw_text, file_bytes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data["ID"],
            data["email"],
            data["parsed_data"]["skills"],
            data["parsed_data"]["experience"],
            data["parsed_data"]["education"],
            data.get("fit_score"),
            data["Resume_str"],
            data["file_bytes"]
        ))
        conn.commit()
    except sqlite3.Error as e:
        print(f"Error saving resume {data['ID']} to DB: {e}")

def get_all_parsed_resumes(conn):
    """Retrieves all records from the parsed_resumes table."""
    resumes = []
    try:
        # Set row_factory here to get dict-like rows
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT filename, email, skills, experience, education, fit_score, upload_date
            FROM parsed_resumes
            ORDER BY upload_date DESC
        """)

        for row in cursor.fetchall():
            resumes.append(dict(row))

    except sqlite3.Error as e:
        print(f"Database error during retrieval: {e}")
    return resumes

def get_resume_bytes_from_db(conn, filename: str) -> bytes:
    """Retrieves the file_bytes for a single resume from the database."""
    file_bytes = None
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT file_bytes FROM parsed_resumes WHERE filename = ?", (filename,)
        )
        result = cursor.fetchone()
        if result:
            file_bytes = result[0]
    except sqlite3.Error as e:
        print(f"Database error retrieving bytes for {filename}: {e}")
    return file_bytes



# --- LLM/SBERT Model Loading (MODIFIED for Streamlit) ---
@st.cache_resource
def load_model_once():
    """Load SBERT model once using Streamlit's resource cache."""
    try:
        model = SentenceTransformer("all-mpnet-base-v2", device="cpu")
        print("SBERT model loaded successfully for Streamlit resource cache.")
        return model
    except Exception as e:
        # Using st.error here, as the UI will be handled by the Streamlit app file
        # st.error(f"Error loading SBERT model: {e}")
        print(f"Error loading SBERT model: {e}")
        return None

# --- PDF/Email/Text Cleaning (Existing) ---
def extract_text_from_pdf(file):
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text.strip() or "No content"
    except Exception as e:
        try:
            file.seek(0)
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
            return text.strip() or "No content"
        except:
            return "No content"

def extract_email(text):
    if not isinstance(text, str) or not text.strip():
        return "No email found"
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    match = re.search(email_pattern, text)
    return match.group(0) if match else "No email found"

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower().strip()
    text = re.sub(r'\b(professional summary|summary|references|education|certifications)\b.*?(?=\b\w+\b|$)', '', text, flags=re.IGNORECASE)
    return text or "No content"

# --- Structured Data Extraction (ULTIMATE FIX implementation) ---
# --- Structured Data Extraction (ULTIMATE FIX implementation) ---
def extract_section_text(text: str, section_name: str, next_section_names: List[str]) -> str:
    """Extracts text content for a specific section (e.g., 'Education') using a pattern that prioritizes stop words."""

    section_name_normalized = section_name.strip()

    # 1. Create the START pattern
    pattern_start = r'[\r\n]{1,3}\s*[\-\*\.]*\s*' + re.escape(section_name_normalized) + r'\s*[:\-\.]{0,5}\s*[\r\n]{1,3}'

    # 2. Create the STOP pattern (very simple to ensure lookahead works)
    stop_patterns = [r'\s*' + re.escape(name.strip()) + r'\s*' for name in next_section_names if name.strip() != section_name_normalized]
    pattern_stop = r'|'.join(stop_patterns)

    # 3. Final regex: Lookahead for the next header (or end of string)
    regex = re.compile(
        pattern_start + r'(.*?)(?=\s*(' + pattern_stop + r')|$)',
        re.IGNORECASE | re.DOTALL | re.MULTILINE
    )

    text_normalized = re.sub(r'[\r\n]+', '\n', text)
    text_normalized = re.sub(r'[\-\*\•]', '', text_normalized)

    match = regex.search(text_normalized)

    if match:
        content = match.group(1).strip()
        return re.sub(r'[\r\n]{2,}', '\n', content).strip()

    return "Not Found"


def parse_resume_sections(text: str) -> Dict[str, str]:
    """Extracts Skills, Experience, and Education from raw resume text."""

    ALL_SECTIONS = [
        "SKILLS", "TECHNICAL SKILLS", "EXPERIENCE", "WORK HISTORY", "PROFESSIONAL EXPERIENCE",
        "EDUCATION", "PROJECTS", "CERTIFICATION", "ACHIEVEMENTS", "SUMMARY", "CONTACT", "REFERENCES"
    ]

    # --- 1. Extract Education ---
    education_text = extract_section_text(text, "EDUCATION", ALL_SECTIONS)
    if education_text == "Not Found":
        education_text = extract_section_text(text, "ACADEMIC QUALIFICATIONS", ALL_SECTIONS)

    # Improved education pattern
    education_pattern = r'((?:B\.\s?Tech|M\.\s?S|Ph\.\s?D|Bachelor|Master|Degree|Diploma|College|University)[\s\w\d,&-]*?(?:\d{4}|\d{2,4}[-\/]\d{2,4}))'
    education_mentions = [match.strip() for match in re.findall(education_pattern, education_text, re.IGNORECASE) if match.strip()]

    # --- 2. Extract Experience (Work History) ---
    experience_text = extract_section_text(text, "PROFESSIONAL EXPERIENCE", ALL_SECTIONS)
    if experience_text == "Not Found":
        experience_text = extract_section_text(text, "WORK EXPERIENCE", ALL_SECTIONS)

    if experience_text == "Not Found":
        experience_text = extract_section_text(text, "EXPERIENCE", ALL_SECTIONS)

    experience_pattern = r'(\w+\s+(engineer|developer|analyst|scientist|manager|consultant|intern)[\s\w\d,-]*?(at|on|for|\|)\s*[\w\s\&-]+)'
    experience_mentions = [match[0].strip() for match in re.findall(experience_pattern, experience_text, re.IGNORECASE)]

    # --- 3. Extract Skills (UNIVERSAL & DEEPER EXTRACTION) ---

    skills_text = extract_section_text(text, "SKILLS", ALL_SECTIONS)
    if skills_text == "Not Found":
        skills_text = extract_section_text(text, "TECHNICAL SKILLS", ALL_SECTIONS)

    SUB_HEADER_PATTERN = r'^\s*[\-\*]*(Languages|Libraries/Frameworks|Others|Core Libraries|ML/DL Frameworks|Data Libraries|Cloud & DevOps|Tools|Concepts|Programming Languages|Core|Deep Learning|Technical)\s*[:\-\.]{0,2}\s*[\r\n]'

    cleaned_skills_text = re.sub(SUB_HEADER_PATTERN, ' ', skills_text, flags=re.IGNORECASE | re.MULTILINE).strip()

    universal_skills = re.sub(r'[\r\n]+', ', ', cleaned_skills_text).strip()

    # Fallback (optional) - Removed for now to ensure section logic is fully verified.

    return {
        "education": " | ".join(education_mentions) or "Not specified",
        "experience": " | ".join(experience_mentions) or "Not specified",
        "skills": universal_skills if universal_skills and universal_skills != "Not Found" else "Not specified"
    }

# --- LLM-based Scoring and Justification ---
def get_llm_score_and_justification(resume_text: str, job_desc_text: str, api_key: str) -> (float, str):
    """
    Uses OpenAI's GPT model to score and generate a justification for a resume against a job description.
    """
    if not api_key:
        return 0.0, "OpenAI API key not provided."

    openai.api_key = api_key

    prompt = f"""
    You are an expert technical recruiter. Your task is to evaluate a candidate's resume against a job description.
    Provide a "fit_score" from 1.0 to 10.0, where 10.0 is a perfect match.
    Also, provide a concise "justification" (2-3 sentences) explaining your reasoning, highlighting key strengths or gaps.

    Return your response ONLY as a valid JSON object with two keys: "fit_score" and "justification".

    --- JOB DESCRIPTION ---
    {job_desc_text}

    --- RESUME ---
    {resume_text}
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        result_text = response.choices[0].message.content
        result_json = json.loads(result_text)

        score = float(result_json.get("fit_score", 0.0))
        justification = result_json.get("justification", "Could not parse justification from LLM response.")
        return score, justification
    except Exception as e:
        return 0.0, f"LLM API call failed: {str(e)}"
# --- Ranking and Similarity Logic (MODIFIED for Streamlit) ---

def vectorize_texts_sbert(texts, model, batch_size=100):
    # ... existing implementation ...
    valid_texts = [clean_text(t) if isinstance(t, str) and t.strip() else "No content" for t in texts]
    vectors = []
    for i in range(0, len(valid_texts), batch_size):
        batch = valid_texts[i:i + batch_size]
        batch_vectors = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        vectors.append(batch_vectors)
    return np.vstack(vectors)

def extract_keywords_from_job_desc(job_desc_text, top_n=10):
    # ... existing implementation ...
    stop_words = set(stopwords.words('english')).union({'seeking', 'expertise', 'summary', 'responsibilities', 'qualifications', 'must', 'have', 'experience', 'ability'})
    vectorizer = TfidfVectorizer(
        stop_words=list(stop_words),
        token_pattern=r'\b\w+\b',
        max_features=50,
        ngram_range=(1, 2)
    )
    try:
        tfidf_matrix = vectorizer.fit_transform([clean_text(job_desc_text)])
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray().flatten()
        keyword_scores = [(feature_names[i], tfidf_scores[i]) for i in range(len(feature_names))]
        keyword_scores.sort(key=lambda x: x[1], reverse=True)
        keywords = [kw for kw, score in keyword_scores[:top_n] if score > 0]
        return keywords if keywords else ["skills", "project"]
    except:
        return ["skills", "project"]

def filter_resumes(df, keywords, job_desc_text):
    # ... existing implementation ...
    # Check if user provided any keywords
    if isinstance(keywords, str) and keywords.strip():
        keywords_list = [k.strip().lower() for k in keywords.split(",")]
        # Apply a strict filter: only resumes matching user keywords are kept.
        # If no resumes match, an empty DataFrame is returned.
        filtered_df = df[df["Resume_str"].str.lower().str.contains("|".join(keywords_list), na=False)].copy()
        return filtered_df
    else:
        # If no keywords are provided by the user, return the original DataFrame without filtering.
        return df

def extract_key_matches(resume_text, job_desc_text, top_n=3):
    # ... existing implementation ...
    resume_text = clean_text(resume_text)
    job_desc_text = clean_text(job_desc_text)
    texts = [job_desc_text, resume_text]
    stop_words = set(stopwords.words('english')).union({'seeking', 'expertise', 'summary', 'responsibilities', 'qualifications', 'must', 'have', 'experience', 'ability'})

    vectorizer = TfidfVectorizer(
        stop_words=list(stop_words),
        token_pattern=r'\b\w{2,}\b',
        max_features=100,
        ngram_range=(1, 2)
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
    except:
        return ["No significant matches"]

    job_tfidf = tfidf_matrix[0].toarray().flatten()
    resume_tfidf = tfidf_matrix[1].toarray().flatten()

    common_terms = []
    for i, term in enumerate(feature_names):
        if job_tfidf[i] > 0.05 and resume_tfidf[i] > 0.05:
            common_terms.append((term, job_tfidf[i] + resume_tfidf[i]))

    common_terms.sort(key=lambda x: x[1], reverse=True)
    matches = [term for term, _ in common_terms][:top_n]

    return [match.replace(" ", "_") for match in matches] if matches else ["No significant matches"]


def rank_resumes(job_desc_text, keywords, top_n, uploaded_resumes, model, api_key: str, use_llm: bool, db_conn):
    """Rank resumes against a job description, save data to DB, using the passed SBERT model."""

    if not model:
        return None, "SBERT model not loaded. Please ensure the model is downloaded."

    # --- Refactor: Use process_uploaded_files to handle all file reading and parsing ---
    df = process_uploaded_files(uploaded_resumes, db_conn)
    if df.empty:
        return None, "No valid resumes uploaded or processed."

    job_desc_text = clean_text(job_desc_text.strip() or "No content")
    job_vector = vectorize_texts_sbert([job_desc_text], model)[0]

    df_filtered = filter_resumes(df, keywords, job_desc_text)

    if df_filtered.empty:
        return pd.DataFrame(), "No resumes found that match the specified keywords."

    top_n = min(top_n, len(df_filtered))

    # We still use SBERT for initial semantic filtering to find the most relevant candidates
    # before sending them to the more expensive LLM API.
    if not df_filtered.empty:
        resume_vectors = vectorize_texts_sbert(df_filtered["Resume_str"].tolist(), model)
        similarities = cosine_similarity(resume_vectors, job_vector.reshape(1, -1)).flatten()
        df_filtered.loc[:, "similarity"] = similarities

    final_top_resumes = []

    # Sort by SBERT similarity first to select the best candidates for the LLM
    candidates_for_llm = df_filtered.sort_values(by="similarity", ascending=False).head(top_n).copy()

    for index, row in candidates_for_llm.iterrows():
        original_entry = df[df["ID"] == row["ID"]].iloc[0]
        original_resume_text = original_entry["Resume_str"]
        parsed_info = original_entry["parsed_data"]

        matches = extract_key_matches(original_resume_text, job_desc_text)

        if use_llm:
            # --- NEW: Call LLM for score and justification ---
            rating_10, justification = get_llm_score_and_justification(original_resume_text, job_desc_text, api_key)

            # --- FALLBACK: If LLM fails (e.g., quota error), use SBERT score ---
            if rating_10 == 0.0 and "LLM API call failed" in justification:
                # Use the SBERT similarity to calculate a fallback score
                rating_10 = 1 + round(row["similarity"] * 9, 1)
                # Provide a user-friendly fallback justification
                justification = f"LLM call failed. Falling back to SBERT => Strong alignment on key terms like: {', '.join(matches).replace('_', ' ')}."
        else:
            # --- ORIGINAL METHOD: Use SBERT for score and TF-IDF for justification ---
            rating_10 = 1 + round(row["similarity"] * 9, 1)
            justification = f"Strong alignment on key terms like: {', '.join(matches).replace('_', ' ')}."

        final_top_resumes.append({
            "ID": row["ID"],
            "similarity": row["similarity"],
            "email": row["email"],
            "rating_10": rating_10,
            "key_matches": ", ".join(matches),
            "justification": justification,
            "skills": parsed_info["skills"],
            "experience": parsed_info["experience"],
            "education": parsed_info["education"],
            "fit_score": rating_10,  # Add fit_score for database
        })

    if not final_top_resumes:
        return pd.DataFrame(), "No candidates found after filtering."

    # Final ranking is now based on the LLM's score
    final_df = pd.DataFrame(final_top_resumes).sort_values(by="rating_10", ascending=False)
    return final_df, None

def process_uploaded_files(uploaded_resumes, conn):
    """
    Processes uploaded files to extract text, email, parsed data, and file bytes.
    This function is designed to be called once to avoid re-reading files.
    """
    data = []
    for resume in uploaded_resumes:
        if resume is None or not hasattr(resume, 'name') or resume.size == 0:
            continue

        filename = resume.name
        try:
            resume.seek(0)
            file_bytes = resume.getvalue() # Read bytes once

            if filename.lower().endswith(".txt"):
                text = file_bytes.decode("utf-8", errors="ignore").strip() or "No content"
            elif filename.lower().endswith(".pdf"):
                text = extract_text_from_pdf(resume)
            else:
                continue

            email = extract_email(text)
            parsed_info = parse_resume_sections(text)

            entry = {
                "ID": filename,
                "Resume_str": text,
                "email": email,
                "parsed_data": parsed_info,
                "file_bytes": file_bytes, # Store bytes
                "fit_score": None  # Initialize fit_score
            }
            data.append(entry)
            # Update fit_score in database after ranking
            if "fit_score" in entry and entry["fit_score"] is not None:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE parsed_resumes SET fit_score = ? WHERE filename = ?
                """, (entry["fit_score"], entry["ID"]))
                conn.commit()
            save_parsed_resume(conn, entry)

        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            continue

    return pd.DataFrame(data) if data else pd.DataFrame()
