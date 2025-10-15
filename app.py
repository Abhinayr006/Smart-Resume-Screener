import streamlit as st
import pandas as pd
import os
from utils_new import load_model_once, rank_resumes, get_all_parsed_resumes, init_db, get_resume_bytes_from_db
from io import BytesIO
import base64
from dotenv import load_dotenv
import sqlite3

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration ---
st.set_page_config(
    layout="wide",
    page_title="Smart Resume Screener",
    page_icon="🚀",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Dark Theme and Unique Styling ---
st.markdown("""
<style>
    .main {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #ff6b6b;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #ff5252;
    }
    .stTextInput>div>div>input, .stTextArea>div>textarea {
        background-color: #2d2d2d;
        color: #ffffff;
        border: 1px solid #555;
        border-radius: 5px;
    }
    .stFileUploader>div>div {
        background-color: #2d2d2d;
        border: 1px solid #555;
        border-radius: 5px;
    }
    .stExpander {
        background-color: #2d2d2d;
        border: 1px solid #555;
        border-radius: 5px;
    }
    .stDataFrame {
        background-color: #2d2d2d;
        color: #ffffff;
    }
    .stSelectbox>div>div {
        background-color: #333;
        color: #ffffff;
    }
    h1, h2, h3 {
        color: #ff6b6b;
    }
    .stSuccess {
        background-color: #4caf50;
        color: white;
    }
    .stError {
        background-color: #f44336;
        color: white;
    }
    .stWarning {
        background-color: #ff9800;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- Resource Caching ---
@st.cache_resource
def get_db_connection(version="1.1"):
    """Creates and caches an in-memory SQLite database connection for the session."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    st.write(f"Initializing DB v{version}...")
    init_db(conn)
    return conn

model = load_model_once()
db_conn = get_db_connection()

if model is None:
    st.error("SBERT model loading failed. Please check dependencies and memory allocation.")

# --- Initialize state ---
if 'ranked_resumes' not in st.session_state:
    st.session_state.ranked_resumes = None
if 'job_desc_for_results' not in st.session_state:
    st.session_state.job_desc_for_results = None

# --- Main App ---
st.title("🚀 Smart Resume Screener")
st.markdown("Welcome to the smart resume screening tool! Upload job descriptions and resumes to find top candidates.")

# --- Dropdown for Navigation ---
page = st.selectbox("Select Page", ["🏆 Rank Resumes", "💾 Database Viewer"])

if page == "🏆 Rank Resumes":
    # --- Layout: Left for main content, Right for configuration ---
    col_left, col_right = st.columns([3, 1])

    with col_left:
        # --- Job Description Input ---
        st.subheader("📋 Job Description")
        job_desc_file = st.file_uploader("Upload Job Description (.txt)", type=['txt'])

        if job_desc_file:
            job_desc_bytes = job_desc_file.read()
            final_job_desc = job_desc_bytes.decode("utf-8", errors="ignore").strip()
        else:
            final_job_desc = None

        if not final_job_desc:
            st.warning("Please provide a job description.")

        # --- Resume Upload ---
        st.subheader("📄 Upload Resumes")
        resumes = st.file_uploader("Select Resumes (.txt, .pdf)", type=['txt', 'pdf'], accept_multiple_files=True)
        if resumes:
            st.info(f"📎 {len(resumes)} resume(s) uploaded.")

        # --- Rank Button ---
        if st.button("🚀 Rank Candidates", type="primary", use_container_width=True):
            st.session_state.ranked_resumes = None
            st.session_state.job_desc_for_results = None

            valid_resumes = [f for f in resumes if f and f.size > 0] if resumes else []

            if not final_job_desc:
                st.error("Job description is required.")
            elif not valid_resumes:
                st.error("Please upload at least one resume.")
            elif st.session_state.get("use_llm") and not st.session_state.get("openai_api_key"):
                st.error("OpenAI API key required for LLM ranking.")
            elif model is None:
                st.error("SBERT model not loaded.")
            else:
                with st.spinner('🔍 Analyzing and ranking candidates...'):
                    top_resumes_df, error = rank_resumes(
                        job_desc_text=final_job_desc,
                        keywords="",  # Removed keywords
                        top_n=10,  # Fixed to 10
                        uploaded_resumes=valid_resumes,
                        model=model,
                        api_key=st.session_state.get("openai_api_key", ""),
                        use_llm=st.session_state.get("use_llm", False),
                        db_conn=db_conn
                    )

                if error:
                    st.error(error)
                elif top_resumes_df.empty:
                    st.warning("No matching resumes found.")
                else:
                    st.success(f"✅ Ranked {len(top_resumes_df)} candidates!")
                    st.session_state.ranked_resumes = top_resumes_df
                    st.session_state.job_desc_for_results = final_job_desc

        # --- Display Results ---
        if st.session_state.ranked_resumes is not None and not st.session_state.ranked_resumes.empty:
            df = st.session_state.ranked_resumes
            st.subheader(f"🏅 Top {len(df)} Candidates")

            for idx, row in df.iterrows():
                with st.expander(f"#{idx+1} - {row['ID']} | Score: {row['rating_10']}/10", expanded=idx < 3):
                    st.write(f"**Fit Score:** {row['similarity']:.4f}")
                    st.write(f"**Justification:** {row['justification']}")

                    # Structured Data
                    st.markdown("---")
                    col_s, col_e, col_ed = st.columns(3)
                    with col_s:
                        st.metric("Skills", "Extracted" if row['skills'] != "Not specified" else "Failed")
                        st.caption(row['skills'])
                    with col_e:
                        st.metric("Experience", "Extracted" if row['experience'] != "Not specified" else "Failed")
                        st.caption(row['experience'])
                    with col_ed:
                        st.metric("Education", "Extracted" if row['education'] != "Not specified" else "Failed")
                        st.caption(row['education'])

    with col_right:
        # --- Configuration Section ---
        st.header("⚙️ Configuration")
        default_api_key = os.getenv("OPENAI_API_KEY", "")
        st.session_state.use_llm = st.checkbox("Use LLM for Advanced Ranking", value=bool(default_api_key), help="Uses GPT for scoring and justification.")
        st.session_state.openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=default_api_key,
            help="Required for LLM-based ranking.",
            disabled=not st.session_state.use_llm
        )
        st.markdown("[Get API Key](https://platform.openai.com/api-keys)")

        # --- Database Options ---
        st.markdown("---")
        st.subheader("Database Options")
        if st.button("Clear Database", help="Clear all stored resumes from the database."):
            # Clear the database
            cursor = db_conn.cursor()
            cursor.execute("DELETE FROM parsed_resumes")
            db_conn.commit()
            st.success("Database cleared successfully!")
            st.rerun()

        # --- Database Viewer ---
        st.markdown("### Stored Resumes")
        resumes_data = get_all_parsed_resumes(db_conn)
        if resumes_data:
            st.success(f"📊 {len(resumes_data)} resumes stored.")
            df_db = pd.DataFrame(resumes_data)
            df_db = df_db.rename(columns={
                'filename': 'File Name',
                'email': 'Email',
                'skills': 'Skills',
                'experience': 'Experience',
                'education': 'Education',
                'fit_score': 'Fit Score',
                'upload_date': 'Upload Date'
            })
            st.dataframe(df_db, use_container_width=True, hide_index=True)

            csv_db = BytesIO()
            df_db.to_csv(csv_db, index=False)
            st.download_button("💾 Download Full Database", data=csv_db.getvalue(), file_name="database.csv", mime="text/csv", use_container_width=True)
        else:
            st.info("No resumes stored yet. Rank some resumes first!")

elif page == "💾 Database Viewer":
    resumes_data = get_all_parsed_resumes(db_conn)
    if resumes_data:
        st.success(f"📊 {len(resumes_data)} resumes stored.")
        df_db = pd.DataFrame(resumes_data)
        df_db = df_db.rename(columns={
            'filename': 'File Name',
            'email': 'Email',
            'skills': 'Skills',
            'experience': 'Experience',
            'education': 'Education',
            'fit_score': 'Fit Score',
            'upload_date': 'Upload Date'
        })
        st.dataframe(df_db, use_container_width=True, hide_index=True)

        csv_db = BytesIO()
        df_db.to_csv(csv_db, index=False)
        st.download_button("💾 Download Full Database", data=csv_db.getvalue(), file_name="database.csv", mime="text/csv", use_container_width=True)
    else:
        st.info("No resumes stored yet. Rank some resumes first!")
