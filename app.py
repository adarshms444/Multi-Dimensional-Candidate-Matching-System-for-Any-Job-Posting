
import streamlit as st
import os
import importlib
import time
import sys

sys.path.append(os.getcwd())

# Set page config to wide mode and add a title/icon
st.set_page_config(
    page_title="TalentScout AI | Smart Hiring Assistant",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Professional Look ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
        border-color: #45a049;
        color: white;
    }
    .report-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border-left: 5px solid #4CAF50;
    }
    .metric-box {
        background-color: #f1f3f4;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    </style>
    """, unsafe_allow_html=True)


def save_uploaded_file(uploaded_file, save_dir="data"):
    """Saves an uploaded file to a directory."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    
    filename, ext = os.path.splitext(uploaded_file.name)
    file_path = os.path.join(save_dir, f"{filename}_{int(time.time())}{ext}")
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

@st.cache_resource
def get_system():
    """Caches the main system class."""
    import config, utils, llm_interface, retrieval, scoring, matching_system
    importlib.reload(config)
    importlib.reload(utils)
    importlib.reload(llm_interface)
    importlib.reload(retrieval)
    importlib.reload(scoring)
    importlib.reload(matching_system)
    
    from matching_system import CandidateMatchingSystem
    return CandidateMatchingSystem()

# --- 2. Sidebar (Configuration) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712009.png", width=60)
    st.title("TalentScout AI")
    st.markdown("---")
    
    st.subheader("1. Configuration")
    api_key = st.text_input("OpenRouter API Key", type="password", help="Enter your API key to enable the AI engine.")
    if api_key:
        os.environ['OPENROUTER_API_KEY'] = api_key
        st.success("✅ API Key Connected")
    
    st.subheader("2. Job Details")
    job_file = st.file_uploader("Upload Job Description", type=["pdf", "docx", "txt"], help="Upload the JD to match candidates against.")
    
    st.subheader("3. Candidate Pool")
    resume_files = st.file_uploader("Upload Resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True, help="Select multiple resumes to analyze.")
    
    st.markdown("---")
    run_button = st.button("🚀 Start Matching Analysis")
    
    st.markdown("---")
    st.caption("Powered by Mistral 7B & Vector Search")


st.title("🚀 AI Candidate Matching Dashboard")
st.markdown("### Intelligent ranking based on skills, experience, and semantic relevance.")

if run_button:
    if not api_key:
        st.warning("⚠️ Please enter your OpenRouter API Key in the sidebar to proceed.")
    elif not job_file:
        st.warning("⚠️ Please upload a Job Description document.")
    elif not resume_files:
        st.warning("⚠️ Please upload at least one candidate resume.")
    else:
        status_container = st.container()
        
        try:
            with status_container:
                st.info("🔄 Initializing AI Engine...")
                st.cache_resource.clear()
                system = get_system()
            
            with status_container:
                with st.spinner("📄 Analyzing Job Description..."):
                    job_path = save_uploaded_file(job_file, "data/job")
                    system.process_job_posting(job_path)
                st.success(f"✅ Job Processed: **{system.job.job_title}**")

                with st.spinner(f"👥 Analyzing {len(resume_files)} Resumes... (This may take a moment)"):
                    resume_paths = [save_uploaded_file(f, "data/resumes") for f in resume_files]
                    system.process_resumes(resume_paths)
                st.success(f"✅ {len(system.candidates_db)} Candidates Analyzed")
            
            with status_container:
                with st.spinner("🧠 Performing Hybrid Search & Multi-Dimensional Scoring..."):
                    final_reports = system.run_matching_pipeline()
            
            status_container.empty()
            st.balloons()
            
            st.subheader(f"🏆 Top Candidates ({len(final_reports)} Matches Found)")
            
            if not final_reports:
                st.error("No suitable candidates found based on the current criteria.")

            for i, report in enumerate(final_reports):
                score = report.get('final_score', 0)
                
                if score >= 85:
                    score_color = "green"
                    verdict = "🌟 Highly Recommended"
                    border_color = "#2ecc71"
                elif score >= 70:
                    score_color = "blue"
                    verdict = "✅ Recommended"
                    border_color = "#3498db"
                elif score >= 50:
                    score_color = "orange"
                    verdict = "⚠️ Consider"
                    border_color = "#f39c12"
                else:
                    score_color = "red"
                    verdict = "❌ Not Recommended"
                    border_color = "#e74c3c"

                with st.container():
                    st.markdown(f"""
                    <div class="report-card" style="border-left: 5px solid {border_color};">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h3 style="margin:0;">#{i+1} {report.get('name', 'Unknown Candidate')}</h3>
                            <div style="text-align: right;">
                                <h2 style="margin:0; color: {border_color};">{score:.1f}%</h2>
                                <span style="background-color: {border_color}; color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.8em;">{verdict}</span>
                            </div>
                        </div>
                        <p style="color: gray; font-size: 0.9em;">File: {report.get('filename', 'N/A').split('/')[-1]}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    tab1, tab2, tab3 = st.tabs(["📝 AI Analysis", "📊 Score Breakdown", "🔍 Raw Data"])
                    
                    with tab1:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("#### ✅ Strengths")
                            st.info(report.get('strengths', 'No analysis available.'))
                        with col2:
                            st.markdown("#### ⚠️ Gaps")
                            st.warning(report.get('gaps', 'No analysis available.'))
                        
                        st.markdown("#### 💡 Recommendation")
                        st.success(report.get('notes', 'No recommendation available.'))

                    with tab2:
                        st.markdown("#### Scoring Dimensions")
                        cols = st.columns(3)
                        metrics = [
                            ("Must-Have Skills", int(report.get('must_have_score', 0))),
                            ("Important Skills", int(report.get('important_score', 0))),
                            ("Experience Match", int(report.get('experience_score', 0))),
                            ("Recency", int(report.get('recency_score', 0))),
                            ("Domain Match", int(report.get('domain_score', 0))),
                            ("Nice-to-Haves", int(report.get('nice_to_have_score', 0)))
                        ]
                        
                        for idx, (label, value) in enumerate(metrics):
                            with cols[idx % 3]:
                                st.metric(label, f"{value}/100")
                                st.progress(min(value, 100) / 100)

                    with tab3:
                        st.json(report)
                
                st.write("")

        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            st.exception(e)
else:
    st.info("👈 Please upload a Job Description and Candidate Resumes in the sidebar to start.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        ### 🤖 Intelligent Parsing
        Uses LLMs to understand context, not just keywords. Extracts implicit skills and domains.
        """)
    with col2:
        st.markdown("""
        ### ⚖️ Hybrid Retrieval
        Combines semantic vector search with traditional keyword matching for best-in-class accuracy.
        """)
    with col3:
        st.markdown("""
        ### 📊 Explainable Scoring
        Provides detailed score breakdowns and AI-written explanations for every decision.
        """)
