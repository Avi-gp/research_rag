import sys
import os
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import requests
import json
from pathlib import Path
import os
from config.settings import settings

# Page config
st.set_page_config(
    page_title="🧠 ResearchMind:RAG-Powered Research Paper Q&A and Summarization System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
def load_custom_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styling */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom title styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Navigation buttons */
    .nav-button {
        display: flex;
        align-items: center;
        width: 100%;
        padding: 12px 16px;
        margin: 8px 0;
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        color: #374151;
        text-decoration: none;
        font-weight: 500;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .nav-button:hover {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .nav-button.active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .nav-button-icon {
        margin-right: 12px;
        font-size: 1.2rem;
    }
    
    /* Cards */
    .card {
        background: white;
        border-radius: 16px;
        padding: 8px 0 6px 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
        margin-bottom: 20px;
    }
    
    .card-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #111827;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
    }
    
    .card-icon {
        margin-right: 12px;
        font-size: 1.5rem;
    }
    
    /* Summary card with better text visibility */
    .summary-card {
        background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
        border-radius: 16px;
        padding: 25px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
        margin: 5px 0;
        color: #1f2937;
        line-height: 1.6;
    }
    
    /* Answer section styling */
    .answer-section {
        background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        border-left: 4px solid #667eea;
        color: #1f2937;
        line-height: 1.6;
    }
    
    /* Status indicators */
    .status-connected {
        background: #10b981;
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .status-error {
        background: #ef4444;
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* Custom buttons */
    .primary-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .primary-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .danger-button {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .danger-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);
    }
    
    /* Metrics */
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #e2e8f0;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 4px;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6b7280;
        font-weight: 500;
    }
    
    /* Source item styling */
    .source-item {
        background: #f8fafc;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border-left: 3px solid #667eea;
    }
    
    .source-header {
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 8px;
    }
    
    .source-score {
        background: #667eea;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        display: inline-block;
        margin: 5px 0;
    }
    
    /* Upload area */
    .upload-area {
        border: 2px dashed #d1d5db;
        border-radius: 12px;
        padding: 40px;
        text-align: center;
        margin: 20px 0;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #667eea;
        background: #f8fafc;
    }
    
    /* Hide Streamlit branding */
    .stDeployButton {display: none;}
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }
    
    /* Animation for page transitions */
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """, unsafe_allow_html=True)
    
def load_footer():
    st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background: rgba(0, 0, 0, 0);
        color: #94a3b8;
        text-align: center;
        padding: 10px 0;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        font-size: 0.9rem;
        font-weight: 500;
        z-index: 999;
        backdrop-filter: blur(15px);
    }
    
    .footer a {
        color: #94a3b8;
        text-decoration: none;
        font-weight: 500;
    }
    
    .footer a:hover {
        color: #c4b5fd;
        text-decoration: none;
    }
    
    /* Add padding to main content to avoid footer overlap */
    .main .block-container {
        padding-bottom: 70px;
    }
    </style>
    <div class="footer">
        <p>© 2025 ResearchMind - Developed by <a href="https://github.com/Avi-gp" target="_blank">Suryansh Gupta</a> | Powered by RAG</p>
    </div>
    """, unsafe_allow_html=True)

# API base URL
API_BASE_URL = f"http://localhost:8000/api/v1"

def check_api_connection():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def create_nav_button(icon, text, is_active=False):
    """Create a navigation button"""
    active_class = "active" if is_active else ""
    return f"""
    <div class="nav-button {active_class}">
        <span class="nav-button-icon">{icon}</span>
        <span>{text}</span>
    </div>
    """

def main():
    load_custom_css()
    load_footer()
    
    # Initialize session state for page navigation
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Upload Research Papers"
    
    # Header
    st.markdown('<h1 class="main-title">🧠 ResearchMind</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">RAG-Powered Research Paper Q&A and Summarization System</p>', unsafe_allow_html=True)
    
    # Check API connection
    api_status = check_api_connection()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if api_status:
            st.markdown('<div class="status-connected">✅ API Connected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-error">❌ API Disconnected</div>', unsafe_allow_html=True)
            st.error("API is not running. Please start the FastAPI server first.")
            st.code("python api/main.py", language="bash")
            return
    
    # Sidebar Navigation
    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        
        # Navigation buttons
        pages = [
            ("📤", "Upload Research Papers"),
            ("❓", "Ask Questions"),
            ("📋", "Research Summary"),
            ("📊", "System Statistics"),
            ("🗑️", "Database Management"),
            ("❔", "Help & Instructions")
        ]
        
        for icon, page_name in pages:
            if st.button(f"{icon} {page_name}", key=f"nav_{page_name}", use_container_width=True):
                st.session_state.current_page = page_name
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        **ResearchMind** leverages AI to help you:
        - 📚 Analyze research papers
        - 🔍 Extract key information
        - 💡 Answer questions
        - 📈 Generate summaries
        """)
    
    # Main content area with fade-in animation
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    
    # Route to appropriate page
    current_page = st.session_state.current_page
    
    if current_page == "Upload Research Papers":
        upload_page()
    elif current_page == "Ask Questions":
        question_page()
    elif current_page == "Research Summary":
        summary_page()
    elif current_page == "System Statistics":
        stats_page()
    elif current_page == "Database Management":
        database_management_page()
    elif current_page == "Help & Instructions":
        help_page()
    
    st.markdown('</div>', unsafe_allow_html=True)

def upload_page():
    st.markdown('<div class="card"><div class="card-header"><span class="card-icon">📤</span>Upload Research Papers</div>', unsafe_allow_html=True)
    
    st.markdown("Upload your research papers in PDF format to build your knowledge base.")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Select one or more PDF research papers to upload",
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        # Display upload preview
        st.markdown("### 📋 Upload Preview")
        for i, file in enumerate(uploaded_files, 1):
            st.markdown(f"**{i}.** {file.name} ({file.size / 1024 / 1024:.2f} MB)")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🚀 Process Documents", type="primary", use_container_width=True):
                process_documents(uploaded_files)
    
    st.markdown('</div>', unsafe_allow_html=True)

def process_documents(uploaded_files):
    """Process uploaded documents"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Upload files
        status_text.text("📤 Uploading files...")
        progress_bar.progress(25)
        
        files_data = []
        for file in uploaded_files:
            files_data.append(("files", (file.name, file.read(), "application/pdf")))
        
        response = requests.post(f"{API_BASE_URL}/upload", files=files_data)
        
        if response.status_code != 200:
            st.error("❌ Error uploading files")
            return
        
        # Step 2: Process documents
        status_text.text("🔄 Processing documents...")
        progress_bar.progress(75)
        
        response = requests.post(f"{API_BASE_URL}/ingest")
        
        if response.status_code == 200:
            result = response.json()
            progress_bar.progress(100)
            status_text.text("✅ Processing complete!")
            
            st.success(f"🎉 Successfully processed {len(uploaded_files)} documents!")
            st.json(result)
        else:
            st.error("❌ Error processing documents")
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
    finally:
        # Clear progress indicators after a delay
        import time
        time.sleep(2)
        progress_bar.empty()
        status_text.empty()

def question_page():
    st.markdown('<div class="card"><div class="card-header"><span class="card-icon">❓</span>Ask Questions</div>', unsafe_allow_html=True)
    
    st.markdown("### 🤔 Ask questions about your research papers")
    st.markdown("""
    Get AI-powered insights by asking questions about:
    - 🔍 Main findings and conclusions
    - 📊 Research methodologies
    - 📈 Results and analysis
    - 🔄 Comparisons with existing literature
    """)
    
    # Question input
    question = st.text_area(
        "Your Question",
        height=120,
        placeholder="e.g., What are the main findings of the research? What methodologies were used? How do the results compare to existing literature?",
        help="Ask specific questions about the research papers you've uploaded"
    )
    
    # Advanced settings
    with st.expander("🔧 Advanced Search Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            k = st.slider("📊 Relevant Chunks", 1, 15, 5, help="Number of document chunks to analyze")
        with col2:
            # Add checkbox for threshold usage
            use_threshold = st.checkbox("🎯 Use Similarity Threshold", value=True, 
                                      help="Enable/disable similarity threshold filtering")
            
            if use_threshold:
                threshold = st.slider("🎯 Similarity Threshold", 0.2, 0.9, 0.5, 0.05, 
                                    help="Minimum similarity score (higher = more strict)")
            else:
                threshold = None
                st.info("🔄 No threshold filtering - all top results will be returned")
    
    if st.button("🔍 Get Answer", type="primary", key="get_answer_btn"):
        if question.strip():
            get_answer(question, k, threshold)
        else:
            st.warning("⚠️ Please enter a question first!")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    
def get_answer(question, k, threshold):
    """Get answer for the question using similarity search"""
    with st.spinner("🧠 Analyzing documents and generating answer..."):
        try:
            data = {
                "question": question, 
                "k": k
            }
            
            # Only include threshold if it's not None
            if threshold is not None:
                data["threshold"] = threshold
            
            response = requests.post(f"{API_BASE_URL}/ask", data=data)
            
            if response.status_code == 200:
                result = response.json()
                
                # Answer section
                st.markdown("### 🤖 Answer")
                st.markdown(f"""
                <div class="answer-section">
                    {result.get('answer', 'No answer generated')}
                </div>
                """, unsafe_allow_html=True)
                
                # Show threshold information - FIXED
                threshold_applied = result.get('threshold_applied', False)
                threshold_value = result.get('threshold_value')  # Changed from 'threshold_used' to 'threshold_value'
                
                if threshold_applied and threshold_value is not None:
                    st.info(f"🎯 Similarity threshold {threshold_value} was applied")
                elif not threshold_applied:
                    st.info("🔄No threshold filtering - all top results were considered")

                # Sources section
                if result.get('sources'):
                    st.markdown("### 📚 Sources")
                    
                    for i, source in enumerate(result['sources'], 1):
                        similarity_score = source.get('similarity_score', 0)
                        rank = source.get('rank', i)
                        
                        # Color code similarity scores
                        if similarity_score >= 0.7:
                            score_color = "#0c6b22"  # Green - highly similar
                        elif similarity_score >= 0.5:
                            score_color = "#fbe200"  # Yellow - good similarity
                        elif similarity_score >= 0.3:
                            score_color = "#ff8c00"  # Orange - moderate similarity
                        else:
                            score_color = "#4e050c"  # Red - lower similarity
                        
                        st.markdown(f"""
                        <div class="source-item">
                            <div class="source-header">📄 Source {i}: {source.get('source', 'Unknown')}</div>
                            <div class="source-score" style="color: {score_color};">
                            Similarity: {similarity_score:.3f} | Rank: {rank}
                            </div>
                            <div><strong>Chunk:</strong> {source.get('chunk_id', 'N/A')}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    if threshold_applied:
                        st.warning(f"⚠️ No sources found above similarity threshold {threshold_value}. Try lowering the threshold or disabling it.")
                    else:
                        st.warning("⚠️ No sources found.")
                
                # Analytics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📊 Chunks Used", result.get('context_used', 0))
                with col2:
                    st.metric("📄 Sources Found", len(result.get('sources', [])))
                with col3:
                    avg_sim = result.get('average_similarity', 0)
                    st.metric("📈 Avg Similarity", f"{avg_sim:.3f}")
                    
            else:
                st.error(f"❌ Error generating answer: HTTP {response.status_code}")
                try:
                    error_detail = response.json()
                    st.json(error_detail)
                except:
                    st.text(response.text)
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


def summary_page():
    st.markdown('<div class="card"><div class="card-header"><span class="card-icon">📋</span>Research Summary</div>', unsafe_allow_html=True)
    
    st.markdown("Generate comprehensive summaries of your research papers.")
    
    # Query input
    query = st.text_input(
        "Focus Topic (Optional)",
        placeholder="e.g., methodology, results, conclusions, limitations",
        help="Specify a particular aspect to focus the summary on, or leave blank for a general summary"
    )
    
    # Advanced settings
    with st.expander("🔧 Advanced Summary Settings", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            k = st.slider("📊 Document Chunks", 5, 25, 10, help="Number of document chunks to include in summary")
        with col2:
            # Add checkbox for threshold usage
            use_threshold = st.checkbox("🎯 Use Similarity Threshold", value=True, 
                                      help="Enable/disable similarity threshold filtering", key="summary_threshold_checkbox")
            
            if use_threshold:
                threshold = st.slider("🎯 Similarity Threshold", 0.2, 0.9, 0.4, 0.05, 
                                    help="Minimum similarity score", key="summary_threshold_slider")
            else:
                threshold = None
                st.info("🔄 No threshold filtering")
                
        with col3:
            summary_type = st.selectbox(
                "📋 Summary Type",
                ["Comprehensive", "Executive Summary", "Key Findings", "Methodology Focus"],
                help="Choose the type of summary you want"
            )
    
    if st.button("📋 Generate Summary", type="primary"):
        generate_summary(query, k, threshold, summary_type)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    
def generate_summary(query, k, threshold, summary_type):
    """Generate summary of documents using similarity search"""
    with st.spinner("🔍 Analyzing documents and generating summary..."):
        try:
            data = {
                "k": k
            }
            
            # Only include threshold if it's not None
            if threshold is not None:
                data["threshold"] = threshold
            
            # Combine query with summary type
            if query:
                data["query"] = f"{summary_type}: {query}" if query else summary_type
            else:
                data["query"] = summary_type
            
            response = requests.post(f"{API_BASE_URL}/summarize", data=data)
            
            if response.status_code == 200:
                result = response.json()
                
                # Summary section
                st.markdown(f"### 📄 {summary_type}")
                st.markdown(f"""
                <div class="summary-card">
                    {result.get('summary', 'No summary generated')}
                </div>
                """, unsafe_allow_html=True)
                
                # Show threshold information - FIXED
                threshold_applied = result.get('threshold_applied', False)
                threshold_value = result.get('threshold_value')  # Changed from 'threshold_used' to 'threshold_value'

                if threshold_applied and threshold_value is not None:
                    st.info(f"🎯 Similarity threshold {threshold_value} was applied")
                elif not threshold_applied:
                    st.info("🔄No threshold filtering - all top results were considered")

                
                # Sources section
                if result.get('sources'):
                    st.markdown("### 📚 Sources Analyzed")
                    
                    for source in result['sources']:
                        if isinstance(source, dict):
                            source_name = source.get('source', 'Unknown')
                            chunks_used = source.get('chunks_used', 0)
                            avg_similarity = source.get('avg_similarity', 0.0)
                            best_similarity = source.get('best_similarity', 0.0)
                            
                            # Color code based on best similarity
                            if best_similarity >= 0.8:
                                similarity_color = "#0c6b22"  # Green
                            elif best_similarity >= 0.7:
                                similarity_color = "#033978"  # Yellow
                            elif best_similarity >= 0.5:
                                similarity_color = "#660c4e"  # Orange
                            else:
                                similarity_color = "#620303"  # Red
                            
                            st.markdown(f"""
                            <div class="source-summary-item">
                                <div class="source-name">📄 {source_name}</div>
                                <div class="source-metrics" style="color: {similarity_color};">
                                    📊 {chunks_used} chunks | 📈 Avg: {avg_similarity:.3f} | 🎯 Best: {best_similarity:.3f}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.write(f"📄 {str(source)}")
                else:
                    if threshold_applied:
                        st.warning(f"⚠️ No sources found above similarity threshold {threshold_value}. Try lowering the threshold or disabling it.")
                    else:
                        st.warning("⚠️ No sources found.")
                
                # Analytics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📊 Chunks Analyzed", result.get('context_used', 0))
                with col2:
                    st.metric("📄 Sources Used", len(result.get('sources', [])))
                with col3:
                    avg_sim = result.get('average_similarity', 0)
                    st.metric("📈 Avg Similarity", f"{avg_sim:.3f}")
                    
            else:
                st.error(f"❌ Error generating summary: HTTP {response.status_code}")
                try:
                    error_detail = response.json()
                    st.json(error_detail)
                except:
                    st.text(response.text)
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

            
def stats_page():
    st.markdown('<div class="card"><div class="card-header"><span class="card-icon">📊</span>System Statistics</div>', unsafe_allow_html=True)
    
    try:
        response = requests.get(f"{API_BASE_URL}/stats")
        if response.status_code == 200:
            stats = response.json()
            
            # Main metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_docs = stats.get('vector_store', {}).get('total_documents', 0)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{total_docs}</div>
                    <div class="metric-label">📄 Total Documents</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                files_processed_count = stats.get('pdf_storage', {}).get('files_count', 0)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{files_processed_count}</div>
                    <div class="metric-label">🗂 Files Processed</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                dimension = stats.get('vector_store', {}).get('embedding_dimension', 'N/A')
                if dimension == 'N/A' or dimension is None or dimension == 0:
                    dimension_display = 'N/A'
                else:
                    dimension_display = str(dimension)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{dimension_display}</div>
                    <div class="metric-label">🔢 Vector Dimension</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                vector_store_stats = stats.get('vector_store', {})
                status = vector_store_stats.get('status', 'Not Ready')
                
                # Determine status color
                if status == "Ready":
                    status_color = "#0c6b22"  # Green
                    display_status = "✅ Ready"
                else:
                    status_color = "#4e050c"  # Red
                    display_status = "❌ Not Ready"
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="font-size: 1.5rem; color: {status_color};">{display_status}</div>
                    <div class="metric-label">🗂️ Vector Store</div>
                </div>
                """, unsafe_allow_html=True)
            
            # System capabilities
            st.markdown("### 🔧 System Capabilities")
            search_capabilities = stats.get('search_capabilities', {})
            quality_thresholds = stats.get('quality_thresholds', {})
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"""
                **🔍 Search Features:**
                - Similarity Search: {'✅' if search_capabilities.get('similarity_search') else '❌'}
                - Source Search: {'✅' if search_capabilities.get('source_based_search') else '❌'}
                - Threshold Support: {'✅' if search_capabilities.get('threshold_support') else '❌'}
                - Optional Threshold: ✅ 
                - Cosine Similarity: {'✅' if search_capabilities.get('cosine_similarity') else '❌'}
                """)
            
            with col2:
                st.info(f"""
                **🎯 Recommended Quality Thresholds:**
                - High Quality: {quality_thresholds.get('high_quality', 0.7)}
                - Good Quality: {quality_thresholds.get('good_quality', 0.5)}  
                - Moderate: {quality_thresholds.get('moderate_quality', 0.3)}
                - Low Quality: {quality_thresholds.get('low_quality', 0.2)}
                - No Threshold: All results returned
                """)
            
            # Vector Store Details
            st.markdown("### 🗄️ Vector Store Details")
            vector_store_stats = stats.get('vector_store', {})
            
            col1, col2, col3 = st.columns(3)
            with col1:
                store_type = vector_store_stats.get('vector_store_type', 'Unknown')
                st.info(f"**Type:** {store_type}")
                
            with col2:
                distance_strategy = vector_store_stats.get('distance_strategy', 'Unknown')
                st.info(f"**Distance:** {distance_strategy}")
                
            with col3:
                collection_name = vector_store_stats.get('collection_name', 'Unknown')
                st.info(f"**Collection:** {collection_name}")
            
            # Storage information
            st.markdown("### 💾 Storage Information")
            col1, col2 = st.columns(2)
            
            with col1:
                storage_size = stats.get('pdf_storage', {}).get('total_size_mb', 0)
                st.info(f"📁 **Total Storage:** {storage_size} MB")
            
            with col2:
                chunks_processed = stats.get('pdf_processor', {}).get('total_chunks_processed', 0)
                st.info(f"📊 **Total Chunks:** {chunks_processed}")
            
            # Files breakdown - removed file size column
            processed_files_info = stats.get('pdf_processor', {}).get('processed_files_info', {})
            if processed_files_info:
                st.markdown("### 📂 Processed Files")
                for filename, info in processed_files_info.items():
                    chunks = info.get('chunks', 0)
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"📄 {filename}")
                    with col2:
                        st.write(f"📊 {chunks} chunks")
                
        else:
            st.error(f"❌ Error fetching statistics: HTTP {response.status_code}")
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
def database_management_page():
    st.markdown('<div class="card"><div class="card-header"><span class="card-icon">🗑️</span>Database Management</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Manage your vector database and stored documents. 
    
    ⚠️ **Warning**: These operations will permanently delete data and cannot be undone.
    """)
    
    # Get current stats
    try:
        response = requests.get(f"{API_BASE_URL}/stats")
        if response.status_code == 200:
            stats = response.json()
            
            # Display current status
            st.markdown("### 📊 Current System Status")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_docs = stats.get('vector_store', {}).get('total_documents', 0)
                st.info(f"📄 **Vector Store Documents:** {total_docs}")
            with col2:
                files_count = stats.get('pdf_storage', {}).get('files_count', 0)
                st.info(f"🗂 **PDF Files:** {files_count}")
            with col3:
                storage_size = stats.get('pdf_storage', {}).get('total_size_mb', 0)
                st.info(f"💾 **Storage Size:** {storage_size} MB")
                
        else:
            st.warning("⚠️ Could not fetch database statistics")
            stats = None
    except Exception as e:
        st.error(f"❌ Error fetching stats: {str(e)}")
        stats = None
    
    st.markdown("---")
    
    # Reset options
    st.markdown("### 🔧 Reset Options")
    
    # Option 1: Clear Vector Store Only
    with st.expander("🗂️ Clear Vector Store Only", expanded=False):
        st.markdown("""
        This will:
        - 🗂️ Delete all vector embeddings and document chunks
        - 🔄 Reset processing tracking (files will be reprocessed on next ingest)
        - 🧹 Clean up the vector database properly
        - ✅ **Keep PDF files** (you can re-ingest them later)
        """)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            confirm_vector = st.checkbox("I want to clear only the vector store", key="confirm_vector")
        with col2:
            if st.button("🗂️ Clear Vector Store", disabled=not confirm_vector, use_container_width=True):
                clear_vector_store_only()
    
    # Option 2: Clear PDF Storage Only  
    with st.expander("🗂 Clear PDF Storage Only", expanded=False):
        st.markdown("""
        This will:
        - 🗂 Delete all uploaded PDF files
        - 🔄 Reset processing tracking
        - ✅ **Keep vector embeddings** (until you clear them separately)
        """)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            confirm_pdf = st.checkbox("I want to clear only PDF files", key="confirm_pdf")
        with col2:
            if st.button("🗂 Clear PDF Files", disabled=not confirm_pdf, use_container_width=True):
                clear_pdf_storage_only()
    
    # Option 3: Complete System Reset
    with st.expander("🔥 Complete System Reset", expanded=False):
        st.markdown("""
        This will:
        - 🗂️ Delete all vector embeddings and document chunks
        - 🗂 Delete all uploaded PDF files  
        - 🔄 Reset all processing tracking
        - 🧹 Properly clean up the entire vector database directory
        - 📄 Reset the entire system to initial state
        """)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            confirm_complete = st.checkbox("I understand this will delete ALL data permanently", key="confirm_complete")
        with col2:
            if st.button("🔥 Complete Reset", disabled=not confirm_complete, use_container_width=True):
                clear_database()

    st.markdown('</div>', unsafe_allow_html=True)

def clear_vector_store_only():
    """Clear only the vector store"""
    with st.spinner("🗂️ Clearing vector store..."):
        try:
            response = requests.delete(f"{API_BASE_URL}/reset/vector-store")
            
            if response.status_code == 200:
                result = response.json()
                st.success("✅ Vector store cleared successfully!")
                
                with st.expander("📊 Details", expanded=True):
                    st.json(result)
                
                # Show metrics
                details = result.get("details", {})
                if details.get("status") == "success":
                    docs_cleared = details.get("documents_cleared", 0)
                    st.metric("📄 Documents Cleared", docs_cleared)
                
                st.info("🗂 Your PDF files are still available for re-ingestion.")
                st.rerun()
                
            else:
                st.error(f"❌ Error: HTTP {response.status_code}")
                
        except Exception as e:
            st.error(f"❌ Error clearing vector store: {str(e)}")

def clear_pdf_storage_only():
    """Clear only PDF storage"""
    with st.spinner("📁 Clearing PDF storage..."):
        try:
            response = requests.delete(f"{API_BASE_URL}/reset/pdf-storage")
            
            if response.status_code == 200:
                result = response.json()
                st.success("✅ PDF storage cleared successfully!")
                
                with st.expander("📊 Details", expanded=True):
                    st.json(result)
                
                # Show metrics
                details = result.get("details", {})
                if details.get("status") == "success":
                    files_cleared = details.get("files_count", 0)
                    st.metric("📁 Files Cleared", files_cleared)
                
                st.info("🗂️ Your vector embeddings are still available for querying.")
                st.rerun()
                
            else:
                st.error(f"❌ Error: HTTP {response.status_code}")
                
        except Exception as e:
            st.error(f"❌ Error clearing PDF storage: {str(e)}")
            
def clear_database():
    """Clear the vector database"""
    with st.spinner("🗑️ Clearing database..."):
        try:
            # Call the reset API endpoint
            response = requests.delete(f"{API_BASE_URL}/reset", data={"include_pdfs": True})
            
            if response.status_code == 200:
                result = response.json()
                st.success("✅ Database cleared successfully!")
                
                # Show detailed results
                with st.expander("📊 Reset Details", expanded=True):
                    st.json(result)
                
                # Show summary
                details = result.get("details", {})
                vector_result = details.get("vector_store_reset", {})
                pdf_result = details.get("pdf_storage_cleared", {})
                
                col1, col2 = st.columns(2)
                with col1:
                    if vector_result.get("status") == "success":
                        docs_cleared = vector_result.get("documents_cleared", 0)
                        st.metric("📄 Documents Cleared", docs_cleared)
                    else:
                        st.error("❌ Vector store reset failed")
                
                with col2:
                    if pdf_result.get("status") == "success":
                        files_cleared = pdf_result.get("files_count", 0)
                        st.metric("🗂️ PDF Files Cleared", files_cleared)
                    else:
                        st.error("❌ PDF storage reset failed")
                
                st.success("🧹 Vector database has been properly reset and reinitialized.")
                st.balloons()
                
                # Rerun to refresh stats after a short delay
                import time
                time.sleep(2)
                st.rerun()
                
            else:
                st.error(f"❌ Error clearing database: HTTP {response.status_code}")
                try:
                    error_detail = response.json()
                    st.json(error_detail)
                except:
                    st.text(response.text)
                
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Connection error: {str(e)}")
            st.info("Make sure the FastAPI server is running on http://localhost:8000")
            
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            st.markdown("""
            **Manual Database Clearing:**
            
            If the API is not working, you can manually clear the database:
            
            1. Stop the FastAPI server
            2. Delete the vector database and PDF files:
               ```bash
               rm -rf data/vector_db/*
               rm -rf data/pdfs/*
               ```
            3. Restart the FastAPI server
            """)
            
            
def help_page():
    st.markdown('<div class="card"><div class="card-header"><span class="card-icon">❔</span>Help & Instructions</div>', unsafe_allow_html=True)
    st.markdown("""
    # ResearchMind Usage Guide
    
    ## Quick Start
    
    1. **Upload**: Upload Research Papers → Choose PDFs → Process Documents
    2. **Ask**: Ask Questions → Type your question → Get Answer  
    3. **Summarize**: Research Summary → Generate Summary
    
    **Setup:** Run `python api/main.py` and wait for "Server running on http://localhost:8000"
    
    ---
    
    ## Core Features
    
    ### 1. Upload Research Papers
    1. Select PDF files
    2. Click "Process Documents"
    3. Wait for processing completion
    
    **Notes:**
    - Only PDF files supported
    - System tracks processed files to avoid reprocessing
    - Can add papers anytime without losing existing ones
    - Uses ChromaDB vector store for efficient similarity search
    
    ### 2. Ask Questions
    1. Type your question
    2. Adjust settings if needed
    3. Click "Get Answer"
    
    **Question Examples:**
    - "What are the main findings?"
    - "What methodology was used?"
    - "What are the limitations?"
    - "How do these papers define [concept]?"
    
    **Settings:**
    - **Chunks (1-15):** Use 3-5 for specific questions, 7-10 for broad ones
    - **Similarity Threshold:** Optional filtering - enable for more precise results, disable for broader coverage
    
    ### 3. Generate Summaries  
    1. Optional: Enter focus topic
    2. Adjust settings
    3. Click "Generate Summary"
    
    **Focus Examples:**
    - "methodology"
    - "results and findings" 
    - "limitations"
    - Leave blank for general overview
    
    **Settings:**
    - **Chunks (5-25):** Default 10, use 15-25 for comprehensive summaries
    - **Similarity Threshold:** Optional filtering - recommended for focused summaries
    
    ---
    
    ## Understanding Results
    
    ### Similarity Scores (ChromaDB with Cosine Similarity)
    **Higher Score = Higher Similarity**
    - **≥ 0.8:** Highly Similar (Green) - Excellent match to your query
    - **0.7 - 0.8:** Very Similar (Yellow) - Very good match
    - **0.5 - 0.7:** Moderately Similar (Orange) - Good match
    - **0.3 - 0.5:** Somewhat Similar (Red) - Fair match
    - **< 0.3:** Less Similar - May include some noise
    
    ### Threshold Filtering Options
    - **With Threshold:** Only results above the threshold are returned
    - **Without Threshold:** All top-k results are returned, ranked by similarity
    - **Recommended Thresholds:**
      - High Quality: 0.7
      - Good Quality: 0.5
      - Moderate Quality: 0.3
      - Low Quality: 0.2
    
    ### Result Quality Guidelines
    - **High Quality (≥ 0.7):** Precise, directly relevant results
    - **Good Quality (0.5-0.7):** Relevant with good coverage
    - **Fair Quality (0.3-0.5):** Broader coverage, may include tangential content
    - **No Threshold:** Maximum coverage, includes all ranked results
    
    ---
    
    ## Vector Store Technology
    
    ### ChromaDB with Cosine Similarity
    - **High Performance**: Optimized for large-scale similarity search
    - **Efficient Storage**: Persistent storage with automatic optimization
    - **Cosine Similarity**: Measures semantic similarity between concepts
    - **Flexible Filtering**: Optional threshold filtering for precision control
    - **Scalable**: Handles growing document collections efficiently
    
    ---
    
    ## Threshold vs No-Threshold Usage
    
    ### When to Use Thresholds
    - **Precise Queries**: When you need highly relevant results only
    - **Quality over Quantity**: When you prefer fewer, better matches
    - **Focused Analysis**: When analyzing specific topics or concepts
    - **Time Constraints**: When you need to review fewer results
    
    ### When to Skip Thresholds
    - **Exploratory Research**: When discovering connections and patterns
    - **Comprehensive Coverage**: When you need to see all available information
    - **Broad Topics**: When researching general or interdisciplinary subjects
    - **Small Collections**: When you have limited documents and need maximum coverage
    
    ### Best Practices
    - **Start Without Threshold**: See the full range of available results
    - **Apply Threshold Gradually**: Start with lower values (0.3-0.5) and increase as needed
    - **Monitor Result Counts**: Ensure you're not filtering out too much relevant content
    - **Context Matters**: Different queries may need different threshold strategies
    
    ---
    
    ## Troubleshooting
    
    ### "API Connection Error"
    1. Run `python api/main.py` 
    2. Wait for server startup message
    3. Refresh interface
    
    ### "No documents found"
    1. Check if documents are uploaded and processed
    2. Try disabling similarity threshold for broader results
    3. Use different keywords or phrasing
    4. Verify ChromaDB collection status in System Statistics
    5. Increase number of chunks to retrieve more results
    
    ### Poor Quality Results (Low Similarity Scores)
    1. Use more specific questions
    2. Enable similarity threshold to filter low-quality matches
    3. Check if your query matches the content of uploaded documents
    4. Try different phrasings or synonyms
    5. Consider re-processing documents if consistently poor results
    
    ### Too Few Results
    1. **Disable similarity threshold** for maximum coverage
    2. Lower the threshold value (try 0.3 or 0.2)
    3. Increase number of chunks
    4. Use broader, more general query terms
    5. Check that documents are properly ingested
    
    ### Too Many Irrelevant Results
    1. **Enable similarity threshold** (start with 0.5)
    2. Use more specific query terms
    3. Reduce number of chunks to get only top matches
    4. Increase threshold value for stricter filtering
    
    ### Slow Performance
    1. Reduce chunks to 3-5
    2. Enable threshold to reduce result processing
    3. Restart server if needed
    4. Check vector store status in Statistics
    5. Monitor system resources for large document collections
    
    ---
    
    ## System Management
    
    ### Data Cleanup Options
    - **Clear Vector Store**: Removes ChromaDB index, keeps PDFs
    - **Clear PDF Files**: Removes originals, keeps processed embeddings  
    - **Reset System**: Complete fresh start, rebuilds ChromaDB collection
    
    ### Best Practices
    - **Quality over Quantity**: Better documents = better results
    - **Experiment with Thresholds**: Find optimal settings for your use cases
    - **Monitor Similarity Scores**: Aim for average scores above 0.5 for good results
    - **Regular Cleanup**: Remove irrelevant documents
    - **Flexible Approach**: Use thresholds for precision, skip for exploration
    
    ---
    
    ## Quick Reference
    
    **Default Settings:**
    - Questions: 5 chunks, threshold optional
    - Summaries: 10 chunks, threshold optional
    
    **When to Adjust:**
    - Need more results → Increase chunks or disable threshold
    - Want only best matches → Enable threshold (0.5-0.7)
    - Need more context → Increase chunks
    - Want faster responses → Decrease chunks or enable threshold
    - Exploring broadly → Disable threshold
    - Focusing precisely → Enable higher threshold (0.6-0.8)
    
    **Technical Details:**
    - **Vector Store**: ChromaDB with Cosine similarity
    - **File Support**: PDF only, text-based (not scanned images)
    - **Processing**: Automatic hash-based tracking prevents reprocessing
    - **Storage**: Persistent ChromaDB collections with automatic optimization
    - **Flexibility**: Optional threshold filtering for different use cases
    
    **Threshold Strategy:**
    - **Exploration Phase**: Start without thresholds, see what's available
    - **Refinement Phase**: Apply thresholds to focus on quality matches
    - **Analysis Phase**: Use appropriate thresholds for your specific needs
    
    ---
    
    ## Advanced Features
    
    ### Flexible Search Options
    - **Threshold Mode**: Precision-focused with similarity filtering
    - **No-Threshold Mode**: Coverage-focused with all ranked results
    - **Adaptive Strategy**: Switch between modes based on query needs
    
    ### Source-Based Search
    - Search documents from specific sources/papers
    - Useful for focused analysis on particular studies
    - Available through the system's search capabilities
    
    ### Incremental Processing
    - Add new documents without reprocessing existing ones
    - Hash-based file tracking prevents duplicate work
    - Maintains system efficiency as collection grows
    
    ### Dynamic Result Interpretation
    Results adapt based on threshold settings:
    - **With Threshold**: Only shows results meeting quality criteria
    - **Without Threshold**: Shows all results with similarity rankings
    - **Clear Indicators**: UI shows whether filtering is active
    
    ---
    
    ## API Integration
    
    ### REST API Endpoints
    - `/upload`: Upload PDF files
    - `/ingest`: Process uploaded documents
    - `/ask`: Query with optional threshold parameter
    - `/summarize`: Generate summaries with optional threshold
    - `/stats`: Get system statistics including threshold recommendations
    
    ### Threshold Parameter Usage
    ```python
    # With threshold filtering
    {"question": "your query", "k": 5, "threshold": 0.6}
    
    # Without threshold filtering  
    {"question": "your query", "k": 5}  # threshold omitted
    ```
    
    ### Response Format
    ```json
    {
      "answer": "Generated response",
      "threshold_applied": true/false,
      "threshold_used": 0.6 or null,
      "sources": [...],
      "average_similarity": 0.75
    }
    ```
    
    ---
    
    ## Data Security & Privacy
    
    ### Local Processing
    - All documents processed locally on your machine
    - No data sent to external services
    - Complete control over your research materials
    
    ### File Management
    - Original PDFs stored in designated folder
    - Vector embeddings stored locally in ChromaDB format
    - Easy backup and restore capabilities
    
    ---
    
    ## Understanding Similarity-Based Search
    
    ### How It Works
    - Each document chunk is converted to a high-dimensional vector
    - Your query is also converted to a vector using the same method
    - ChromaDB calculates cosine similarity between query and document vectors
    - Higher similarity scores indicate better semantic matches
    
    ### Similarity Interpretation
    - **Perfect Match**: Score = 1.0 (rarely achieved in practice)
    - **Excellent**: Score ≥ 0.8
    - **Good**: Score 0.6-0.8
    - **Fair**: Score 0.4-0.6
    - **Poor**: Score < 0.4
    
    ### Advantages of Optional Thresholds
    - **Flexible Control**: Choose precision vs coverage based on needs
    - **Transparent Scoring**: Always see similarity scores for informed decisions
    - **Adaptive Usage**: Different strategies for different types of queries
    - **No Lost Information**: Can always see all results when needed
    
    ---
    
    ## Support & Updates
    
    ### Getting Help
    - Check troubleshooting section first
    - Experiment with threshold settings for different results
    - Monitor system statistics for performance issues
    - Restart API server if experiencing connection problems
    - Use similarity scores to assess result quality
    
    ### Optimization Tips
    - **For Precision**: Use higher thresholds (0.6-0.8)
    - **For Coverage**: Disable thresholds or use lower values (0.2-0.4)
    - **For Speed**: Enable thresholds to reduce processing
    - **For Exploration**: Start without thresholds, then refine
    
    ---
    
    **Happy Researching!** 🎓📚✨
    
    *Your AI-powered research assistant now offers flexible similarity-based search with optional threshold filtering. Choose the approach that best fits your research goals - precision when you need focused results, or comprehensive coverage when exploring broadly.*
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
if __name__ == "__main__":
    main()