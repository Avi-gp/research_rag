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
        **ResearchMind** leverages advanced AI to help you:
        - 📚 Analyze research papers
        - 🔍 Extract key insights  
        - 💡 Answer complex questions
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
    
    # Advanced settings with threshold support
    with st.expander("🔧 Advanced Search Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            k = st.slider("📊 Relevant Chunks", 1, 15, 5, help="Number of document chunks to analyze")
        with col2:
            threshold = st.slider(
                "🎯 Similarity Threshold", 
                0.0, 1.0, 0.5, 0.1,
                help="Minimum similarity score for relevant documents (higher = more strict)"
            )
    
    if st.button("🔍 Get Answer", type="primary", key="get_answer_btn"):
        if question.strip():
            get_answer(question, k, threshold)
        else:
            st.warning("⚠️ Please enter a question first!")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    
def get_answer(question, k, threshold=0.5):
    """Get answer for the question using similarity search with threshold"""
    with st.spinner("🧠 Analyzing documents and generating answer..."):
        try:
            data = {
                "question": question, 
                "k": k,
                "threshold": threshold
            }
            
            response = requests.post(f"{API_BASE_URL}/ask", data=data)
            
            if response.status_code == 200:
                result = response.json()
                
                # Answer section with better styling
                st.markdown("### 🤖 Answer")
                st.markdown(f"""
                <div class="answer-section">
                    {result.get('answer', 'No answer generated')}
                </div>
                """, unsafe_allow_html=True)
                
                # Search info with threshold
                st.info(f"🔍 Search: Similarity | 📊 Chunks: {k} | 🎯 Threshold: {threshold}")
                
                # Sources section with similarity scores
                if result.get('sources'):
                    st.markdown("### 📚 Sources")
                    for i, source in enumerate(result['sources'], 1):
                        similarity_score = source.get('similarity_score', 0)
                        rank = source.get('rank', i)
                        
                        # Color code similarity scores
                        if similarity_score >= 0.8:
                            score_color = "#0c6b22"  # Green
                        elif similarity_score >= 0.6:
                            score_color = "#91700e"  # Yellow
                        else:
                            score_color = "#4e050c"  # Red
                        
                        st.markdown(f"""
                        <div class="source-item">
                            <div class="source-header">📄 Source {i}: {source.get('source', 'Unknown')}</div>
                            <div class="source-score" style="color: {score_color};">
                            Score: {similarity_score:.4f} | Rank: {rank}
                            </div>
                            <div><strong>Chunk ID:</strong> {source.get('chunk_id', 'N/A')}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning(f"⚠️ No sources found above similarity threshold of {threshold}. Try lowering the threshold.")
                
                # Analytics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📊 Chunks Used", result.get('context_used', 0))
                with col2:
                    st.metric("📄 Sources Found", len(result.get('sources', [])))
                with col3:
                    if result.get('sources'):
                        avg_similarity = sum(s.get('similarity_score', 0) for s in result['sources']) / len(result['sources'])
                        st.metric("📈 Avg Similarity", f"{avg_similarity:.3f}")
                    else:
                        st.metric("📈 Avg Similarity", "N/A")
                with col4:
                    st.metric("🎯 Threshold Used", f"{result.get('threshold_used', threshold):.2f}")
                    
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
    
    # Advanced settings with threshold support
    with st.expander("🔧 Advanced Summary Settings", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            k = st.slider("📊 Document Chunks", 5, 25, 15, help="Number of document chunks to include in summary")
        with col2:
            summary_type = st.selectbox(
                "📋 Summary Type",
                ["Comprehensive", "Executive Summary", "Key Findings", "Methodology Focus"],
                help="Choose the type of summary you want"
            )
        with col3:
            threshold = st.slider(
                "🎯 Similarity Threshold", 
                0.0, 1.0, 0.4, 0.1,
                help="Minimum similarity score (lower for broader summaries)"
            )
    
    if st.button("📋 Generate Summary", type="primary"):
        generate_summary(query, k, summary_type, threshold)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    
def generate_summary(query, k, summary_type, threshold=0.4):
    """Generate summary of documents using similarity search with threshold"""
    with st.spinner("🔍 Analyzing documents and generating summary..."):
        try:
            data = {
                "k": k,
                "threshold": threshold
            }
            
            # Combine query with summary type
            if query:
                data["query"] = f"{summary_type}: {query}" if query else summary_type
            else:
                data["query"] = summary_type
            
            response = requests.post(f"{API_BASE_URL}/summarize", data=data)
            
            if response.status_code == 200:
                result = response.json()
                
                # Summary section with better visibility
                st.markdown(f"### 📄 {summary_type}")
                st.markdown(f"""
                <div class="summary-card">
                    {result.get('summary', 'No summary generated')}
                </div>
                """, unsafe_allow_html=True)
                
                # Search info with threshold
                st.info(f"🔍 Search: Similarity | 📊 Chunks: {k} | 🎯 Threshold: {threshold}")
                
                # Sources section with similarity information
                if result.get('sources'):
                    st.markdown("### 📚 Sources Analyzed")
                    
                    # Display sources in a more detailed format
                    for source in result['sources']:
                        if isinstance(source, dict):
                            source_name = source.get('source', 'Unknown')
                            chunks_used = source.get('chunks_used', 0)
                            avg_similarity = source.get('avg_similarity', 0.0)
                            max_similarity = source.get('max_similarity', 0.0)
                            
                            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                            with col1:
                                st.write(f"📄 {source_name}")
                            with col2:
                                st.write(f"📊 {chunks_used} chunks")
                            with col3:
                                st.write(f"📈 Avg: {avg_similarity:.3f}")
                            with col4:
                                st.write(f"🎯 Max: {max_similarity:.3f}")
                        else:
                            # Fallback for simple source format
                            source_name = str(source)
                            st.write(f"📄 {source_name}")
                else:
                    st.warning(f"⚠️ No sources found above similarity threshold of {threshold}. Try lowering the threshold.")
                
                # Analytics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📊 Chunks Analyzed", result.get('context_used', 0))
                with col2:
                    st.metric("📄 Sources Used", len(result.get('sources', [])))
                with col3:
                    st.metric("🔍 Search Method", "Similarity")
                with col4:
                    st.metric("🎯 Threshold Used", f"{result.get('threshold_used', threshold):.2f}")
                    
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
                index_exists = stats.get('vector_store', {}).get('index_exists', False)
                index_status = "✅ Active" if index_exists else "❌ Inactive"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="font-size: 1.5rem;">{index_status}</div>
                    <div class="metric-label">🗂️ Vector Index</div>
                </div>
                """, unsafe_allow_html=True)
            
            # System capabilities section
            st.markdown("### 🔧 System Capabilities")
            search_capabilities = stats.get('search_capabilities', {})
            
            col1, col2 = st.columns(2)
            with col1:
                search_method = search_capabilities.get('search_method', 'similarity_score_threshold')
                default_threshold = search_capabilities.get('default_threshold', 0.5)
                st.info(f"""
                **🔍 Search Configuration:**
                - Method: {search_method.upper()}
                - Default Threshold: {default_threshold}
                - LangChain Retriever: {'✅' if search_capabilities.get('langchain_retriever_support') else '❌'}
                - Source-based Search: {'✅' if search_capabilities.get('source_based_search') else '❌'}
                """)
            
            with col2:
                st.info(f"""
                **📊 Analytics Features:**
                - Similarity Analysis: {'✅' if search_capabilities.get('similarity_analysis') else '❌'}
                - Incremental Processing: {'✅' if search_capabilities.get('incremental_processing') else '❌'}
                - Threshold Filtering: ✅
                - Embedding Dimension: {stats.get('vector_store', {}).get('embedding_dimension', 'N/A')}
                """)
            
            # Storage information
            st.markdown("### 💾 Storage Information")
            col1, col2 = st.columns(2)
            
            with col1:
                storage_size = stats.get('pdf_storage', {}).get('total_size_mb', 0)
                st.info(f"📁 **Total Storage:** {storage_size} MB")
            
            with col2:
                storage_path = stats.get('pdf_storage', {}).get('storage_path', 'N/A')
                st.info(f"📍 **Storage Path:** {storage_path}")
            
            # PDF Processor information
            pdf_processor = stats.get('pdf_processor', {})
            if pdf_processor:
                st.markdown("### 🔄 PDF Processing")
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"📊 **Processed Files:** {pdf_processor.get('processed_files_count', 0)}")
                with col2:
                    st.info(f"📋 **Total Chunks:** {pdf_processor.get('total_chunks_processed', 0)}")
            
            # Files breakdown
            processed_files_info = pdf_processor.get('processed_files_info', {})
            if processed_files_info:
                st.markdown("### 📂 Files Breakdown")
                for filename, info in processed_files_info.items():
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.write(f"📄 {filename}")
                    with col2:
                        st.write(f"📊 {info.get('chunks', 0)} chunks")
                    with col3:
                        if 'last_modified' in info:
                            st.write(f"⏰ {info['last_modified']}")
            
            # Search method info - updated for threshold-based search
            st.markdown("### 🔍 Search Method Details")
            search_method = stats.get('search_capabilities', {}).get('search_method', 'similarity_score_threshold')
            default_threshold = search_capabilities.get('default_threshold', 0.5)
            
            st.markdown(f"""
            - **Search Method**: {search_method.upper()}
            - **Description**: Cosine similarity-based document retrieval with configurable threshold filtering
            - **Default Threshold**: {default_threshold} (documents below this similarity score are filtered out)
            - **Threshold Range**: 0.0 to 1.0 (higher values = more strict filtering)
            - **Benefits**: Reduces noise by filtering out low-relevance documents
            """)
            
            # Detailed stats
            st.markdown("### 🔍 Detailed Information")
            with st.expander("📋 Raw Statistics", expanded=False):
                st.json(stats)
                
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
        - 🗂️ Delete all vector embeddings
        - 📄 Remove all document chunks from memory
        - 📄 Reset the vector index
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
        - 🗂️ Delete all vector embeddings  
        - 📄 Remove all document chunks
        - 🗂 Delete all uploaded PDF files
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
    ## 🧠 ResearchMind Usage Guide
    
    Welcome to ResearchMind! This comprehensive guide will help you get the most out of the platform.
    
    ---
    
    ## 🚀 Getting Started
    
    ### Prerequisites
    - Ensure the FastAPI server is running (`python api/main.py`)
    - Have your research papers ready in PDF format
    - Check that you have a stable internet connection
    
    ---
    
    ## 📚 Step-by-Step Instructions
    
    ### 1. 📤 Upload Research Papers
    **What it does:** Ingests and processes your PDF research papers for analysis.
    
    **Steps:**
    1. Navigate to the **"Upload Research Papers"** tab in the sidebar
    2. Click the **"Choose PDF files"** button
    3. Select one or multiple PDF files (supports batch upload)
    4. Click **"Process Documents"** and wait for processing to complete
    5. You'll see a success message when documents are ready
    
    **📝 Tips:**
    - Supported formats: PDF only
    - File size limit: Check with your system administrator
    - Processing time varies with document length and complexity
    - You can upload multiple papers at once for batch processing
    
    ---
    
    ### 2. ❓ Ask Questions
    **What it does:** Get AI-powered answers from your uploaded research papers with source citations and similarity-based filtering.
    
    **Steps:**
    1. Go to the **"Ask Questions"** tab
    2. Type your question in the text area
    3. Adjust settings in **"Advanced Search Settings"**:
       - **Relevant chunks:** Number of document sections to analyze (1-15, default: 5)
       - **Similarity Threshold:** Minimum relevance score (0.0-1.0, default: 0.5)
    4. Click **"Get Answer"** 
    5. Review the answer with source references and similarity scores
    
    **📝 Question Examples:**
    - "What are the main findings of this research?"
    - "What methodology was used in the study?"
    - "What are the limitations mentioned by the authors?"
    - "How does this research contribute to the field?"
    - "What future research directions are suggested?"
    
    **🎯 Understanding Similarity Threshold:**
    - **Higher threshold (0.7-0.9):** More precise, fewer but highly relevant results
    - **Medium threshold (0.5-0.7):** Balanced relevance and coverage (recommended)
    - **Lower threshold (0.3-0.5):** Broader search, more results but potentially less relevant
    - **Very low threshold (0.0-0.3):** Maximum coverage, may include tangentially related content
    
    **⚙️ Advanced Options:**
    - **Relevant chunks:** Higher numbers (7-10) for complex questions, lower (3-5) for specific queries
    - **Threshold tuning:** If no results, try lowering the threshold; if too many irrelevant results, raise it
    - The system shows similarity scores for each source to help you understand relevance
    
    ---
    
    ### 3. 📊 Generate Research Summaries
    **What it does:** Creates comprehensive overviews of your research papers using similarity-based document selection.
    
    **Steps:**
    1. Navigate to **"Research Summary"** tab
    2. (Optional) Enter a **focus topic** to narrow the summary scope
    3. Configure **"Advanced Summary Settings"**:
       - **Document chunks:** Amount of content to include (5-25, default: 15)
       - **Summary type:** Choose focus area
       - **Similarity Threshold:** Relevance filter (0.0-1.0, default: 0.4)
    4. Click **"Generate Summary"**
    
    **📝 Focus Topic Examples:**
    - "methodology and experimental design"
    - "results and key findings"
    - "limitations and future work"
    - "literature review and background"
    - Leave blank for a general summary
    
    **🎯 Summary Threshold Guidelines:**
    - **Default (0.4):** Good balance for comprehensive summaries
    - **Higher (0.5-0.7):** More focused summaries with highly relevant content
    - **Lower (0.2-0.4):** Broader summaries including more diverse content
    
    ---
    
    ### 4. 📈 View System Statistics
    **What it does:** Shows system health, usage information, and similarity analysis tools.
    
    **Information available:**
    - Number of processed documents and vector embeddings
    - Total storage usage and embedding dimensions
    - Search method configuration (similarity threshold-based)
    - System performance metrics
    - API connection status
     
    ---
    
    ### 5. 🗂️ Database Management
    **What it does:** Allows you to manage and clean your data.
    
    **⚠️ IMPORTANT:** All database management actions are **irreversible**. Make sure you want to proceed.
    
    **Options:**
    - **Clear Vector Store Only:** Removes processed embeddings but keeps original PDFs
    - **Clear PDF Files Only:** Removes original files but keeps processed embeddings
    - **Reset Entire System:** Completely clears all data and starts fresh
    
    **When to use:**
    - Vector store clearing: When you want to reprocess documents with updated settings
    - PDF clearing: When you need storage space but want to keep processed data
    - Full reset: When starting a new research project
    
    ---
    
    ## 🔧 Troubleshooting
    
    ### Common Issues and Solutions
    
    **❌ "API Connection Error"**
    - **Solution:** Start the FastAPI server:
      ```bash
      cd your-project-directory
      python api/main.py
      ```
    - Wait for "Server running on http://localhost:8000" message
    
    **❌ "No documents found" or "No sources found above similarity threshold"**
    - **Solution 1:** Upload and process documents first in the Upload tab
    - **Solution 2:** Lower the similarity threshold (try 0.3 or 0.2)
    - **Solution 3:** Rephrase your question with different keywords
    - **Solution 4:** Use the similarity analysis tool to understand your document collection
    
    **❌ "Processing failed"**
    - **Solution:** 
      - Check PDF file isn't corrupted
      - Ensure PDF contains readable text (not just images)
      - Try uploading files one at a time
    
    **❌ "Slow responses"**
    - **Solution:**
      - Reduce number of relevant chunks
      - Increase similarity threshold to filter results
      - Check your internet connection
      - Restart the FastAPI server
    
    **❌ "Results not relevant enough"**
    - **Solution:**
      - Increase similarity threshold (try 0.6-0.8)
      - Rephrase question with more specific terms
      - Use fewer chunks to focus on top matches
    
    **❌ "Too few results"**
    - **Solution:**
      - Decrease similarity threshold (try 0.3-0.4)
      - Increase number of chunks
      - Use broader question terms
      - Check similarity analysis to understand score distribution
    
    ### Manual Database Clearing
    If the interface options don't work, you can manually clear the database:
    
    1. **Stop the server** (Ctrl+C in terminal)
    2. **Delete files:**
       ```bash
       # Clear vector database
       rm -rf data/vector_db/*
       
       # Clear PDF files
       rm -rf data/pdfs/*
       
       # Clear everything
       rm -rf data/*
       ```
    3. **Restart the server:** `python api/main.py`
    
    ---
    
    ## 💡 Best Practices & Tips
    
    ### For Better Results:
    - **Specific Questions:** Ask targeted questions rather than very broad ones
    - **Context Matters:** Include context in your questions (e.g., "In the machine learning papers...")
    - **Multiple Angles:** Try rephrasing questions if you don't get the desired answer
    - **Threshold Tuning:** Start with default thresholds, then adjust based on results
    - **Similarity Analysis:** Use the analysis tool to understand optimal thresholds for your documents
    
    ### Threshold Strategy:
    1. **Start with defaults:** 0.5 for questions, 0.4 for summaries
    2. **Too few results?** Lower threshold by 0.1-0.2
    3. **Too many irrelevant results?** Raise threshold by 0.1-0.2
    4. **Use analysis tool** to understand your document collection's similarity patterns
    
    ### Workflow Recommendations:
    1. **Start Small:** Upload 2-3 papers first to test the system
    2. **Test Thresholds:** Use similarity analysis to understand optimal settings
    3. **Explore:** Use the summary feature to get familiar with your papers
    4. **Dive Deep:** Ask specific questions with appropriate thresholds
    5. **Organize:** Use focused summaries to organize information by themes
    
    ### Performance Optimization:
    - **Regular Cleanup:** Clear unused documents to maintain performance
    - **Batch Processing:** Upload related papers together for better context
    - **Strategic Questioning:** Start with broad questions, then narrow down
    - **Threshold Optimization:** Find the sweet spot for your specific document collection
    
    ---
    
    ## 🎯 Understanding Similarity Scores
    
    The system uses cosine similarity to measure relevance between your questions and document content:
    
    - **0.9-1.0:** Extremely relevant (rare, usually exact matches)
    - **0.8-0.9:** Highly relevant (excellent matches)
    - **0.7-0.8:** Very relevant (good matches)
    - **0.6-0.7:** Moderately relevant (decent matches)
    - **0.5-0.6:** Somewhat relevant (borderline useful)
    - **0.3-0.5:** Loosely relevant (may contain useful context)
    - **0.0-0.3:** Minimally relevant (likely noise)
    
    **Color Coding in Results:**
    - 🟢 **Green (≥0.8):** High relevance
    - 🟡 **Yellow (0.6-0.8):** Medium relevance  
    - 🔴 **Red (<0.6):** Low relevance
    
    ---
    
    ## 🆘 Need More Help?
    
    - **Check System Statistics** for current system status and similarity analysis
    - **Try the troubleshooting steps** above for common issues
    - **Experiment with threshold settings** using the analysis tool
    - **Restart the system** if problems persist
    - **Review this help page** anytime by clicking the Help tab
    
    ---
    
    *Happy researching! 🎓*
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()