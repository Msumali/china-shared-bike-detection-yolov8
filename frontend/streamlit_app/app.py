import streamlit as st
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import components
from components.video_upload import render_video_upload_component
from components.realtime_detection import render_realtime_detection_component
from components.batch_processing import render_batch_processing_component
from components.statistics import render_statistics_component, render_system_health
# from components.statistics import render_statistics_component
from config.settings import APP_CONFIG, SESSION_KEYS
from utils.ui_helpers import setup_page_config, render_sidebar

def main():
    """Main application function."""
    
    # Setup page configuration
    setup_page_config()
    
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Main content area
    render_main_content()
    
    # Footer
    render_footer()


def initialize_session_state():
    """Initialize session state variables."""
    
    # Initialize all session keys if they don't exist
    for key in SESSION_KEYS.values():
        if isinstance(key, str) and key not in st.session_state:
            st.session_state[key] = None
    
    # Initialize specific default values
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = 'Upload'
    
    if 'api_connected' not in st.session_state:
        st.session_state.api_connected = False


def render_main_content():
    """Render the main content area."""
    
    # App header
    st.title("🚴 Bike Detection System")
    st.markdown("Real-time bike detection using YOLOv8 deep learning model")
    
    # Connection status
    render_connection_status()
    
    # Navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📹 Upload", 
        "⚡ Real-time", 
        "🔄 Batch Processing", 
        "📊 Statistics"
    ])
    
    with tab1:
        render_video_upload_component()
    
    with tab2:
        render_realtime_detection_component()
    
    with tab3:
        render_batch_processing_component()
    
    with tab4:
        render_statistics_component()
        
        # Add system health section
        # st.divider()
        # render_system_health()


def render_connection_status():
    """Render API connection status."""
    
    from utils.api_client import APIClient
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🔌 Test Connection"):
            test_api_connection()
    
    with col2:
        # Connection status indicator
        if st.session_state.get('api_connected', False):
            st.success("✅ Connected to Backend")
        else:
            st.error("❌ Backend Disconnected")
    
    with col3:
        # Backend URL info
        backend_url = APP_CONFIG.get('BACKEND_URL', 'http://localhost:8000')
        st.info(f"Backend URL: {backend_url}")


def test_api_connection():
    """Test connection to the backend API."""
    
    try:
        from utils.api_client import APIClient
        
        api_client = APIClient()
        response = api_client.health_check()
        
        # Check if the response indicates healthy status
        if response.get('status') == 'healthy':
            st.session_state.api_connected = True
            st.success("Successfully connected to backend!")
        else:
            st.session_state.api_connected = False
            error_msg = response.get('error', 'Backend is not healthy')
            st.error(f"Backend connection failed: {error_msg}")
            
    except Exception as e:
        st.session_state.api_connected = False
        st.error(f"Failed to connect to backend: {str(e)}")
        
        # Additional debugging info
        with st.expander("Debug Information"):
            st.write("**Error Details:**")
            st.code(str(e))
            st.write("**Backend URL:**")
            from config.settings import API_BASE_URL
            st.code(API_BASE_URL)


def render_footer():
    """Render application footer."""
    
    st.divider()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(
            """
            <div style='text-align: center; color: #666; font-size: 14px;'>
                🚴 Bike Detection System v1.0<br>
                Powered by YOLOv8 & Streamlit<br>
                <a href='#' style='color: #666;'>Documentation</a> | 
                <a href='#' style='color: #666;'>Support</a> | 
                <a href='#' style='color: #666;'>GitHub</a>
            </div>
            """,
            unsafe_allow_html=True
        )


def render_debug_info():
    """Render debug information (only in development)."""
    
    if APP_CONFIG.get('DEBUG', False):
        with st.expander("🐛 Debug Information"):
            st.write("**Session State:**")
            st.json(dict(st.session_state))
            
            st.write("**App Configuration:**")
            st.json(APP_CONFIG)
            
            st.write("**Environment Variables:**")
            env_vars = {
                'BACKEND_URL': os.getenv('BACKEND_URL', 'Not set'),
                'API_KEY': '***' if os.getenv('API_KEY') else 'Not set',
                'DEBUG': os.getenv('DEBUG', 'False')
            }
            st.json(env_vars)


def handle_errors():
    """Global error handler for the application."""
    
    try:
        main()
    except Exception as e:
        st.error("An unexpected error occurred!")
        st.exception(e)
        
        if APP_CONFIG.get('DEBUG', False):
            st.write("**Full Error Details:**")
            import traceback
            st.code(traceback.format_exc())


if __name__ == "__main__":
    # Run with error handling in production
    if APP_CONFIG.get('DEBUG', False):
        main()
        render_debug_info()
    else:
        handle_errors()