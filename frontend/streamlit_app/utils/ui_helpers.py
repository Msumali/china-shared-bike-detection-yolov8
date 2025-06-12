import streamlit as st
import cv2
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
from config.settings import BRAND_COLORS, CHART_HEIGHT, CHART_WIDTH, PAGE_TITLE, PAGE_ICON, LAYOUT


def display_video_info(video_info: Dict[str, Any]):
    """Display video information in a formatted way."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Duration", f"{video_info['duration']:.2f}s")
        st.metric("FPS", f"{video_info['fps']:.2f}")
    
    with col2:
        st.metric("Resolution", video_info['resolution'])
        st.metric("Frame Count", f"{video_info['frame_count']:,}")
    
    with col3:
        st.metric("Width", f"{video_info['width']}px")
        st.metric("Height", f"{video_info['height']}px")


def display_detection_results(detections: List[Dict[str, Any]], frame_number: int):
    """Display detection results for a frame."""
    if not detections:
        st.info(f"No detections found in frame {frame_number}")
        return
    
    st.subheader(f"🎯 Detections in Frame {frame_number}")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_detections = len(detections)
    brands = {}
    avg_confidence = 0
    
    for detection in detections:
        brand = detection.get('brand', 'Unknown')
        brands[brand] = brands.get(brand, 0) + 1
        avg_confidence += detection.get('confidence', 0)
    
    avg_confidence = avg_confidence / total_detections if total_detections > 0 else 0
    
    with col1:
        st.metric("Total Detections", total_detections)
    
    with col2:
        st.metric("Avg Confidence", f"{avg_confidence:.2%}")
    
    with col3:
        most_common_brand = max(brands.items(), key=lambda x: x[1])[0] if brands else "None"
        st.metric("Most Common", most_common_brand)
    
    with col4:
        st.metric("Unique Brands", len(brands))
    
    # Detailed detection table
    with st.expander("📋 Detailed Detection Results", expanded=False):
        detection_data = []
        for i, detection in enumerate(detections):
            detection_data.append({
                'ID': i + 1,
                'Brand': detection.get('brand', 'Unknown'),
                'Confidence': f"{detection.get('confidence', 0):.2%}",
                'BBox': f"{detection.get('bbox', [0,0,0,0])}",
                'Area': detection.get('area', 0),
                'Center': f"{detection.get('center', [0,0])}"
            })
        
        st.dataframe(detection_data, use_container_width=True)


def create_brand_distribution_chart(brand_stats: Dict[str, int]) -> go.Figure:
    """Create a pie chart for brand distribution."""
    if not brand_stats or sum(brand_stats.values()) == 0:
        return None
    
    brands = list(brand_stats.keys())
    counts = list(brand_stats.values())
    colors = [BRAND_COLORS.get(brand, '#808080') for brand in brands]
    
    fig = go.Figure(data=[go.Pie(
        labels=brands,
        values=counts,
        marker=dict(colors=colors),
        hole=0.3,
        textinfo='label+percent+value'
    )])
    
    fig.update_layout(
        title="Brand Distribution",
        height=CHART_HEIGHT,
        showlegend=True
    )
    
    return fig


def create_confidence_histogram(detections: List[Dict[str, Any]]) -> go.Figure:
    """Create a histogram of confidence scores."""
    if not detections:
        return None
    
    confidences = [d.get('confidence', 0) for d in detections]
    
    fig = go.Figure(data=[go.Histogram(
        x=confidences,
        nbinsx=20,
        marker_color='skyblue',
        opacity=0.7
    )])
    
    fig.update_layout(
        title="Confidence Score Distribution",
        xaxis_title="Confidence Score",
        yaxis_title="Count",
        height=CHART_HEIGHT
    )
    
    return fig


def display_processing_progress(progress: float, status: str = "Processing"):
    """Display processing progress bar."""
    progress_bar = st.progress(progress / 100)
    st.text(f"{status}: {progress:.1f}% complete")
    return progress_bar


def display_frame_with_detections(frame: np.ndarray, detections: List[Dict[str, Any]]):
    """Display frame with detection annotations."""
    if frame is None:
        st.error("Cannot display frame")
        return
    
    # Convert BGR to RGB for display
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        frame_rgb = frame
    
    # Convert to PIL Image for display
    pil_image = Image.fromarray(frame_rgb)
    
    # Display the frame
    st.image(pil_image, caption=f"Frame with {len(detections)} detections", use_column_width=True)


def create_detection_timeline(processing_results: List[Dict[str, Any]]) -> go.Figure:
    """Create a timeline chart of detections over frames."""
    if not processing_results:
        return None
    
    frame_numbers = []
    detection_counts = []
    
    for result in processing_results:
        frame_numbers.append(result.get('frame_number', 0))
        detection_counts.append(len(result.get('detections', [])))
    
    fig = go.Figure(data=[go.Scatter(
        x=frame_numbers,
        y=detection_counts,
        mode='lines+markers',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=6)
    )])
    
    fig.update_layout(
        title="Detection Count Over Time",
        xaxis_title="Frame Number",
        yaxis_title="Number of Detections",
        height=CHART_HEIGHT
    )
    
    return fig


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"


def show_success_message(message: str, duration: int = 3):
    """Show success message."""
    success_placeholder = st.empty()
    success_placeholder.success(message)
    import time
    time.sleep(duration)
    success_placeholder.empty()


def show_error_message(message: str):
    """Show error message."""
    st.error(f"❌ {message}")


def show_info_message(message: str):
    """Show info message."""
    st.info(f"ℹ️ {message}")


def create_sidebar_controls() -> Dict[str, Any]:
    """Create sidebar controls and return their values."""
    st.sidebar.title("🔧 Detection Settings")
    
    # Detection parameters
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence score for detections"
    )
    
    iou_threshold = st.sidebar.slider(
        "IoU Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.45,
        step=0.05,
        help="Intersection over Union threshold for NMS"
    )
    
    frame_skip = st.sidebar.number_input(
        "Frame Skip",
        min_value=1,
        max_value=10,
        value=1,
        help="Process every Nth frame (1 = process all frames)"
    )
    
    save_annotated = st.sidebar.checkbox(
        "Save Annotated Video",
        value=True,
        help="Save video with detection annotations"
    )
    
    return {
        'confidence_threshold': confidence_threshold,
        'iou_threshold': iou_threshold,
        'frame_skip': frame_skip,
        'save_annotated_video': save_annotated
    }

# Add these functions to your ui_helpers.py file

def setup_page_config():
    """Setup Streamlit page configuration."""
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout=LAYOUT,
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': f"# {PAGE_TITLE}\n\nBike Detection System using YOLOv8"
        }
    )


def render_sidebar():
    """Render the application sidebar with navigation and controls."""
    
    with st.sidebar:
        # App logo/title
        st.markdown(
            """
            <div style='text-align: center; padding: 20px 0;'>
                <h1 style='color: #1f77b4; margin: 0;'>🚴</h1>
                <h3 style='color: #333; margin: 5px 0;'>Bike Detector</h3>
                <p style='color: #666; font-size: 12px; margin: 0;'>YOLOv8 Detection System</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.divider()
        
        # Navigation info
        st.markdown("### 🧭 Navigation")
        st.markdown("""
        - **📹 Upload**: Upload videos for analysis
        - **⚡ Real-time**: Live detection from camera
        - **🔄 Batch**: Process multiple files
        - **📊 Statistics**: View analytics
        """)
        
        st.divider()
        
        # Detection settings (reuse existing function)
        detection_settings = create_sidebar_controls()
        
        st.divider()
        
        # System info
        st.markdown("### ℹ️ System Info")
        
        # Memory usage (if available)
        try:
            import psutil
            memory = psutil.virtual_memory()
            st.metric("Memory Usage", f"{memory.percent:.1f}%")
        except ImportError:
            st.info("Install psutil for memory monitoring")
        
        # App version
        st.text("Version: 1.0.0")
        st.text("Model: YOLOv8")
        
        return detection_settings