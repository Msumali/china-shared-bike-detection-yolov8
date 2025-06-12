import streamlit as st
import time
from typing import Dict, Any, List
from ..utils.api_client import APIClient
from ..utils.ui_helpers import show_error_message, show_success_message, show_info_message
from ..config.settings import SESSION_KEYS
from .video_upload import get_current_file_id, get_current_video_info, has_uploaded_video

def render_batch_processing_component():
    """Render batch processing component."""
    
    st.header("⚡ Batch Processing")
    
    if not has_uploaded_video():
        show_info_message("Please upload a video first in the Upload tab.")
        return
    
    file_id = get_current_file_id()
    video_info = get_current_video_info()
    
    if not file_id or not video_info:
        show_error_message("No video information available.")
        return
    
    # Processing options
    st.subheader("🎯 Processing Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        processing_mode = st.selectbox(
            "Processing Mode",
            ["All Frames", "Every N Frames", "Time Interval", "Frame Range"],
            help="Choose how to process the video"
        )
    
    with col2:
        output_format = st.selectbox(
            "Output Format",
            ["JSON", "CSV", "Video with Annotations"],
            help="Choose output format for results"
        )
    
    # Mode-specific options
    render_processing_options(processing_mode, video_info)
    
    # Start processing button
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("🚀 Start Batch Processing", type="primary", use_container_width=True):
            start_batch_processing(file_id, video_info, processing_mode, output_format)
    
    with col2:
        if st.button("📊 View Results"):
            display_batch_results()


def render_processing_options(mode: str, video_info: Dict[str, Any]):
    """Render processing options based on selected mode."""
    
    total_frames = video_info.get('frame_count', 0)
    fps = video_info.get('fps', 30)
    duration = video_info.get('duration', 0)
    
    if mode == "Every N Frames":
        st.subheader("⚙️ Sampling Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            frame_interval = st.number_input(
                "Frame Interval",
                min_value=1,
                max_value=total_frames,
                value=30,
                help="Process every N frames"
            )
        
        with col2:
            estimated_frames = total_frames // frame_interval
            st.metric("Estimated Frames to Process", estimated_frames)
        
        st.session_state['frame_interval'] = frame_interval
    
    elif mode == "Time Interval":
        st.subheader("⏰ Time-based Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            time_interval = st.number_input(
                "Time Interval (seconds)",
                min_value=0.1,
                max_value=duration,
                value=1.0,
                step=0.1,
                help="Process frame every N seconds"
            )
        
        with col2:
            estimated_frames = int(duration / time_interval)
            st.metric("Estimated Frames to Process", estimated_frames)
        
        st.session_state['time_interval'] = time_interval
    
    elif mode == "Frame Range":
        st.subheader("🎯 Range Selection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            start_frame = st.number_input(
                "Start Frame",
                min_value=0,
                max_value=total_frames - 1,
                value=0
            )
        
        with col2:
            end_frame = st.number_input(
                "End Frame",
                min_value=start_frame,
                max_value=total_frames - 1,
                value=min(start_frame + 100, total_frames - 1)
            )