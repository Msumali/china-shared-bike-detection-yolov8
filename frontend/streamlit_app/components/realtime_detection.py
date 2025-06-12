import streamlit as st
import cv2
import numpy as np
from typing import Dict, Any, Optional
import time
import json
from utils.api_client import APIClient
from utils.ui_helpers import (
    display_detection_results, 
    display_frame_with_detections,
    show_error_message,
    show_info_message,
    create_sidebar_controls
)
from config.settings import SESSION_KEYS
from .video_upload import get_current_file_id, get_current_video_info, has_uploaded_video


def render_realtime_detection_component():
    """Render real-time detection component."""
    
    st.header("🎥 Real-time Detection")
    
    if not has_uploaded_video():
        show_info_message("Please upload a video first in the Upload tab.")
        return
    
    file_id = get_current_file_id()
    video_info = get_current_video_info()
    
    if not file_id or not video_info:
        show_error_message("No video information available.")
        return
    
    # Frame selection
    st.subheader("🎯 Frame Selection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        frame_number = st.number_input(
            "Frame Number",
            min_value=0,
            max_value=video_info['frame_count'] - 1,
            value=st.session_state.get(SESSION_KEYS['current_frame'], 0),
            help="Select frame to process"
        )
    
    with col2:
        timestamp = frame_number / video_info['fps'] if video_info['fps'] > 0 else 0
        st.metric("Timestamp", f"{timestamp:.2f}s")
    
    # Frame navigation
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("⏮️ First Frame"):
            frame_number = 0
            st.session_state[SESSION_KEYS['current_frame']] = frame_number
            st.rerun()
    
    with col2:
        if st.button("⏪ Previous"):
            frame_number = max(0, frame_number - 1)
            st.session_state[SESSION_KEYS['current_frame']] = frame_number
            st.rerun()
    
    with col3:
        if st.button("⏩ Next"):
            frame_number = min(video_info['frame_count'] - 1, frame_number + 1)
            st.session_state[SESSION_KEYS['current_frame']] = frame_number
            st.rerun()
    
    with col4:
        if st.button("⏭️ Last Frame"):
            frame_number = video_info['frame_count'] - 1
            st.session_state[SESSION_KEYS['current_frame']] = frame_number
            st.rerun()
    
    # Process frame button
    if st.button("🔍 Process Frame", type="primary", use_container_width=True):
        process_single_frame(file_id, frame_number)
    
    # Auto-play controls
    st.subheader("▶️ Auto-play Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        auto_play = st.checkbox("Auto-play", value=False)
    
    with col2:
        playback_speed = st.slider("Speed (FPS)", min_value=1, max_value=30, value=5)
    
    with col3:
        if st.button("🎬 Start Auto-play"):
            if auto_play:
                start_auto_play(file_id, video_info, frame_number, playback_speed)
    
    # Display current frame results
    if SESSION_KEYS['detection_results'] in st.session_state:
        results = st.session_state[SESSION_KEYS['detection_results']]
        if results and results.get('frame_number') == frame_number:
            display_frame_results(results)


def process_single_frame(file_id: str, frame_number: int):
    """Process a single frame and display results."""
    
    with st.spinner(f"Processing frame {frame_number}..."):
        try:
            api_client = APIClient()
            
            # Process frame
            result = api_client.process_frame(file_id, frame_number)
            
            # Store results
            st.session_state[SESSION_KEYS['detection_results']] = result
            st.session_state[SESSION_KEYS['current_frame']] = frame_number
            
            # Display results
            display_frame_results(result)
            
        except Exception as e:
            show_error_message(f"Failed to process frame: {str(e)}")


def display_frame_results(results: Dict[str, Any]):
    """Display frame processing results."""
    
    frame_number = results.get('frame_number', 0)
    detections = results.get('detections', [])
    processing_time = results.get('processing_time', 0)
    frame_image = results.get('annotated_frame')
    
    # Results summary
    st.subheader(f"📊 Frame {frame_number} Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Detections", len(detections))
    
    with col2:
        st.metric("Processing Time", f"{processing_time:.3f}s")
    
    with col3:
        if detections:
            avg_confidence = sum(d.get('confidence', 0) for d in detections) / len(detections)
            st.metric("Avg Confidence", f"{avg_confidence:.2%}")
    
    # Display annotated frame if available
    if frame_image:
        st.subheader("🖼️ Annotated Frame")
        st.image(frame_image, caption=f"Frame {frame_number} with detections", use_column_width=True)
    
    # Display detections
    if detections:
        display_detection_results(detections, frame_number)
        
        # Show detection details in expandable section
        with st.expander("🔍 Detection Details", expanded=False):
            for i, detection in enumerate(detections):
                st.write(f"**Detection {i+1}:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"Class: {detection.get('class_name', 'Unknown')}")
                    st.write(f"Confidence: {detection.get('confidence', 0):.2%}")
                with col2:
                    bbox = detection.get('bbox', {})
                    st.write(f"Position: ({bbox.get('x', 0):.0f}, {bbox.get('y', 0):.0f})")
                    st.write(f"Size: {bbox.get('width', 0):.0f}x{bbox.get('height', 0):.0f}")
                st.divider()
    else:
        show_info_message(f"No detections found in frame {frame_number}")


def start_auto_play(file_id: str, video_info: Dict[str, Any], start_frame: int, fps: int):
    """Start auto-play processing."""
    
    st.subheader("🎬 Auto-play Processing")
    
    # Create placeholders for dynamic updates
    progress_placeholder = st.empty()
    results_placeholder = st.empty()
    stop_placeholder = st.empty()
    
    frame_delay = 1.0 / fps
    current_frame = start_frame
    max_frame = video_info['frame_count'] - 1
    
    # Stop button
    stop_auto_play = stop_placeholder.button("⏹️ Stop Auto-play", key="stop_autoplay")
    
    try:
        api_client = APIClient()
        
        for frame in range(current_frame, max_frame + 1):
            if stop_auto_play:
                break
                
            # Update progress
            progress = (frame / max_frame) * 100
            progress_placeholder.progress(progress / 100, text=f"Processing frame {frame}/{max_frame}")
            
            # Process frame
            result = api_client.process_frame(file_id, frame)
            
            # Display results
            with results_placeholder.container():
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Frame", frame)
                with col2:
                    st.metric("Detections", len(result.get('detections', [])))
                with col3:
                    st.metric("Processing Time", f"{result.get('processing_time', 0):.3f}s")
                
                if result.get('detections'):
                    st.write("**Recent Detections:**")
                    for i, detection in enumerate(result['detections'][:3]):
                        confidence = detection.get('confidence', 0)
                        class_name = detection.get('class_name', 'Unknown')
                        st.write(f"  • {class_name} ({confidence:.2%})")
            
            # Wait before next frame
            time.sleep(frame_delay)
            
            # Update session state
            st.session_state[SESSION_KEYS['current_frame']] = frame
            st.session_state[SESSION_KEYS['detection_results']] = result
            
    except Exception as e:
        show_error_message(f"Auto-play failed: {str(e)}")
    finally:
        progress_placeholder.empty()
        stop_placeholder.empty()


def render_frame_comparison():
    """Render frame comparison view."""
    
    if not has_uploaded_video():
        return
    
    st.subheader("📷 Frame Comparison")
    
    file_id = get_current_file_id()
    video_info = get_current_video_info()
    
    if not file_id or not video_info:
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        frame1 = st.number_input(
            "Frame 1",
            min_value=0,
            max_value=video_info['frame_count'] - 1,
            value=0,
            key="frame1"
        )
    
    with col2:
        frame2 = st.number_input(
            "Frame 2",
            min_value=0,
            max_value=video_info['frame_count'] - 1,
            value=min(100, video_info['frame_count'] - 1),
            key="frame2"
        )
    
    if st.button("🔍 Compare Frames"):
        compare_frames(file_id, frame1, frame2)


def compare_frames(file_id: str, frame1: int, frame2: int):
    """Compare two frames side by side."""
    
    with st.spinner("Processing frames for comparison..."):
        try:
            api_client = APIClient()
            
            # Process both frames
            result1 = api_client.process_frame(file_id, frame1)
            result2 = api_client.process_frame(file_id, frame2)
            
            # Display comparison
            st.subheader("📊 Comparison Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Frame {frame1}**")
                st.metric("Detections", len(result1.get('detections', [])))
                if result1.get('annotated_frame'):
                    st.image(result1['annotated_frame'], caption=f"Frame {frame1}")
                
                if result1.get('detections'):
                    st.write("**Detections:**")
                    for det in result1['detections']:
                        st.write(f"• {det.get('class_name', 'Unknown')} ({det.get('confidence', 0):.2%})")
            
            with col2:
                st.write(f"**Frame {frame2}**")
                st.metric("Detections", len(result2.get('detections', [])))
                if result2.get('annotated_frame'):
                    st.image(result2['annotated_frame'], caption=f"Frame {frame2}")
                
                if result2.get('detections'):
                    st.write("**Detections:**")
                    for det in result2['detections']:
                        st.write(f"• {det.get('class_name', 'Unknown')} ({det.get('confidence', 0):.2%})")
                        
        except Exception as e:
            show_error_message(f"Failed to compare frames: {str(e)}")


def render_detection_settings():
    """Render detection settings panel."""
    
    st.subheader("⚙️ Detection Settings")
    
    with st.expander("🎛️ Model Parameters", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Minimum confidence for detections"
            )
        
        with col2:
            iou_threshold = st.slider(
                "IoU Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.45,
                step=0.05,
                help="Intersection over Union threshold for NMS"
            )
        
        # Store settings in session state
        st.session_state['confidence_threshold'] = confidence_threshold
        st.session_state['iou_threshold'] = iou_threshold
        
        if st.button("Apply Settings"):
            st.success("Settings updated successfully!")
            st.rerun()