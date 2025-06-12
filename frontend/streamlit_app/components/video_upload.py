import streamlit as st
import time
import pandas as pd
import json
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
        
        st.session_state['start_frame'] = start_frame
        st.session_state['end_frame'] = end_frame
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Selected Range", f"{start_frame} - {end_frame}")
        with col2:
            st.metric("Frames to Process", end_frame - start_frame + 1)


def start_batch_processing(file_id: str, video_info: Dict[str, Any], 
                          processing_mode: str, output_format: str):
    """Start batch processing of the video."""
    
    # Prepare processing parameters
    params = {
        "file_id": file_id,
        "processing_mode": processing_mode.lower().replace(" ", "_"),
        "output_format": output_format.lower(),
    }
    
    # Add mode-specific parameters
    if processing_mode == "Every N Frames":
        params["frame_interval"] = st.session_state.get('frame_interval', 30)
    elif processing_mode == "Time Interval":
        params["time_interval"] = st.session_state.get('time_interval', 1.0)
    elif processing_mode == "Frame Range":
        params["start_frame"] = st.session_state.get('start_frame', 0)
        params["end_frame"] = st.session_state.get('end_frame', 100)
    
    # Start processing
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        api_client = APIClient()
        
        # Start batch processing
        status_text.text("Starting batch processing...")
        response = api_client.start_batch_processing(params)
        
        if response.get("success"):
            task_id = response.get("task_id")
            st.session_state[SESSION_KEYS.BATCH_TASK_ID] = task_id
            
            # Monitor progress
            monitor_batch_processing(task_id, progress_bar, status_text)
        else:
            show_error_message(f"Failed to start processing: {response.get('error', 'Unknown error')}")
            
    except Exception as e:
        show_error_message(f"Error starting batch processing: {str(e)}")
    finally:
        progress_bar.empty()
        status_text.empty()


def monitor_batch_processing(task_id: str, progress_bar, status_text):
    """Monitor batch processing progress."""
    
    api_client = APIClient()
    
    while True:
        try:
            # Get task status
            status_response = api_client.get_batch_status(task_id)
            
            if not status_response.get("success"):
                show_error_message("Failed to get processing status")
                break
            
            status_data = status_response.get("data", {})
            status = status_data.get("status", "unknown")
            progress = status_data.get("progress", 0)
            
            # Update progress
            progress_bar.progress(progress / 100)
            status_text.text(f"Status: {status.title()} - {progress:.1f}%")
            
            if status == "completed":
                progress_bar.progress(1.0)
                status_text.text("Processing completed successfully!")
                
                # Store results
                st.session_state[SESSION_KEYS.BATCH_RESULTS] = status_data.get("results")
                
                show_success_message("Batch processing completed! Check the results below.")
                
                # Auto-display results
                time.sleep(1)
                display_batch_results()
                break
                
            elif status == "failed":
                error_msg = status_data.get("error", "Processing failed")
                show_error_message(f"Processing failed: {error_msg}")
                break
                
            elif status == "running":
                time.sleep(2)  # Wait before next check
            else:
                time.sleep(1)
                
        except Exception as e:
            show_error_message(f"Error monitoring progress: {str(e)}")
            break


def display_batch_results():
    """Display batch processing results."""
    
    if SESSION_KEYS.BATCH_RESULTS not in st.session_state:
        show_info_message("No batch processing results available.")
        return
    
    results = st.session_state[SESSION_KEYS.BATCH_RESULTS]
    
    if not results:
        show_info_message("No results to display.")
        return
    
    st.subheader("📊 Batch Processing Results")
    
    # Results summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Frames Processed", results.get("total_frames", 0))
    
    with col2:
        st.metric("Bikes Detected", results.get("total_detections", 0))
    
    with col3:
        st.metric("Processing Time", f"{results.get('processing_time', 0):.2f}s")
    
    with col4:
        st.metric("Average Confidence", f"{results.get('avg_confidence', 0):.2f}")
    
    # Detailed results
    detections = results.get("detections", [])
    
    if detections:
        # Convert to DataFrame for better display
        df = pd.DataFrame(detections)
        
        # Results table
        st.subheader("🔍 Detection Details")
        
        # Add filters
        col1, col2 = st.columns(2)
        
        with col1:
            min_confidence = st.slider(
                "Minimum Confidence",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05
            )
        
        with col2:
            max_results = st.number_input(
                "Max Results to Show",
                min_value=10,
                max_value=len(df),
                value=min(100, len(df))
            )
        
        # Filter results
        filtered_df = df[df['confidence'] >= min_confidence].head(max_results)
        
        # Display table
        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Download options
        st.subheader("💾 Download Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_data = filtered_df.to_csv(index=False)
            st.download_button(
                "📄 Download CSV",
                data=csv_data,
                file_name="batch_results.csv",
                mime="text/csv"
            )
        
        with col2:
            json_data = filtered_df.to_json(orient="records", indent=2)
            st.download_button(
                "📋 Download JSON",
                data=json_data,
                file_name="batch_results.json",
                mime="application/json"
            )
        
        with col3:
            if results.get("output_video_path"):
                st.download_button(
                    "🎥 Download Video",
                    data="Video download not implemented",
                    file_name="processed_video.mp4",
                    mime="video/mp4",
                    disabled=True
                )
    
    # Clear results button
    if st.button("🗑️ Clear Results"):
        if SESSION_KEYS.BATCH_RESULTS in st.session_state:
            del st.session_state[SESSION_KEYS.BATCH_RESULTS]
        if SESSION_KEYS.BATCH_TASK_ID in st.session_state:
            del st.session_state[SESSION_KEYS.BATCH_TASK_ID]
        st.rerun()


def get_batch_processing_history():
    """Get batch processing history."""
    
    try:
        api_client = APIClient()
        response = api_client.get_processing_history()
        
        if response.get("success"):
            return response.get("data", [])
        else:
            return []
            
    except Exception as e:
        st.error(f"Error getting processing history: {str(e)}")
        return []


def render_processing_history():
    """Render processing history section."""
    
    st.subheader("📚 Processing History")
    
    history = get_batch_processing_history()
    
    if not history:
        show_info_message("No processing history available.")
        return
    
    # Display history
    for i, item in enumerate(history[-10:]):  # Show last 10 items
        with st.expander(f"Processing {i+1} - {item.get('timestamp', 'Unknown')}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Status:** {item.get('status', 'Unknown')}")
                st.write(f"**Mode:** {item.get('processing_mode', 'Unknown')}")
                st.write(f"**Output Format:** {item.get('output_format', 'Unknown')}")
            
            with col2:
                st.write(f"**Frames Processed:** {item.get('frames_processed', 0)}")
                st.write(f"**Detections:** {item.get('total_detections', 0)}")
                st.write(f"**Processing Time:** {item.get('processing_time', 0):.2f}s")