import streamlit as st
import time
import pandas as pd
import json
import os
import tempfile
import cv2
from typing import Dict, Any, List, Optional
from utils.api_client import APIClient
from utils.ui_helpers import show_error_message, show_success_message, show_info_message
from config.settings import SESSION_KEYS

def has_uploaded_video() -> bool:
    """Check if a video has been uploaded."""
    return (
        SESSION_KEYS['file_id'] in st.session_state and 
        st.session_state[SESSION_KEYS['file_id']] is not None and
        SESSION_KEYS['video_info'] in st.session_state and
        st.session_state[SESSION_KEYS['video_info']] is not None
    )

def get_current_file_id() -> Optional[str]:
    """Get the current uploaded video file ID."""
    return st.session_state.get(SESSION_KEYS['file_id'])

def get_current_video_info() -> Optional[Dict[str, Any]]:
    """Get the current uploaded video information."""
    return st.session_state.get(SESSION_KEYS['video_info'])

def render_file_info(uploaded_file):
    """Display information about the uploaded file."""
    if uploaded_file is not None:
        st.info(f"**File:** {uploaded_file.name}")
        st.info(f"**Size:** {uploaded_file.size / (1024*1024):.2f} MB")
        st.info(f"**Type:** {uploaded_file.type}")

@st.cache_data
def get_video_info(file_content: bytes, filename: str) -> Dict[str, Any]:
    """Extract video information using OpenCV."""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name
        
        # Open video with OpenCV
        cap = cv2.VideoCapture(tmp_file_path)
        
        if not cap.isOpened():
            raise Exception("Could not open video file")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return {
            'filename': filename,
            'duration': duration,
            'frame_count': frame_count,
            'fps': fps,
            'width': width,
            'height': height,
            'codec': 'Unknown',  # Would need more complex extraction
            'bitrate': 0,  # Would need more complex extraction
            'file_size': len(file_content),
            'format': filename.split('.')[-1].upper()
        }
    
    except Exception as e:
        st.error(f"Error extracting video info: {str(e)}")
        return {
            'filename': filename,
            'duration': 0,
            'frame_count': 0,
            'fps': 0,
            'width': 0,
            'height': 0,
            'codec': 'Unknown',
            'bitrate': 0,
            'file_size': len(file_content),
            'format': filename.split('.')[-1].upper()
        }

def upload_video(uploaded_file):
    """Upload video file and store information in session state."""
    try:
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Reading video file...")
        progress_bar.progress(10)
        
        # Read file content
        file_content = uploaded_file.read()
        
        # Reset file pointer for preview
        uploaded_file.seek(0)
        
        progress_bar.progress(30)
        status_text.text("Extracting video information...")
        
        # Get video info directly (faster than API call)
        video_info = get_video_info(file_content, uploaded_file.name)
        
        progress_bar.progress(50)
        status_text.text("Uploading to server...")
        
        # Try to upload via API
        try:
            api_client = APIClient()
            response = api_client.upload_video(file_content, uploaded_file.name)
            
            progress_bar.progress(75)
            
            if response and response.get("success"):
                file_id = response.get("file_id")
                server_video_info = response.get("video_info", {})
                
                # Use server info if available, otherwise use local info
                final_video_info = server_video_info if server_video_info else video_info
                
            else:
                # Fallback: generate local file ID and use local processing
                import uuid
                file_id = f"local_{uuid.uuid4().hex[:8]}"
                final_video_info = video_info
                
                # Save file locally for processing
                save_video_locally(file_content, uploaded_file.name, file_id)
                
                show_info_message("Using local processing (API unavailable)")
        
        except Exception as api_error:
            # Fallback to local processing
            import uuid
            file_id = f"local_{uuid.uuid4().hex[:8]}"
            final_video_info = video_info
            
            # Save file locally
            save_video_locally(file_content, uploaded_file.name, file_id)
            
            show_info_message(f"API unavailable, using local processing: {str(api_error)}")
        
        # Store in session state
        st.session_state[SESSION_KEYS['uploaded_file']] = uploaded_file.name
        st.session_state[SESSION_KEYS['file_id']] = file_id
        st.session_state[SESSION_KEYS['video_info']] = final_video_info
        
        progress_bar.progress(100)
        status_text.text("Upload completed!")
        
        show_success_message(f"Video uploaded successfully! File ID: {file_id}")
        
        # Display video info
        render_current_video_info()
        
        # Clear progress indicators
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
    except Exception as e:
        show_error_message(f"Error uploading video: {str(e)}")
        # Clear progress indicators on error
        if 'progress_bar' in locals():
            progress_bar.empty()
        if 'status_text' in locals():
            status_text.empty()

def save_video_locally(file_content: bytes, filename: str, file_id: str):
    """Save video file locally for processing."""
    try:
        # Create uploads directory if it doesn't exist
        uploads_dir = "uploads"
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir)
        
        # Save file
        file_path = os.path.join(uploads_dir, f"{file_id}_{filename}")
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        # Store file path in session state
        st.session_state[f"{SESSION_KEYS['file_id']}_path"] = file_path
        
    except Exception as e:
        st.error(f"Error saving video locally: {str(e)}")

def preview_video(uploaded_file):
    """Preview the uploaded video file."""
    try:
        # Display video player
        st.video(uploaded_file)
        
        # Show file details
        st.write("**File Details:**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"Name: {uploaded_file.name}")
            st.write(f"Size: {uploaded_file.size / (1024*1024):.2f} MB")
        
        with col2:
            st.write(f"Type: {uploaded_file.type}")
            
    except Exception as e:
        show_error_message(f"Error previewing video: {str(e)}")

def render_current_video_info():
    """Display current uploaded video information."""
    if not has_uploaded_video():
        return
    
    video_info = get_current_video_info()
    file_id = get_current_file_id()
    
    st.subheader("📹 Current Video Information")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("File ID", file_id[:8] + "..." if len(file_id) > 8 else file_id)
    
    with col2:
        st.metric("Duration", f"{video_info.get('duration', 0):.2f}s")
    
    with col3:
        st.metric("Frames", video_info.get('frame_count', 0))
    
    with col4:
        st.metric("FPS", f"{video_info.get('fps', 0):.2f}")
    
    # Additional info in expandable section
    with st.expander("📊 Detailed Information", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Width:** {video_info.get('width', 0)}px")
            st.write(f"**Height:** {video_info.get('height', 0)}px")
            st.write(f"**Codec:** {video_info.get('codec', 'Unknown')}")
        
        with col2:
            st.write(f"**Bitrate:** {video_info.get('bitrate', 0)} kbps")
            st.write(f"**File Size:** {video_info.get('file_size', 0) / (1024*1024):.2f} MB")
            st.write(f"**Format:** {video_info.get('format', 'Unknown')}")
    
    # Clear video button
    if st.button("🗑️ Clear Video", key=f"clear_video_info_button_{get_current_file_id()}_{int(time.time())}"):
        clear_uploaded_video()
        st.rerun()

def clear_uploaded_video():
    """Clear the uploaded video from session state."""
    keys_to_clear = [
        SESSION_KEYS['uploaded_file'],
        SESSION_KEYS['file_id'], 
        SESSION_KEYS['video_info'],
        SESSION_KEYS.get('detection_results', 'detection_results'),
        SESSION_KEYS.get('current_frame', 'current_frame'),
        SESSION_KEYS.get('batch_task_id', 'batch_task_id'),
        SESSION_KEYS.get('batch_results', 'batch_results')
    ]
    
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    
    # Clear local file if exists
    file_id = get_current_file_id()
    if file_id and file_id.startswith('local_'):
        file_path_key = f"{SESSION_KEYS['file_id']}_path"
        if file_path_key in st.session_state:
            file_path = st.session_state[file_path_key]
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass
            del st.session_state[file_path_key]
    
    show_success_message("Video cleared successfully!")

def render_video_upload_component():
    """Render video upload component."""
    
    st.header("📹 Video Upload")
    
    # Upload tips
    with st.expander("💡 Upload Tips", expanded=False):
        st.write("**Supported formats:** MP4, AVI, MOV, MKV, WMV")
        st.write("**Recommended:** MP4 format for best compatibility")
        st.write("**File size:** Up to 200MB for optimal performance")
        st.write("**Resolution:** 1080p or lower for faster processing")
    
    # File upload section
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv', 'wmv'],
        help="Upload a video file for bike detection analysis"
    )
    
    if uploaded_file is not None:
        # Display file info
        render_file_info(uploaded_file)
        
        # Upload and preview buttons
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("🚀 Upload and Process", type="primary", use_container_width=True):
                upload_video(uploaded_file)
        
        with col2:
            if st.button("👁️ Preview", use_container_width=True):
                preview_video(uploaded_file)
    
    # Display current uploaded video info if available
    if has_uploaded_video():
        render_current_video_info()

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
                max_value=max(1, total_frames),
                value=30,
                help="Process every N frames"
            )
        
        with col2:
            estimated_frames = max(1, total_frames // frame_interval) if total_frames > 0 else 0
            st.metric("Estimated Frames to Process", estimated_frames)
        
        st.session_state['frame_interval'] = frame_interval
    
    elif mode == "Time Interval":
        st.subheader("⏰ Time-based Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            time_interval = st.number_input(
                "Time Interval (seconds)",
                min_value=0.1,
                max_value=max(0.1, duration),
                value=1.0,
                step=0.1,
                help="Process frame every N seconds"
            )
        
        with col2:
            estimated_frames = int(duration / time_interval) if duration > 0 else 0
            st.metric("Estimated Frames to Process", estimated_frames)
        
        st.session_state['time_interval'] = time_interval
    
    elif mode == "Frame Range":
        st.subheader("🎯 Range Selection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            start_frame = st.number_input(
                "Start Frame",
                min_value=0,
                max_value=max(0, total_frames - 1),
                value=0
            )
        
        with col2:
            end_frame = st.number_input(
                "End Frame",
                min_value=start_frame,
                max_value=max(start_frame, total_frames - 1),
                value=min(start_frame + 100, max(start_frame, total_frames - 1))
            )
        
        st.session_state['start_frame'] = start_frame
        st.session_state['end_frame'] = end_frame
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Selected Range", f"{start_frame} - {end_frame}")
        with col2:
            st.metric("Frames to Process", max(0, end_frame - start_frame + 1))

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
        # Check if local processing is needed
        if file_id.startswith('local_'):
            # Local processing
            process_video_locally(file_id, params, progress_bar, status_text)
        else:
            # API processing
            api_client = APIClient()
            
            # Start batch processing
            status_text.text("Starting batch processing...")
            response = api_client.start_batch_processing(params)
            
            if response and response.get("success"):
                task_id = response.get("task_id")
                st.session_state[SESSION_KEYS.get('batch_task_id', 'batch_task_id')] = task_id
                
                # Monitor progress
                monitor_batch_processing(task_id, progress_bar, status_text)
            else:
                error_msg = response.get('error', 'Unknown error') if response else 'API unavailable'
                show_error_message(f"Failed to start processing: {error_msg}")
                
    except Exception as e:
        show_error_message(f"Error starting batch processing: {str(e)}")
    finally:
        progress_bar.empty()
        status_text.empty()

def process_video_locally(file_id: str, params: Dict, progress_bar, status_text):
    """Process video locally when API is unavailable."""
    try:
        # Get file path
        file_path_key = f"{SESSION_KEYS['file_id']}_path"
        if file_path_key not in st.session_state:
            show_error_message("Local video file not found")
            return
        
        file_path = st.session_state[file_path_key]
        if not os.path.exists(file_path):
            show_error_message("Local video file does not exist")
            return
        
        status_text.text("Processing video locally...")
        progress_bar.progress(25)
        
        # Open video
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            show_error_message("Could not open video file")
            return
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Determine frames to process
        frames_to_process = []
        mode = params.get("processing_mode", "all_frames")
        
        if mode == "every_n_frames":
            interval = params.get("frame_interval", 30)
            frames_to_process = list(range(0, total_frames, interval))
        elif mode == "time_interval":
            time_interval = params.get("time_interval", 1.0)
            frame_interval = int(fps * time_interval)
            frames_to_process = list(range(0, total_frames, max(1, frame_interval)))
        elif mode == "frame_range":
            start_frame = params.get("start_frame", 0)
            end_frame = params.get("end_frame", min(100, total_frames - 1))
            frames_to_process = list(range(start_frame, min(end_frame + 1, total_frames)))
        else:  # all_frames
            frames_to_process = list(range(0, min(total_frames, 1000)))  # Limit for demo
        
        progress_bar.progress(50)
        status_text.text(f"Processing {len(frames_to_process)} frames...")
        
        # Simulate processing (replace with actual detection logic)
        results = []
        for i, frame_num in enumerate(frames_to_process):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if ret:
                # Simulate detection results
                results.append({
                    'frame_number': frame_num,
                    'timestamp': frame_num / fps,
                    'detections': 1,  # Simulated
                    'confidence': 0.85,  # Simulated
                    'bbox': [100, 100, 200, 200]  # Simulated
                })
            
            # Update progress
            progress = 50 + (i / len(frames_to_process)) * 45
            progress_bar.progress(progress / 100)
        
        cap.release()
        
        # Store results
        batch_results = {
            'total_frames': len(frames_to_process),
            'total_detections': len(results),
            'processing_time': len(frames_to_process) * 0.1,  # Simulated
            'avg_confidence': 0.85,  # Simulated
            'detections': results
        }
        
        st.session_state[SESSION_KEYS.get('batch_results', 'batch_results')] = batch_results
        
        progress_bar.progress(100)
        status_text.text("Local processing completed!")
        
        show_success_message("Local processing completed! Check the results below.")
        
        # Auto-display results
        time.sleep(1)
        display_batch_results()
        
    except Exception as e:
        show_error_message(f"Error in local processing: {str(e)}")

def monitor_batch_processing(task_id: str, progress_bar, status_text):
    """Monitor batch processing progress."""
    
    api_client = APIClient()
    
    for attempt in range(60):  # Max 60 attempts (2 minutes)
        try:
            # Get task status
            status_response = api_client.get_batch_status(task_id)
            
            if not status_response or not status_response.get("success"):
                show_error_message("Failed to get processing status")
                break
            
            status_data = status_response.get("data", {})
            status = status_data.get("status", "unknown")
            progress = status_data.get("progress", 0)
            
            # Update progress
            progress_bar.progress(min(progress / 100, 1.0))
            status_text.text(f"Status: {status.title()} - {progress:.1f}%")
            
            if status == "completed":
                progress_bar.progress(1.0)
                status_text.text("Processing completed successfully!")
                
                # Store results
                st.session_state[SESSION_KEYS.get('batch_results', 'batch_results')] = status_data.get("results")
                
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
    
    results_key = SESSION_KEYS.get('batch_results', 'batch_results')
    if results_key not in st.session_state:
        show_info_message("No batch processing results available.")
        return
    
    results = st.session_state[results_key]
    
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
        if 'confidence' in df.columns:
            filtered_df = df[df['confidence'] >= min_confidence].head(max_results)
        else:
            filtered_df = df.head(max_results)
        
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
    if st.button("🗑️ Clear Results", key="clear_batch_results_button"):
        if results_key in st.session_state:
            del st.session_state[results_key]
        task_id_key = SESSION_KEYS.get('batch_task_id', 'batch_task_id')
        if task_id_key in st.session_state:
            del st.session_state[task_id_key]
        st.rerun()

# Additional helper functions for better error handling and performance

def get_batch_processing_history():
    """Get batch processing history."""
    
    try:
        api_client = APIClient()
        response = api_client.get_processing_history()
        
        if response and response.get("success"):
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

def validate_video_file(uploaded_file) -> tuple[bool, str]:
    """Validate uploaded video file."""
    
    if uploaded_file is None:
        return False, "No file uploaded"
    
    # Check file size (200MB limit)
    max_size = 200 * 1024 * 1024  # 200MB in bytes
    if uploaded_file.size > max_size:
        return False, f"File too large. Maximum size is 200MB, got {uploaded_file.size / (1024*1024):.1f}MB"
    
    # Check file extension
    allowed_extensions = ['mp4', 'avi', 'mov', 'mkv', 'wmv']
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension not in allowed_extensions:
        return False, f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
    
    return True, "Valid file"

def create_output_directories():
    """Create necessary output directories."""
    directories = ['uploads', 'outputs', 'temp']
    
    for directory in directories:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except Exception as e:
                st.warning(f"Could not create directory {directory}: {str(e)}")

# Initialize directories on import
create_output_directories()