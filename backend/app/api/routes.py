from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends, Query
from fastapi.responses import JSONResponse, FileResponse
from typing import Dict, Any, List, Optional
import os
import json
import time
import uuid
from datetime import datetime, timedelta
import pandas as pd
import cv2
import numpy as np
from PIL import Image
import io
import base64

from ..core.detector import BikeDetector
from ..core.video_processor import VideoProcessor
from app.core.config import settings
from ..models.schemas import (
    DetectionResponse,
    VideoInfoResponse,
    BatchProcessRequest,
    BatchProcessResponse,
    StatisticsResponse,
    # SystemHealthResponse
)
from ..utils.file_handler import FileHandler
from .dependencies import get_detector, get_video_processor, get_file_handler
from .dependencies import get_settings
import logging

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
settings = get_settings()

# In-memory storage for demo purposes (replace with database in production)
batch_jobs = {}
processing_history = []
statistics_cache = {}

@router.post("/upload-video/", response_model=Dict[str, Any])
async def upload_video(
    file: UploadFile = File(...),
    file_handler: FileHandler = Depends(get_file_handler)
):
    """Upload and process a video file with YOLO."""
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            raise HTTPException(status_code=400, detail="Invalid file type. Only video files are allowed.")
        
        # Save uploaded file using your existing method
        file_id, file_path = file_handler.save_uploaded_file(file)
        
        # Get video information
        video_processor = VideoProcessor()
        video_info = video_processor.get_video_info(file_path)
        
        # YOLO processing
        start_time = datetime.now()
        yolo_results = video_processor.process_with_yolo(file_path)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Store in statistics cache
        statistics_cache[file_id] = {
            'filename': file.filename,
            'upload_time': datetime.now().isoformat(),
            'processing_time': processing_time,
            'fps': video_info.get('fps', 0),
            'detections': yolo_results.get('detections', [])
        }
        
        return {
            "success": True,
            "file_id": file_id,
            "filename": file.filename,
            "video_info": video_info,
            "processing_time": processing_time,
            "detections_count": len(yolo_results.get('detections', [])),
            "message": "Video uploaded and processed successfully"
        }
        
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")



@router.get("/video-info/{file_id}", response_model=VideoInfoResponse)
async def get_video_info(file_id: str):
    """Get video information by file ID."""
    try:
        file_info = statistics_cache.get(f"file_{file_id}")
        if not file_info:
            raise HTTPException(status_code=404, detail="Video not found")
        
        return {
            "success": True,
            "data": file_info["video_info"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process-frame/", response_model=DetectionResponse)
async def process_frame(
    file_id: str,
    frame_number: int,
    confidence_threshold: float = 0.5,
    iou_threshold: float = 0.45,
    detector: BikeDetector = Depends(get_detector)
):
    """Process a single frame from the uploaded video."""
    try:
        # Get file info
        file_info = statistics_cache.get(f"file_{file_id}")
        if not file_info:
            raise HTTPException(status_code=404, detail="Video not found")
        
        file_path = file_info["file_path"]
        
        # Extract frame
        cap = cv2.VideoCapture(file_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise HTTPException(status_code=400, detail="Could not extract frame")
        
        # Process frame
        start_time = time.time()
        detections = detector.detect_bikes(
            frame, 
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold
        )
        processing_time = time.time() - start_time
        
        # Create annotated frame
        annotated_frame = detector.draw_detections(frame, detections)
        
        # Convert annotated frame to base64 for frontend display
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Format detections
        formatted_detections = []
        for det in detections:
            formatted_detections.append({
                "class_name": "bike",
                "confidence": float(det["confidence"]),
                "bbox": {
                    "x": float(det["bbox"][0]),
                    "y": float(det["bbox"][1]),
                    "width": float(det["bbox"][2] - det["bbox"][0]),
                    "height": float(det["bbox"][3] - det["bbox"][1])
                }
            })
        
        return {
            "success": True,
            "frame_number": frame_number,
            "detections": formatted_detections,
            "processing_time": processing_time,
            "annotated_frame": f"data:image/jpeg;base64,{img_base64}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Frame processing failed: {str(e)}")


@router.post("/start-batch-processing/", response_model=BatchProcessResponse)
async def start_batch_processing(
    request: BatchProcessRequest,
    background_tasks: BackgroundTasks,
    detector: BikeDetector = Depends(get_detector),
    video_processor: VideoProcessor = Depends(get_video_processor)
):
    """Start batch processing of a video."""
    try:
        # Generate task ID
        task_id = str(uuid.uuid4())
        
        # Get file info
        file_info = statistics_cache.get(f"file_{request.file_id}")
        if not file_info:
            raise HTTPException(status_code=404, detail="Video not found")
        
        # Initialize job status
        batch_jobs[task_id] = {
            "task_id": task_id,
            "status": "pending",
            "progress": 0,
            "start_time": datetime.now().isoformat(),
            "file_id": request.file_id,
            "parameters": request.dict()
        }
        
        # Start background processing
        background_tasks.add_task(
            process_video_batch,
            task_id,
            file_info["file_path"],
            request,
            detector,
            video_processor
        )
        
        return {
            "success": True,
            "task_id": task_id,
            "message": "Batch processing started"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start batch processing: {str(e)}")


@router.get("/batch-status/{task_id}")
async def get_batch_status(task_id: str):
    """Get batch processing status."""
    try:
        if task_id not in batch_jobs:
            raise HTTPException(status_code=404, detail="Task not found")
        
        job = batch_jobs[task_id]
        
        return {
            "success": True,
            "data": job
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics/")
async def get_statistics(
    start_time: datetime = Query(..., description="Start time for statistics"),
    end_time: datetime = Query(..., description="End time for statistics")
):
    """Get statistics from YOLO video processing."""
    try:
        stats = generate_real_statistics(start_time, end_time)
        return stats
    except Exception as e:
        logger.error(f"Error generating statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating statistics: {str(e)}")


# Make sure your upload endpoint stores YOLO results like this:
"""
statistics_cache[video_id] = {
    'filename': filename,
    'upload_time': datetime.now().isoformat(),
    'processing_time': processing_time_seconds,
    'fps': video_fps,
    'detections': [
        {
            'confidence': 0.85,
            'class': 'motorcycle',  # YOLO class name
            'bbox': [x, y, w, h]    # bounding box coordinates
        }
    ]
}
"""

# @router.get("/system-health/", response_model=SystemHealthResponse)
# async def get_system_health():
#     """Get system health metrics."""
#     try:
#         import psutil
        
#         # Get system metrics
#         cpu_usage = psutil.cpu_percent(interval=1)
#         memory = psutil.virtual_memory()
#         disk = psutil.disk_usage('/')
        
#         # Mock GPU usage (replace with actual GPU monitoring if available)
#         gpu_usage = 0.0
#         try:
#             import GPUtil
#             gpus = GPUtil.getGPUs()
#             if gpus:
#                 gpu_usage = gpus[0].load * 100
#         except:
#             pass
        
#         health_data = {
#             "cpu_usage": cpu_usage,
#             "memory_usage": memory.percent,
#             "disk_usage": disk.percent,
#             "gpu_usage": gpu_usage,
#             "services": {
#                 "api": "healthy",
#                 "detector": "healthy",
#                 "storage": "healthy"
#             },
#             "timestamp": datetime.now().isoformat()
#         }
        
#         return {
#             "success": True,
#             "data": health_data
#         }
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to get system health: {str(e)}")


@router.get("/processing-history/")
async def get_processing_history():
    """Get batch processing history."""
    try:
        return {
            "success": True,
            "data": processing_history[-50:]  # Return last 50 jobs
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/clear-history/")
async def clear_processing_history():
    """Clear processing history."""
    try:
        global processing_history
        processing_history.clear()
        
        return {
            "success": True,
            "message": "Processing history cleared"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download-results/{task_id}")
async def download_results(task_id: str, format: str = "json"):
    """Download batch processing results."""
    try:
        if task_id not in batch_jobs:
            raise HTTPException(status_code=404, detail="Task not found")
        
        job = batch_jobs[task_id]
        
        if job["status"] != "completed":
            raise HTTPException(status_code=400, detail="Job not completed")
        
        results = job.get("results", {})
        
        if format.lower() == "csv":
            # Convert to CSV
            detections = results.get("detections", [])
            if detections:
                df = pd.DataFrame(detections)
                csv_content = df.to_csv(index=False)
                
                return Response(
                    content=csv_content,
                    media_type="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=results_{task_id}.csv"}
                )
        else:
            # Return JSON
            json_content = json.dumps(results, indent=2)
            return Response(
                content=json_content,
                media_type="application/json",
                headers={"Content-Disposition": f"attachment; filename=results_{task_id}.json"}
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Background task functions
async def process_video_batch(
    task_id: str,
    video_path: str,
    request: BatchProcessRequest,
    detector: BikeDetector,
    video_processor: VideoProcessor
):
    """Background task for batch processing."""
    try:
        # Update job status
        batch_jobs[task_id]["status"] = "running"
        batch_jobs[task_id]["progress"] = 0
        
        # Get video info
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Determine frames to process based on mode
        frames_to_process = []
        
        if request.processing_mode == "all_frames":
            frames_to_process = list(range(total_frames))
        elif request.processing_mode == "every_n_frames":
            interval = getattr(request, 'frame_interval', 30)
            frames_to_process = list(range(0, total_frames, interval))
        elif request.processing_mode == "time_interval":
            time_interval = getattr(request, 'time_interval', 1.0)
            frame_interval = int(fps * time_interval)
            frames_to_process = list(range(0, total_frames, frame_interval))
        elif request.processing_mode == "frame_range":
            start_frame = getattr(request, 'start_frame', 0)
            end_frame = getattr(request, 'end_frame', min(100, total_frames - 1))
            frames_to_process = list(range(start_frame, end_frame + 1))
        
        # Process frames
        all_detections = []
        processed_count = 0
        
        for i, frame_num in enumerate(frames_to_process):
            try:
                # Extract frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                
                if ret:
                    # Detect bikes
                    detections = detector.detect_bikes(frame)
                    
                    # Add frame info to detections
                    for det in detections:
                        det["frame_number"] = frame_num
                        det["timestamp"] = frame_num / fps
                        all_detections.append(det)
                    
                    processed_count += 1
                
                # Update progress
                progress = (i + 1) / len(frames_to_process) * 100
                batch_jobs[task_id]["progress"] = progress
                
            except Exception as e:
                print(f"Error processing frame {frame_num}: {str(e)}")
                continue
        
        cap.release()
        
        # Compile results
        results = {
            "total_frames": len(frames_to_process),
            "processed_frames": processed_count,
            "total_detections": len(all_detections),
            "detections": all_detections,
            "avg_confidence": sum(d["confidence"] for d in all_detections) / len(all_detections) if all_detections else 0,
            "processing_time": time.time() - time.mktime(datetime.fromisoformat(batch_jobs[task_id]["start_time"]).timetuple())
        }
        
        # Update job status
        batch_jobs[task_id]["status"] = "completed"
        batch_jobs[task_id]["progress"] = 100
        batch_jobs[task_id]["results"] = results
        batch_jobs[task_id]["end_time"] = datetime.now().isoformat()
        
        # Add to history
        processing_history.append({
            "task_id": task_id,
            "timestamp": batch_jobs[task_id]["start_time"],
            "status": "completed",
            "processing_mode": request.processing_mode,
            "output_format": request.output_format,
            "frames_processed": processed_count,
            "total_detections": len(all_detections),
            "processing_time": results["processing_time"]
        })
        
    except Exception as e:
        # Update job status on error
        batch_jobs[task_id]["status"] = "failed"
        batch_jobs[task_id]["error"] = str(e)
        batch_jobs[task_id]["end_time"] = datetime.now().isoformat()
        
        # Add to history
        processing_history.append({
            "task_id": task_id,
            "timestamp": batch_jobs[task_id]["start_time"],
            "status": "failed",
            "processing_mode": request.processing_mode,
            "error": str(e)
        })


def generate_real_statistics(start_time: datetime, end_time: datetime) -> Dict[str, Any]:
    """Generate real statistics from actual YOLO video processing data."""
    
    # Filter statistics_cache for the time range
    filtered_data = []
    for video_data in statistics_cache.values():
        upload_time = datetime.fromisoformat(video_data.get('upload_time', ''))
        if start_time <= upload_time <= end_time:
            filtered_data.append(video_data)
    
    if not filtered_data:
        # Return empty statistics if no data in range
        return {
            "total_detections": 0,
            "total_videos_processed": 0,
            "average_confidence": 0,
            "total_processing_time": 0,
            "detections_change": 0,
            "videos_change": 0,
            "confidence_change": 0,
            "processing_time_change": 0,
            "unique_bikes_detected": 0,
            "average_fps": 0,
            "detection_accuracy": 0,
            "processing_success_rate": 100.0,
            "time_series": [],
            "confidence_distribution": [],
            "brands": {},
            "performance_metrics": {
                "avg_frame_time": 0,
                "fps": 0,
                "memory_usage": 0,
                "cpu_usage": 0,
                "gpu_usage": 0
            },
            "quality_metrics": {
                "high_confidence": 0,
                "medium_confidence": 0,
                "low_confidence": 0,
                "false_positives": 0,
                "missed_detections": 0
            },
            "top_videos": []
        }
    
    # Calculate from real YOLO detection data
    total_detections = sum(len(data.get('detections', [])) for data in filtered_data)
    total_videos = len(filtered_data)
    
    # Get all confidence scores from YOLO detections
    all_confidences = []
    for data in filtered_data:
        detections = data.get('detections', [])
        for detection in detections:
            if 'confidence' in detection:
                all_confidences.append(detection['confidence'])
    
    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
    total_processing_time = sum(data.get('processing_time', 0) for data in filtered_data)
    
    # Generate time series from actual processing
    time_series = []
    current_time = start_time
    
    while current_time <= end_time:
        hour_detections = 0
        hour_confidences = []
        
        for data in filtered_data:
            upload_time = datetime.fromisoformat(data.get('upload_time', ''))
            if (upload_time.replace(minute=0, second=0, microsecond=0) == 
                current_time.replace(minute=0, second=0, microsecond=0)):
                
                detections = data.get('detections', [])
                hour_detections += len(detections)
                
                for detection in detections:
                    if 'confidence' in detection:
                        hour_confidences.append(detection['confidence'])
        
        hour_avg_confidence = sum(hour_confidences) / len(hour_confidences) if hour_confidences else 0
        
        time_series.append({
            "timestamp": current_time.isoformat(),
            "detections": hour_detections,
            "confidence": hour_avg_confidence,
            "hour": current_time.hour
        })
        current_time += timedelta(hours=1)
    
    # Count detections by class/brand from YOLO
    brand_counts = {}
    for data in filtered_data:
        detections = data.get('detections', [])
        for detection in detections:
            brand = detection.get('class', 'motorcycle')  # YOLO class name
            brand_counts[brand] = brand_counts.get(brand, 0) + 1
    
    # Quality metrics based on confidence thresholds
    high_confidence = sum(1 for conf in all_confidences if conf >= 0.7)
    medium_confidence = sum(1 for conf in all_confidences if 0.4 <= conf < 0.7)
    low_confidence = sum(1 for conf in all_confidences if conf < 0.4)
    
    # FPS from video processing
    fps_values = [data.get('fps', 0) for data in filtered_data if data.get('fps', 0) > 0]
    avg_fps = sum(fps_values) / len(fps_values) if fps_values else 0
    
    # Top videos by detection count
    top_videos = sorted(
        [
            {
                "filename": data.get('filename', 'Unknown'),
                "detections": len(data.get('detections', [])),
                "avg_confidence": sum(d.get('confidence', 0) for d in data.get('detections', [])) / 
                                len(data.get('detections', [])) if data.get('detections') else 0,
                "processing_time": data.get('processing_time', 0)
            }
            for data in filtered_data
        ],
        key=lambda x: x['detections'],
        reverse=True
    )[:5]
    
    return {
        "total_detections": total_detections,
        "total_videos_processed": total_videos,
        "average_confidence": avg_confidence,
        "total_processing_time": total_processing_time,
        "detections_change": 0,
        "videos_change": 0,
        "confidence_change": 0,
        "processing_time_change": 0,
        "unique_bikes_detected": total_detections,  # Each detection is unique
        "average_fps": avg_fps,
        "detection_accuracy": avg_confidence * 100,
        "processing_success_rate": 100.0,
        "time_series": time_series,
        "confidence_distribution": all_confidences,
        "brands": brand_counts,
        "performance_metrics": {
            "avg_frame_time": total_processing_time / total_videos if total_videos > 0 else 0,
            "fps": avg_fps,
            "memory_usage": 0,
            "cpu_usage": 0,
            "gpu_usage": 0
        },
        "quality_metrics": {
            "high_confidence": high_confidence,
            "medium_confidence": medium_confidence,
            "low_confidence": low_confidence,
            "false_positives": 0,
            "missed_detections": 0
        },
        "top_videos": top_videos
    }