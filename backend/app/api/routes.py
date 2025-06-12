from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from typing import Optional
import os
import uuid
import json
from datetime import datetime

from app.core.detector import BikeDetector
from app.core.video_processor import VideoProcessor
from app.models.schemas import (
    VideoInfoResponse, DetectionResponse, ProcessingStatus,
    BatchProcessRequest, BatchProcessResponse
)
from app.utils.file_handler import FileHandler
from app.core.config import settings

router = APIRouter()

# Initialize components
detector = BikeDetector()
video_processor = VideoProcessor()
file_handler = FileHandler()

# Store processing jobs
processing_jobs = {}

@router.post("/upload", response_model=VideoInfoResponse)
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file and return video information."""
    try:
        # Validate file
        validation_result = file_handler.validate_file(file)
        if not validation_result["valid"]:
            raise HTTPException(status_code=400, detail=validation_result["message"])
        
        # Save file
        file_id = str(uuid.uuid4())
        file_path = await file_handler.save_file(file, file_id)
        
        # Get video information
        video_info = video_processor.get_video_info(file_path)
        if not video_info:
            raise HTTPException(status_code=400, detail="Invalid video file")
        
        response = VideoInfoResponse(
            file_id=file_id,
            filename=file.filename,
            **video_info
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing upload: {str(e)}")

@router.get("/video/{file_id}/info", response_model=VideoInfoResponse)
async def get_video_info(file_id: str):
    """Get information about an uploaded video."""
    try:
        file_path = file_handler.get_file_path(file_id)
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Video not found")
        
        video_info = video_processor.get_video_info(file_path)
        if not video_info:
            raise HTTPException(status_code=400, detail="Invalid video file")
        
        return VideoInfoResponse(
            file_id=file_id,
            filename=os.path.basename(file_path),
            **video_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting video info: {str(e)}")

@router.post("/detect/frame/{file_id}")
async def detect_frame(file_id: str, frame_number: int = 0):
    """Detect bikes in a specific frame."""
    try:
        file_path = file_handler.get_file_path(file_id)
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Video not found")
        
        # Extract frame
        frame = video_processor.extract_frame(file_path, frame_number)
        if frame is None:
            raise HTTPException(status_code=400, detail="Could not extract frame")
        
        # Detect bikes
        detections = detector.detect_bikes(frame)
        
        return DetectionResponse(
            frame_number=frame_number,
            detections=detections,
            timestamp=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting frame: {str(e)}")

@router.get("/detect/stream/{file_id}")
async def stream_detection(file_id: str, frame_skip: int = 1):
    """Stream real-time detection results."""
    try:
        file_path = file_handler.get_file_path(file_id)
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Video not found")
        
        def generate_detections():
            """Generator for streaming detection results."""
            try:
                for frame_num, annotated_frame, detections, progress in video_processor.process_video_realtime(
                    file_path, frame_skip=frame_skip
                ):
                    result = {
                        "frame_number": frame_num,
                        "detections": detections,
                        "progress": progress,
                        "timestamp": datetime.now().isoformat()
                    }
                    yield f"data: {json.dumps(result, default=str)}\n\n"
            except Exception as e:
                error_result = {"error": str(e)}
                yield f"data: {json.dumps(error_result)}\n\n"
        
        return StreamingResponse(
            generate_detections(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error streaming detection: {str(e)}")

@router.post("/process/batch/{file_id}")
async def start_batch_processing(
    file_id: str, 
    background_tasks: BackgroundTasks,
    request: Optional[BatchProcessRequest] = None
):
    """Start batch processing of entire video."""
    try:
        file_path = file_handler.get_file_path(file_id)
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Video not found")
        
        # Create job ID
        job_id = str(uuid.uuid4())
        
        # Initialize job status
        processing_jobs[job_id] = {
            "status": "started",
            "progress": 0,
            "file_id": file_id,
            "started_at": datetime.now(),
            "output_path": None,
            "statistics": None
        }
        
        # Start background processing
        background_tasks.add_task(
            process_video_background,
            job_id,
            file_path,
            request.save_annotated_video if request else True
        )
        
        return {"job_id": job_id, "status": "started"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting batch processing: {str(e)}")

@router.get("/process/status/{job_id}", response_model=ProcessingStatus)
async def get_processing_status(job_id: str):
    """Get the status of a batch processing job."""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    return ProcessingStatus(**job)

@router.get("/process/result/{job_id}")
async def get_processing_result(job_id: str):
    """Get the result of completed batch processing."""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")
    
    return BatchProcessResponse(
        job_id=job_id,
        statistics=job["statistics"],
        output_file_url=f"/api/v1/download/{job_id}" if job["output_path"] else None,
        completed_at=job.get("completed_at")
    )

@router.get("/download/{job_id}")
async def download_processed_video(job_id: str):
    """Download the processed video file."""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    
    if job["status"] != "completed" or not job["output_path"]:
        raise HTTPException(status_code=400, detail="Processed video not available")
    
    if not os.path.exists(job["output_path"]):
        raise HTTPException(status_code=404, detail="Processed video file not found")
    
    return FileResponse(
        job["output_path"],
        media_type="video/mp4",
        filename=f"processed_{job['file_id']}.mp4"
    )

@router.delete("/video/{file_id}")
async def delete_video(file_id: str):
    """Delete an uploaded video and its associated files."""
    try:
        success = file_handler.delete_file(file_id)
        if not success:
            raise HTTPException(status_code=404, detail="Video not found")
        
        return {"message": "Video deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting video: {str(e)}")

@router.get("/models/info")
async def get_model_info():
    """Get information about the loaded model."""
    try:
        model_info = detector.get_model_info()
        return model_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

async def process_video_background(job_id: str, file_path: str, save_video: bool = True):
    """Background task for video processing."""
    try:
        def progress_callback(progress):
            if job_id in processing_jobs:
                processing_jobs[job_id]["progress"] = progress
        
        processing_jobs[job_id]["status"] = "processing"
        
        if save_video:
            output_path, statistics = video_processor.process_video_batch(
                file_path, progress_callback=progress_callback
            )
            processing_jobs[job_id]["output_path"] = output_path
        else:
            # Process without saving (for statistics only)
            statistics = video_processor.get_detection_statistics(
                file_path, progress_callback=progress_callback
            )
        
        processing_jobs[job_id].update({
            "status": "completed",
            "progress": 100,
            "statistics": statistics,
            "completed_at": datetime.now()
        })
        
    except Exception as e:
        processing_jobs[job_id].update({
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.now()
        })