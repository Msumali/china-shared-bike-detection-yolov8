from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


class VideoInfoResponse(BaseModel):
    """Response model for video information."""
    file_id: str
    filename: str
    fps: float
    frame_count: int
    duration: float
    width: int
    height: int
    resolution: str


class Detection(BaseModel):
    """Model for a single detection."""
    class_id: int = Field(..., alias="class")
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2]
    brand: str
    color: List[int]  # [R, G, B]
    center: List[int]  # [x, y]
    area: int


class DetectionResponse(BaseModel):
    """Response model for detection results."""
    frame_number: int
    detections: List[Detection]
    timestamp: datetime


class BatchProcessRequest(BaseModel):
    """Request model for batch processing."""
    save_annotated_video: bool = True
    frame_skip: int = 1
    confidence_threshold: Optional[float] = None
    iou_threshold: Optional[float] = None


class ProcessingStatus(BaseModel):
    """Model for processing job status."""
    status: str  # started, processing, completed, failed
    progress: float
    file_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    failed_at: Optional[datetime] = None
    output_path: Optional[str] = None
    statistics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class DetectionStatistics(BaseModel):
    """Model for detection statistics."""
    total_detections: int
    brands: Dict[str, int]
    confidence_stats: Dict[str, float]
    bbox_stats: Dict[str, float]
    frames_processed: int
    processing_time: float


class BatchProcessResponse(BaseModel):
    """Response model for completed batch processing."""
    job_id: str
    statistics: DetectionStatistics
    output_file_url: Optional[str] = None
    completed_at: datetime


class ErrorResponse(BaseModel):
    """Model for error responses."""
    detail: str
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    model_path: str
    model_name: str
    classes: List[str]
    confidence_threshold: float
    iou_threshold: float
    input_size: List[int]


class FileUploadResponse(BaseModel):
    """Response model for file upload."""
    file_id: str
    filename: str
    file_size: int
    upload_time: datetime
    file_path: str


class RealtimeDetectionRequest(BaseModel):
    """Request model for real-time detection."""
    file_id: str
    start_frame: int = 0
    max_frames: Optional[int] = None


class FrameDetectionResponse(BaseModel):
    """Response model for single frame detection."""
    frame_number: int
    detections: List[Detection]
    processing_time: float
    frame_timestamp: float