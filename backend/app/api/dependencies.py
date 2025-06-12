from typing import Generator, Optional
from fastapi import Depends, HTTPException, status
from ..core.detector import BikeDetector
from ..core.config import Settings
import os

# Global detector instance
detector_instance: Optional[BikeDetector] = None

def get_settings() -> Settings:
    """Get application settings."""
    return Settings()

def get_detector() -> BikeDetector:
    """Get bike detector instance (singleton)."""
    global detector_instance
    
    if detector_instance is None:
        settings = get_settings()
        
        # Check if model file exists
        if not os.path.exists(settings.MODEL_PATH):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Model file not found at {settings.MODEL_PATH}"
            )
        
        try:
            detector_instance = BikeDetector(
                model_path=settings.MODEL_PATH,
                confidence_threshold=settings.CONFIDENCE_THRESHOLD,
                iou_threshold=settings.IOU_THRESHOLD
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to initialize detector: {str(e)}"
            )
    
    return detector_instance

def validate_file_exists(file_path: str) -> str:
    """Validate that file exists."""
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found: {file_path}"
        )
    return file_path

def validate_frame_number(frame_number: int, total_frames: int) -> int:
    """Validate frame number is within valid range."""
    if frame_number < 0 or frame_number >= total_frames:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Frame number {frame_number} is out of range (0-{total_frames-1})"
        )
    return frame_number

def get_upload_directory() -> str:
    """Get upload directory path."""
    settings = get_settings()
    upload_dir = settings.UPLOAD_DIR
    
    # Create directory if it doesn't exist
    os.makedirs(upload_dir, exist_ok=True)
    
    return upload_dir

def get_output_directory() -> str:
    """Get output directory path."""
    settings = get_settings()
    output_dir = settings.OUTPUT_DIR
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir

def cleanup_temp_files(file_paths: list[str]) -> None:
    """Clean up temporary files."""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            # Ignore cleanup errors
            pass