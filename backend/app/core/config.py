# backend/app/core/config.py

import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # App settings
    app_name: str = "Bike Detection API"
    debug: bool = True
    host: str = "127.0.0.1" 
    port: int = 8000         

    # Model settings
    model_path: str = "models/best.pt"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45

    # Bike classes mapping - adjust based on your trained model
    bike_classes: dict = {
        0: {"name": "Didi", "color": [0, 255, 0]},      # Orange
        1: {"name": "HelloRide", "color": [0, 0, 255]},   # Green
        2: {"name": "Meituan", "color": [255, 255, 0]},   # Yellow 
        
    }

    # Directory settings
    upload_dir: str = "uploads"
    output_dir: str = "outputs"

    # API settings
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_extensions: list = [".jpg", ".jpeg", ".png", ".mp4", ".avi", ".mov"]

    # I have added this for development purpose only
    allowed_origins: list[str] = ["*"]  # or specific domains for CORS

    class Config:
        env_file = ".env"

# Instantiate config
settings = Settings()

UPLOADS_DIR = settings.upload_dir
OUTPUTS_DIR = settings.output_dir
