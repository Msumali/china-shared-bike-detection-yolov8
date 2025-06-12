import os
from typing import List
from pydantic import BaseSettings

class Settings(BaseSettings):
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    # CORS settings
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:8501",  # Streamlit default
        "http://localhost:3000",  # React default
        "http://127.0.0.1:8501",
        "http://127.0.0.1:3000"
    ]
    
    # Project paths
    PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    MODEL_PATH: str = os.path.join(PROJECT_ROOT, "models", "best.pt")
    UPLOADS_DIR: str = os.path.join(PROJECT_ROOT, "uploads")
    OUTPUTS_DIR: str = os.path.join(PROJECT_ROOT, "outputs")
    
    # Detection settings
    CONFIDENCE_THRESHOLD: float = 0.5
    IOU_THRESHOLD: float = 0.45
    
    # File upload settings
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS: List[str] = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"]
    
    # Bike detection classes
    BIKE_CLASSES: dict = {
        0: {"name": "Didi", "color": [255, 165, 0]},      # Orange
        1: {"name": "Meituan", "color": [255, 255, 0]},   # Yellow
        2: {"name": "HelloRide", "color": [0, 255, 0]}    # Green
    }
    
    # Video processing settings
    MAX_FRAME_SKIP: int = 10
    DEFAULT_FRAME_SKIP: int = 1
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create settings instance
settings = Settings()

# Create directories if they don't exist
os.makedirs(settings.UPLOADS_DIR, exist_ok=True)
os.makedirs(settings.OUTPUTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(settings.MODEL_PATH), exist_ok=True)