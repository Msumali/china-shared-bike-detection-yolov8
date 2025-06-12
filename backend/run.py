#!/usr/bin/env python3
"""
Run script for the bike detection backend server.
"""

import uvicorn
import os
import sys
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from app.main import app
from app.core.config import Settings

def main():
    """Run the FastAPI server."""
    settings = Settings()
    
    # Create necessary directories
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
    
    # Check if model file exists
    if not os.path.exists(settings.MODEL_PATH):
        print(f"Warning: Model file not found at {settings.MODEL_PATH}")
        print("Please ensure your trained YOLOv8 model is placed in the models/ directory")
    
    print(f"Starting server on {settings.HOST}:{settings.PORT}")
    print(f"Model path: {settings.MODEL_PATH}")
    print(f"Upload directory: {settings.UPLOAD_DIR}")
    print(f"Output directory: {settings.OUTPUT_DIR}")
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        access_log=True,
        log_level="info" if not settings.DEBUG else "debug"
    )

if __name__ == "__main__":
    main()