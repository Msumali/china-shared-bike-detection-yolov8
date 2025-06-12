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
    os.makedirs(settings.upload_dir, exist_ok=True)
    os.makedirs(settings.output_dir, exist_ok=True)
    
    # Check if model file exists
    if not os.path.exists(settings.model_path):
        print(f"Warning: Model file not found at {settings.model_path}")
        print("Please ensure your trained YOLOv8 model is placed in the models/ directory")
    
    print(f"Starting server on {settings.host}:{settings.port}")
    print(f"Model path: {settings.model_path}")
    print(f"Upload directory: {settings.upload_dir}")
    print(f"Output directory: {settings.output_dir}")
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        access_log=True,
        log_level="info" if not settings.debug else "debug"
    )

if __name__ == "__main__":
    main()