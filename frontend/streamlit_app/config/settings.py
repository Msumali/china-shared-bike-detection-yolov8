import os
from typing import List

# Backend API Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API_BASE_URL = BACKEND_URL  # For compatibility with api_client.py
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))

# Streamlit Configuration
PAGE_TITLE = "Bike Detection System"
PAGE_ICON = "🚴"
LAYOUT = "wide"

# File Upload Configuration
MAX_FILE_SIZE_MB = 100
ALLOWED_VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']

# UI Configuration
SIDEBAR_WIDTH = 300
MAIN_CONTENT_WIDTH = 800

# Detection Configuration
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_IOU_THRESHOLD = 0.45
DEFAULT_FRAME_SKIP = 1

# Brand Colors for Visualization
BRAND_COLORS = {
    'Didi': '#FF6B35',
    'Meituan': '#FFD23F',
    'HelloRide': '#06FFA5'
}

# Chart Configuration
CHART_HEIGHT = 400
CHART_WIDTH = 600

# Session State Keys
SESSION_KEYS = {
    'uploaded_file': 'uploaded_file',
    'file_id': 'file_id',
    'video_info': 'video_info',
    'processing_status': 'processing_status',
    'detection_results': 'detection_results',
    'current_frame': 'current_frame'
}

# Add APP_CONFIG for compatibility with your app.py
APP_CONFIG = {
    'BACKEND_URL': BACKEND_URL,
    'API_TIMEOUT': API_TIMEOUT,
    'DEBUG': os.getenv('DEBUG', 'False').lower() == 'true',
    'MAX_FILE_SIZE': MAX_FILE_SIZE_MB,
    'SUPPORTED_FORMATS': [ext.lstrip('.') for ext in ALLOWED_VIDEO_EXTENSIONS],
    'DETECTION_CONFIDENCE': DEFAULT_CONFIDENCE_THRESHOLD,
}