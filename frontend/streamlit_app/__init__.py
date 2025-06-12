# frontend/streamlit_app/__init__.py
import sys
import os
from pathlib import Path

# Add streamlit_app directory
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Add its parent (frontend/)
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
