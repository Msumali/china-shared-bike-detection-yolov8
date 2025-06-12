#!/usr/bin/env python3
"""
Frontend runner for the Bike Detection System.
This script starts the Streamlit application.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def setup_environment():
    """Setup environment variables and paths."""
    
    # Get the directory of this script
    frontend_dir = Path(__file__).parent
    project_root = frontend_dir.parent
    
    # Add paths to Python path
    sys.path.insert(0, str(frontend_dir))
    sys.path.insert(0, str(project_root))
    
    # Set default environment variables if not already set
    env_vars = {
        'BACKEND_URL': 'http://localhost:8000',
        'FRONTEND_PORT': '8501',
        'DEBUG': 'False',
        'STREAMLIT_SERVER_HEADLESS': 'true',
        'STREAMLIT_SERVER_ENABLE_CORS': 'false',
        'STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION': 'false'
    }
    
    for key, default_value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = default_value
    
    return frontend_dir


def check_requirements():
    """Check if required packages are installed."""
    
    missing_packages = []

    try:
        import streamlit
    except ImportError:
        missing_packages.append('streamlit')
    
    try:
        import requests
    except ImportError:
        missing_packages.append('requests')
    
    try:
        import pandas
    except ImportError:
        missing_packages.append('pandas')
    
    try:
        import plotly
    except ImportError:
        missing_packages.append('plotly')
    
    try:
        import cv2
    except ImportError:
        missing_packages.append('opencv-python')
    
    try:
        from PIL import Image
    except ImportError:
        missing_packages.append('pillow')

    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All required packages are installed")
    return True

def run_streamlit_app(frontend_dir: Path, port: int = 8501, debug: bool = False):
    """Run the Streamlit application."""
    
    app_file = frontend_dir / "streamlit_app" / "app.py"
    
    if not app_file.exists():
        print(f"❌ App file not found: {app_file}")
        return False
    
    # Streamlit command
    cmd = [
        "streamlit",
        "run",
        str(app_file),
        "--server.port", str(port),
        "--server.headless", "true" if not debug else "false",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
        "--theme.base", "dark",
        "--theme.primaryColor", "#ff6b6b",
        "--theme.backgroundColor", "#0e1117",
        "--theme.secondaryBackgroundColor", "#262730"
    ]
    
    print(f"🚀 Starting Streamlit app on port {port}...")
    print(f"📱 Access the app at: http://localhost:{port}")
    
    if debug:
        print("🐛 Debug mode enabled")
        os.environ['DEBUG'] = 'True'
    
    try:
        # Run Streamlit
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start Streamlit: {e}")
        return False
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
        return True
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def main():
    """Main function to run the frontend."""
    
    parser = argparse.ArgumentParser(description="Run the Bike Detection System Frontend")
    parser.add_argument(
        "--port", 
        type=int, 
        default=8501,
        help="Port to run the frontend on (default: 8501)"
    )
    parser.add_argument(
        "--backend-url",
        type=str,
        default="http://localhost:8000",
        help="Backend URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check dependencies and exit"
    )
    
    args = parser.parse_args()
    
    # Setup environment
    frontend_dir = setup_environment()
    
    # Set backend URL
    os.environ['BACKEND_URL'] = args.backend_url
    
    # Check dependencies
    if not check_requirements():
        return 1
    
    if args.check_deps:
        print("✅ All dependencies are satisfied")
        return 0
    
    # Test backend connection
    if not test_backend_connection(args.backend_url):
        print("⚠️ Warning: Backend is not accessible")
        print("Make sure the backend is running before using the app")
    
    # Run the application
    success = run_streamlit_app(frontend_dir, args.port, args.debug)
    
    return 0 if success else 1


def test_backend_connection(backend_url: str) -> bool:
    """Test connection to the backend."""
    
    try:
        import requests
        
        print(f"🔍 Testing backend connection to {backend_url}...")
        
        response = requests.get(f"{backend_url}/health", timeout=5)
        
        if response.status_code == 200:
            print("✅ Backend is accessible")
            return True
        else:
            print(f"⚠️ Backend returned status code: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to backend at {backend_url}")
        return False
    except requests.exceptions.Timeout:
        print(f"⏰ Timeout connecting to backend at {backend_url}")
        return False
    except Exception as e:
        print(f"❌ Error testing backend connection: {e}")
        return False


def install_requirements():
    """Install requirements from requirements.txt."""
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    try:
        print("📦 Installing requirements...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ], check=True)
        print("✅ Requirements installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False


if __name__ == "__main__":
    # Check if we're being run directly
    if len(sys.argv) > 1 and sys.argv[1] == "--install-deps":
        # Install dependencies
        success = install_requirements()
        sys.exit(0 if success else 1)
    
    # Run the main function
    sys.exit(main())