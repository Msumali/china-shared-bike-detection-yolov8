import os
import uuid
import shutil
from pathlib import Path
from typing import Optional, Tuple
from fastapi import UploadFile
from ..core.config import settings, UPLOADS_DIR, OUTPUTS_DIR


class FileHandler:
    def __init__(self):
        """Initialize file handler and ensure directories exist."""
        self.uploads_dir = Path(UPLOADS_DIR)
        self.outputs_dir = Path(OUTPUTS_DIR)
        
        # Create directories if they don't exist
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
    
    def save_uploaded_file(self, file: UploadFile) -> Tuple[str, str]:
        """
        Save uploaded file and return file_id and file_path.
        
        Args:
            file: FastAPI UploadFile object
            
        Returns:
            tuple: (file_id, file_path)
        """
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        
        # Get file extension
        file_extension = Path(file.filename).suffix.lower()
        
        # Validate file extension
        if file_extension not in settings.allowed_extensions:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        # Create file path
        filename = f"{file_id}{file_extension}"
        file_path = self.uploads_dir / filename
        
        # Save file
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            # Clean up if save failed
            if file_path.exists():
                file_path.unlink()
            raise e
        
        return file_id, str(file_path)
    
    def get_file_path(self, file_id: str) -> Optional[str]:
        """
        Get file path by file_id.
        
        Args:
            file_id: Unique file identifier
            
        Returns:
            str: File path or None if not found
        """
        for extension in settings.allowed_extensions:
            file_path = self.uploads_dir / f"{file_id}{extension}"
            if file_path.exists():
                return str(file_path)
        return None
    
    def delete_file(self, file_id: str) -> bool:
        """
        Delete file by file_id.
        
        Args:
            file_id: Unique file identifier
            
        Returns:
            bool: True if deleted successfully, False otherwise
        """
        file_path = self.get_file_path(file_id)
        if file_path and Path(file_path).exists():
            try:
                Path(file_path).unlink()
                return True
            except Exception:
                return False
        return False
    
    def get_output_path(self, filename: str) -> str:
        """
        Get output file path.
        
        Args:
            filename: Output filename
            
        Returns:
            str: Full output file path
        """
        return str(self.outputs_dir / filename)
    
    def cleanup_old_files(self, max_age_hours: int = 24):
        """
        Clean up old files older than max_age_hours.
        
        Args:
            max_age_hours: Maximum age in hours before deletion
        """
        import time
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        # Clean uploads
        for file_path in self.uploads_dir.glob("*"):
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    try:
                        file_path.unlink()
                    except Exception:
                        pass  # Ignore errors during cleanup
        
        # Clean outputs
        for file_path in self.outputs_dir.glob("*"):
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    try:
                        file_path.unlink()
                    except Exception:
                        pass  # Ignore errors during cleanup
    
    def get_file_size(self, file_path: str) -> int:
        """
        Get file size in bytes.
        
        Args:
            file_path: Path to the file
            
        Returns:
            int: File size in bytes
        """
        try:
            return Path(file_path).stat().st_size
        except Exception:
            return 0
    
    def file_exists(self, file_id: str) -> bool:
        """
        Check if file exists by file_id.
        
        Args:
            file_id: Unique file identifier
            
        Returns:
            bool: True if file exists, False otherwise
        """
        return self.get_file_path(file_id) is not None