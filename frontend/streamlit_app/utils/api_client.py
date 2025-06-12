import requests
import json
from typing import Dict, Any, List, Optional
import streamlit as st
from config.settings import API_BASE_URL, API_TIMEOUT


class APIClient:
    """API client for communicating with the backend."""
    
    def __init__(self):
        self.base_url = API_BASE_URL
        self.timeout = API_TIMEOUT
        self.session = requests.Session()
        
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request to the backend API."""
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                timeout=self.timeout,
                **kwargs
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}
        except json.JSONDecodeError:
            return {"success": False, "error": "Invalid JSON response"}
    
    # Video Upload Methods
    def upload_video(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Upload a video file to the backend."""
        files = {"file": (filename, file_content, "video/mp4")}
        return self._make_request("POST", "/api/v1/upload-video", files=files)
    
    def get_video_info(self, file_id: str) -> Dict[str, Any]:
        """Get video information by file ID."""
        return self._make_request("GET", f"/api/v1/video/{file_id}/info")
    
    def delete_video(self, file_id: str) -> Dict[str, Any]:
        """Delete a video file."""
        return self._make_request("DELETE", f"/api/v1/video/{file_id}")
    
    # Real-time Detection Methods
    def process_frame(self, file_id: str, frame_number: int, 
                     confidence_threshold: float = 0.5, 
                     iou_threshold: float = 0.45) -> Dict[str, Any]:
        """Process a single frame for detection."""
        data = {
            "frame_number": frame_number,
            "confidence_threshold": confidence_threshold,
            "iou_threshold": iou_threshold
        }
        return self._make_request("POST", f"/api/v1/video/{file_id}/process_frame", json=data)
    
    def get_frame(self, file_id: str, frame_number: int) -> Dict[str, Any]:
        """Get a specific frame from the video."""
        return self._make_request("GET", f"/api/v1/video/{file_id}/frame/{frame_number}")
    
    # Batch Processing Methods
    def start_batch_processing(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Start batch processing of a video."""
        return self._make_request("POST", "/api/v1/batch/start", json=params)
    
    def get_batch_status(self, task_id: str) -> Dict[str, Any]:
        """Get the status of a batch processing task."""
        return self._make_request("GET", f"/api/v1/batch/status/{task_id}")
    
    def get_batch_results(self, task_id: str) -> Dict[str, Any]:
        """Get the results of a completed batch processing task."""
        return self._make_request("GET", f"/api/v1/batch/results/{task_id}")
    
    def cancel_batch_processing(self, task_id: str) -> Dict[str, Any]:
        """Cancel a running batch processing task."""
        return self._make_request("POST", f"/api/v1/batch/cancel/{task_id}")
    
    def get_processing_history(self, limit: int = 50) -> Dict[str, Any]:
        """Get batch processing history."""
        params = {"limit": limit}
        return self._make_request("GET", "/api/v1/batch/history", params=params)
    
    # Statistics Methods
    def get_statistics(self, start_time: str, end_time: str) -> Dict[str, Any]:
        """Get detection statistics for a time range."""
        params = {
            "start_time": start_time,
            "end_time": end_time
        }
        return self._make_request("GET", "/api/v1/statistics", params=params)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health metrics."""
        return self._make_request("GET", "/api/v1/system/health")
    
    def get_detection_summary(self, file_id: str) -> Dict[str, Any]:
        """Get detection summary for a specific video."""
        return self._make_request("GET", f"/api/v1/video/{file_id}/summary")
    
    # Model Management Methods
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return self._make_request("GET", "/api/v1/model/info")
    
    def update_model_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Update model detection settings."""
        return self._make_request("POST", "/api/v1/model/settings", json=settings)
    
    # Utility Methods
    def health_check(self) -> Dict[str, Any]:
        """Check if the API is healthy."""
        return self._make_request("GET", "/health")  # This endpoint is at root level
    
    def get_supported_formats(self) -> Dict[str, Any]:
        """Get supported video formats."""
        return self._make_request("GET", "/api/v1/formats")
    
    def get_api_version(self) -> Dict[str, Any]:
        """Get API version information."""
        return self._make_request("GET", "/api/v1/version")
    
    # Export Methods
    def export_detections(self, file_id: str, format_type: str = "json") -> Dict[str, Any]:
        """Export detections in various formats."""
        params = {"format": format_type}
        return self._make_request("GET", f"/api/v1/video/{file_id}/export", params=params)
    
    def download_processed_video(self, file_id: str, task_id: str) -> Dict[str, Any]:
        """Download processed video with annotations."""
        return self._make_request("GET", f"/api/v1/video/{file_id}/download/{task_id}")
    
    # Real-time Streaming Methods (if needed)
    def start_stream_processing(self, stream_config: Dict[str, Any]) -> Dict[str, Any]:
        """Start real-time stream processing."""
        return self._make_request("POST", "/api/v1/stream/start", json=stream_config)
    
    def stop_stream_processing(self, stream_id: str) -> Dict[str, Any]:
        """Stop real-time stream processing."""
        return self._make_request("POST", f"/api/v1/stream/stop/{stream_id}")
    
    def get_stream_status(self, stream_id: str) -> Dict[str, Any]:
        """Get stream processing status."""
        return self._make_request("GET", f"/api/v1/stream/status/{stream_id}")
    
    # Configuration Methods
    def get_config(self) -> Dict[str, Any]:
        """Get current system configuration."""
        return self._make_request("GET", "/api/v1/config")
    
    def update_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Update system configuration."""
        return self._make_request("POST", "/api/v1/config", json=config)
    
    # Logging Methods
    def get_logs(self, level: str = "INFO", limit: int = 100) -> Dict[str, Any]:
        """Get system logs."""
        params = {"level": level, "limit": limit}
        return self._make_request("GET", "/api/v1/logs", params=params)
    
    # Performance Methods
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return self._make_request("GET", "/api/v1/metrics/performance")
    
    def get_detection_accuracy(self, file_id: str) -> Dict[str, Any]:
        """Get detection accuracy metrics for a video."""
        return self._make_request("GET", f"/api/v1/video/{file_id}/accuracy")
    
    # Database Methods (if using database)
    def save_detection_results(self, file_id: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Save detection results to database."""
        data = {"results": results}
        return self._make_request("POST", f"/api/v1/video/{file_id}/save_results", json=data)
    
    def get_saved_results(self, file_id: str) -> Dict[str, Any]:
        """Get saved detection results from database."""
        return self._make_request("GET", f"/api/v1/video/{file_id}/saved_results")
    
    # Cleanup Methods
    def cleanup_temp_files(self) -> Dict[str, Any]:
        """Clean up temporary files."""
        return self._make_request("POST", "/api/v1/cleanup/temp")
    
    def cleanup_old_results(self, days: int = 7) -> Dict[str, Any]:
        """Clean up old processing results."""
        params = {"days": days}
        return self._make_request("POST", "/api/v1/cleanup/results", params=params)
    
    # Annotation Methods
    def get_annotations(self, file_id: str, frame_number: int) -> Dict[str, Any]:
        """Get annotations for a specific frame."""
        return self._make_request("GET", f"/api/v1/video/{file_id}/annotations/{frame_number}")
    
    def save_annotations(self, file_id: str, frame_number: int, 
                        annotations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Save manual annotations for a frame."""
        data = {"annotations": annotations}
        return self._make_request("POST", f"/api/v1/video/{file_id}/annotations/{frame_number}", json=data)
    
    # Comparison Methods
    def compare_frames(self, file_id: str, frame1: int, frame2: int) -> Dict[str, Any]:
        """Compare detections between two frames."""
        data = {"frame1": frame1, "frame2": frame2}
        return self._make_request("POST", f"/api/v1/video/{file_id}/compare", json=data)
    
    def compare_videos(self, file_id1: str, file_id2: str) -> Dict[str, Any]:
        """Compare detections between two videos."""
        data = {"file_id1": file_id1, "file_id2": file_id2}
        return self._make_request("POST", "/api/v1/compare/videos", json=data)
    
    # Validation Methods
    def validate_video(self, file_id: str) -> Dict[str, Any]:
        """Validate video file integrity."""
        return self._make_request("GET", f"/api/v1/video/{file_id}/validate")
    
    def validate_detections(self, file_id: str, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate detection results."""
        data = {"detections": detections}
        return self._make_request("POST", f"/api/v1/video/{file_id}/validate_detections", json=data)
    
    # Thumbnail Methods
    def generate_thumbnail(self, file_id: str, frame_number: int = 0) -> Dict[str, Any]:
        """Generate thumbnail for a video."""
        params = {"frame_number": frame_number}
        return self._make_request("POST", f"/api/v1/video/{file_id}/thumbnail", params=params)
    
    def get_thumbnail(self, file_id: str) -> Dict[str, Any]:
        """Get video thumbnail."""
        return self._make_request("GET", f"/api/v1/video/{file_id}/thumbnail")
    
    # Search Methods
    def search_videos(self, query: str, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Search videos by various criteria."""
        data = {"query": query, "filters": filters or {}}
        return self._make_request("POST", "/api/v1/search/videos", json=data)
    
    def search_detections(self, query: str, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Search detections by various criteria."""
        data = {"query": query, "filters": filters or {}}
        return self._make_request("POST", "/api/v1/search/detections", json=data)
    
    # Notification Methods
    def get_notifications(self) -> Dict[str, Any]:
        """Get system notifications."""
        return self._make_request("GET", "/api/v1/notifications")
    
    def mark_notification_read(self, notification_id: str) -> Dict[str, Any]:
        """Mark a notification as read."""
        return self._make_request("POST", f"/api/v1/notifications/{notification_id}/read")
    
    # Backup Methods
    def create_backup(self, include_videos: bool = False) -> Dict[str, Any]:
        """Create system backup."""
        data = {"include_videos": include_videos}
        return self._make_request("POST", "/api/v1/backup/create", json=data)
    
    def get_backup_status(self, backup_id: str) -> Dict[str, Any]:
        """Get backup status."""
        return self._make_request("GET", f"/api/v1/backup/status/{backup_id}")
    
    def restore_backup(self, backup_id: str) -> Dict[str, Any]:
        """Restore from backup."""
        return self._make_request("POST", f"/api/v1/backup/restore/{backup_id}")
    
    # User Management Methods (if multi-user)
    def get_user_info(self) -> Dict[str, Any]:
        """Get current user information."""
        return self._make_request("GET", "/api/v1/user/info")
    
    def update_user_preferences(self, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Update user preferences."""
        return self._make_request("POST", "/api/v1/user/preferences", json=preferences)
    
    # WebSocket Methods (for real-time updates)
    def get_websocket_url(self) -> str:
        """Get WebSocket URL for real-time updates."""
        return self.base_url.replace("http", "ws") + "/ws"
    
    # Testing Methods
    def run_system_test(self) -> Dict[str, Any]:
        """Run system self-test."""
        return self._make_request("POST", "/api/v1/test/system")
    
    def test_detection_model(self) -> Dict[str, Any]:
        """Test detection model performance."""
        return self._make_request("POST", "/api/v1/test/model")
    
    # Context Manager Support
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()


# Singleton instance for global use
api_client = APIClient()


# Helper functions for common operations
def check_api_connection() -> bool:
    """Check if API is available."""
    try:
        response = api_client.health_check()
        return response.get("status") == "healthy"  # Updated to check for correct response
    except Exception:
        return False


def get_api_status() -> Dict[str, Any]:
    """Get comprehensive API status."""
    try:
        health = api_client.health_check()
        
        return {
            "connected": health.get("status") == "healthy",
            "version": health.get("version", "1.0.0"),
            "status": health.get("status", "Unknown"),
            "message": health.get("message", "")
        }
    except Exception as e:
        return {
            "connected": False,
            "error": str(e)
        }


def handle_api_error(response: Dict[str, Any], default_message: str = "API request failed") -> str:
    """Handle API error responses consistently."""
    if response.get("success"):
        return ""
    
    error = response.get("error", default_message)
    
    # Handle common error types
    if "connection" in error.lower():
        return "Cannot connect to the backend server. Please check if the server is running."
    elif "timeout" in error.lower():
        return "Request timed out. The server might be busy or unresponsive."
    elif "not found" in error.lower():
        return "The requested resource was not found."
    elif "unauthorized" in error.lower():
        return "Authentication required or invalid credentials."
    elif "forbidden" in error.lower():
        return "You don't have permission to access this resource."
    else:
        return error