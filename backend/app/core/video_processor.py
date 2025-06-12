import cv2
import os
import tempfile
import numpy as np
from .detector import BikeDetector
# from .config import OUTPUTS_DIR, UPLOADS_DIR
from .config import settings
from typing import Generator, Tuple, Dict, List, Optional, Callable
import time

UPLOADS_DIR = settings.upload_dir
OUTPUTS_DIR = settings.output_dir
class VideoProcessor:
    def __init__(self):
        """Initialize the video processor with bike detector."""
        self.detector = BikeDetector()
    
    def get_video_info(self, video_path: str) -> Optional[Dict]:
        """
        Get basic information about the video.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            dict: Video information (fps, frame_count, duration, resolution)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            return {
                'fps': fps,
                'frame_count': frame_count,
                'duration': duration,
                'width': width,
                'height': height,
                'resolution': f"{width}x{height}"
            }
        finally:
            cap.release()
    
    def process_video_realtime(self, video_path: str, progress_callback: Optional[Callable] = None, 
                              frame_skip: int = 1) -> Generator[Tuple[int, np.ndarray, List[Dict], float], None, None]:
        """
        Generator function for real-time video processing.
        Yields processed frames one by one.
        
        Args:
            video_path: Path to the input video
            progress_callback: Optional callback function for progress updates
            frame_skip: Skip every N frames for faster processing
            
        Yields:
            tuple: (frame_number, annotated_frame, detections, progress_percentage)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_number = 0
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Skip frames if needed
                if frame_number % frame_skip != 0:
                    frame_number += 1
                    continue
                
                # Process the frame
                start_time = time.time()
                annotated_frame, detections = self.detector.process_frame(frame)
                processing_time = time.time() - start_time
                
                # Calculate progress
                progress = (frame_number / total_frames) * 100 if total_frames > 0 else 0
                
                # Call progress callback if provided
                if progress_callback:
                    progress_callback(progress)
                
                yield frame_number, annotated_frame, detections, progress
                
                frame_number += 1
        
        finally:
            cap.release()
    
    def process_video_batch(self, video_path: str, output_path: Optional[str] = None, 
                           progress_callback: Optional[Callable] = None, 
                           save_annotated: bool = True, frame_skip: int = 1) -> Tuple[str, Dict]:
        """
        Process entire video and save the result.
        
        Args:
            video_path: Path to the input video
            output_path: Path for the output video (optional)
            progress_callback: Optional callback function for progress updates
            save_annotated: Whether to save annotated video
            frame_skip: Skip every N frames for faster processing
            
        Returns:
            tuple: (output_path, detection_statistics)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Set output path if not provided
        if output_path is None:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(OUTPUTS_DIR, f"{video_name}_processed.mp4")
        
        # Initialize video writer if saving annotated video
        out = None
        if save_annotated:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_number = 0
        detection_stats = {
            'Didi': 0, 
            'Meituan': 0, 
            'HelloRide': 0,
            'total_detections': 0,
            'frames_processed': 0,
            'confidence_sum': 0,
            'bbox_areas': []
        }
        
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Skip frames if needed
                if frame_number % frame_skip != 0:
                    frame_number += 1
                    continue
                
                # Process the frame
                annotated_frame, detections = self.detector.process_frame(frame)
                detection_stats['frames_processed'] += 1
                
                # Count detections for statistics
                for detection in detections:
                    brand = detection.get('brand', 'unknown')
                    if brand in detection_stats:
                        detection_stats[brand] += 1
                    
                    detection_stats['total_detections'] += 1
                    detection_stats['confidence_sum'] += detection.get('confidence', 0)
                    
                    # Calculate bbox area
                    bbox = detection.get('bbox', [0, 0, 0, 0])
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    detection_stats['bbox_areas'].append(area)
                
                # Write the frame if saving
                if save_annotated and out is not None:
                    out.write(annotated_frame)
                
                # Calculate progress
                progress = (frame_number / total_frames) * 100 if total_frames > 0 else 0
                
                # Call progress callback if provided
                if progress_callback:
                    progress_callback(progress)
                
                frame_number += 1
        
        finally:
            cap.release()
            if out is not None:
                out.release()
        
        # Calculate final statistics
        processing_time = time.time() - start_time
        
        # Calculate average confidence
        avg_confidence = (detection_stats['confidence_sum'] / detection_stats['total_detections'] 
                         if detection_stats['total_detections'] > 0 else 0)
        
        # Calculate bbox statistics
        bbox_areas = detection_stats['bbox_areas']
        avg_bbox_area = sum(bbox_areas) / len(bbox_areas) if bbox_areas else 0
        
        final_stats = {
            'total_detections': detection_stats['total_detections'],
            'brands': {k: v for k, v in detection_stats.items() 
                      if k in ['Didi', 'Meituan', 'HelloRide']},
            'confidence_stats': {
                'average': avg_confidence,
                'count': detection_stats['total_detections']
            },
            'bbox_stats': {
                'average_area': avg_bbox_area,
                'count': len(bbox_areas)
            },
            'frames_processed': detection_stats['frames_processed'],
            'processing_time': processing_time
        }
        
        return output_path if save_annotated else None, final_stats
    
    def extract_frame(self, video_path: str, frame_number: int) -> Optional[np.ndarray]:
        """
        Extract a specific frame from the video.
        
        Args:
            video_path: Path to the video file
            frame_number: Frame number to extract
            
        Returns:
            numpy.ndarray: Extracted frame or None if failed
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None
        
        try:
            # Set the frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            return frame if ret else None
        finally:
            cap.release()
    
    def create_thumbnail(self, video_path: str, output_path: Optional[str] = None, 
                        timestamp: float = 1.0) -> Optional[str]:
        """
        Create a thumbnail from the video at a specific timestamp.
        
        Args:
            video_path: Path to the video file
            output_path: Path for the thumbnail (optional)
            timestamp: Timestamp in seconds for thumbnail extraction
            
        Returns:
            str: Path to the created thumbnail
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None
        
        try:
            # Set the timestamp
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ret, frame = cap.read()
            
            if not ret:
                return None
            
            # Set output path if not provided
            if output_path is None:
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                output_path = os.path.join(OUTPUTS_DIR, f"{video_name}_thumbnail.jpg")
            
            # Save the thumbnail
            cv2.imwrite(output_path, frame)
            
            return output_path
        finally:
            cap.release()
    
    def process_single_frame(self, video_path: str, frame_number: int) -> Optional[Tuple[np.ndarray, List[Dict]]]:
        """
        Process a single frame from the video.
        
        Args:
            video_path: Path to the video file
            frame_number: Frame number to process
            
        Returns:
            tuple: (annotated_frame, detections) or None if failed
        """
        frame = self.extract_frame(video_path, frame_number)
        if frame is None:
            return None
        
        return self.detector.process_frame(frame)