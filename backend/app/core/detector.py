import cv2
import numpy as np
from ultralytics import YOLO
import torch
import os
from typing import List, Dict, Any

from app.core.config import settings


class BikeDetector:
    """YOLOv8-based bike detector for shared bikes (Didi, Meituan, HelloRide)."""
    
    def __init__(self, model_path: str = None):
        """Initialize the bike detector with the trained YOLO model."""
        self.model_path = model_path or settings.MODEL_PATH
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.confidence_threshold = settings.CONFIDENCE_THRESHOLD
        self.iou_threshold = settings.IOU_THRESHOLD
        self.bike_classes = settings.BIKE_CLASSES
        self.load_model()
    
    def load_model(self):
        """Load the YOLO model."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            print(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def detect_bikes(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect bikes in a single frame.
        
        Args:
            frame: Input frame (numpy array)
            
        Returns:
            List of detections with format:
            [{'class': class_id, 'confidence': conf, 'bbox': [x1, y1, x2, y2], 'brand': brand_name}]
        """
        if self.model is None:
            return []
        
        try:
            # Run inference
            results = self.model(
                frame, 
                conf=self.confidence_threshold, 
                iou=self.iou_threshold,
                verbose=False
            )
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract box coordinates and confidence
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Get brand information
                        brand_info = self.bike_classes.get(
                            class_id, 
                            {"name": "Unknown", "color": [255, 255, 255]}
                        )
                        
                        detections.append({
                            "class": class_id,
                            "confidence": float(confidence),
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "brand": brand_info["name"],
                            "color": brand_info["color"],
                            "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
                            "area": int((x2 - x1) * (y2 - y1))
                        })
            
            return detections
        
        except Exception as e:
            print(f"Error during detection: {e}")
            return []
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw bounding boxes and labels on the frame.
        
        Args:
            frame: Input frame
            detections: List of detections from detect_bikes()
            
        Returns:
            Frame with drawn detections
        """
        annotated_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            confidence = detection["confidence"]
            brand = detection["brand"]
            color = tuple(detection["color"])
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Create label text
            label = f"{brand}: {confidence:.2f}"
            
            # Get text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Draw background rectangle for text
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Draw text
            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            
            # Draw center point
            center = detection["center"]
            cv2.circle(annotated_frame, tuple(center), 3, color, -1)
        
        return annotated_frame
    
    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Process a single frame: detect bikes and draw annotations.
        
        Args:
            frame: Input frame
            
        Returns:
            tuple: (annotated_frame, detections)
        """
        detections = self.detect_bikes(frame)
        annotated_frame = self.draw_detections(frame, detections)
        return annotated_frame, detections
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"error": "Model not loaded"}
        
        try:
            model_info = {
                "model_path": self.model_path,
                "device": self.device,
                "confidence_threshold": self.confidence_threshold,
                "iou_threshold": self.iou_threshold,
                "classes": self.bike_classes,
                "model_loaded": True,
                "pytorch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available()
            }
            
            if hasattr(self.model, 'names'):
                model_info["class_names"] = self.model.names
            
            return model_info
            
        except Exception as e:
            return {"error": f"Error getting model info: {str(e)}"}
    
    def update_thresholds(self, confidence: float = None, iou: float = None):
        """
        Update detection thresholds.
        
        Args:
            confidence: New confidence threshold
            iou: New IoU threshold
        """
        if confidence is not None:
            self.confidence_threshold = max(0.0, min(1.0, confidence))
        
        if iou is not None:
            self.iou_threshold = max(0.0, min(1.0, iou))
    
    def get_detection_summary(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get summary statistics for detections.
        
        Args:
            detections: List of detections
            
        Returns:
            Summary dictionary
        """
        if not detections:
            return {
                "total_detections": 0,
                "brands": {},
                "confidence_stats": {},
                "bbox_stats": {}
            }
        
        # Count by brand
        brand_counts = {}
        confidences = []
        areas = []
        
        for detection in detections:
            brand = detection["brand"]
            confidence = detection["confidence"]
            area = detection["area"]
            
            brand_counts[brand] = brand_counts.get(brand, 0) + 1
            confidences.append(confidence)
            areas.append(area)
        
        return {
            "total_detections": len(detections),
            "brands": brand_counts,
            "confidence_stats": {
                "mean": np.mean(confidences),
                "min": np.min(confidences),
                "max": np.max(confidences),
                "std": np.std(confidences)
            },
            "bbox_stats": {
                "mean_area": np.mean(areas),
                "min_area": np.min(areas),
                "max_area": np.max(areas),
                "std_area": np.std(areas)
            }
        }