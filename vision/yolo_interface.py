"""
vision/yolo_interface.py

Milestone 7: YOLOv5 Object Detection Interface.

Responsibility in the research architecture:
- YOLO answers: "What is it, and where is it?"
  Produces semantic object class, bounding box coordinates [x_c, y_c, w, h], and confidence.
- Connectome answers: "How is it moving through time?" (Direction, velocity, loom).

Supports:
1. Ground-truth target detection in simulated pursuit environments.
2. Lightweight OpenCV DNN / ONNX inference for image frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple
import numpy as np


@dataclass
class YOLODetection:
    class_id: int
    class_name: str
    confidence: float
    center_x: float     # Normalized coordinate [-1.0, +1.0] relative to visual field center
    center_y: float     # Normalized coordinate [-1.0, +1.0] relative to visual field center
    width: float        # Normalized width [0.0, 2.0]
    height: float       # Normalized height [0.0, 2.0]

    def to_feature_vector(self) -> np.ndarray:
        """Returns 5D spatial localization and confidence vector [x, y, w, h, conf]."""
        return np.array([self.center_x, self.center_y, self.width, self.height, self.confidence], dtype=float)


class YOLOInterface:
    """
    Standard interface for object detection in the Connectome-Constrained agent.
    """

    def __init__(self, target_classes: Optional[list[str]] = None):
        self.target_classes = target_classes or ["target", "prey", "predator", "obstacle"]

    def detect_from_arena_state(
        self,
        target_pos_deg: tuple[float, float],
        target_size_deg: float = 6.0,
        target_class: str = "target",
        confidence: float = 0.95,
        field_of_view_deg: float = 30.0,
    ) -> list[YOLODetection]:
        """
        Extracts detection from environment state (normalized to [-1, 1] visual coordinates).
        """
        x_norm = target_pos_deg[0] / (field_of_view_deg / 2.0)
        y_norm = target_pos_deg[1] / (field_of_view_deg / 2.0)
        w_norm = target_size_deg / field_of_view_deg
        h_norm = target_size_deg / field_of_view_deg

        # Check if target is inside visual field
        if -1.2 <= x_norm <= 1.2 and -1.2 <= y_norm <= 1.2:
            det = YOLODetection(
                class_id=0,
                class_name=target_class,
                confidence=confidence,
                center_x=float(np.clip(x_norm, -1.0, 1.0)),
                center_y=float(np.clip(y_norm, -1.0, 1.0)),
                width=float(w_norm),
                height=float(h_norm),
            )
            return [det]
        return []

    def detect_from_image(self, frame: np.ndarray) -> list[YOLODetection]:
        """
        Lightweight threshold/contour detector for synthetic frames (fallback if no ONNX model provided).
        """
        if frame.ndim == 3:
            gray = np.mean(frame, axis=2).astype(np.uint8)
        else:
            gray = frame.astype(np.uint8)

        # Simple bright blob detection
        thresh = gray > 128
        y_indices, x_indices = np.where(thresh)

        if len(x_indices) == 0:
            return []

        h, w = gray.shape
        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()

        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        box_w = max(1, x_max - x_min)
        box_h = max(1, y_max - y_min)

        # Normalize to [-1, 1]
        cx_norm = (cx / (w / 2.0)) - 1.0
        cy_norm = 1.0 - (cy / (h / 2.0))  # Flip y so Up is +1
        w_norm = box_w / (w / 2.0)
        h_norm = box_h / (h / 2.0)

        return [
            YOLODetection(
                class_id=0,
                class_name="target",
                confidence=0.92,
                center_x=float(cx_norm),
                center_y=float(cy_norm),
                width=float(w_norm),
                height=float(h_norm),
            )
        ]
