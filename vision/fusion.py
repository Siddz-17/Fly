"""
vision/fusion.py

Milestone 8: Connectome + YOLO Fusion Layer.

Integrates:
- YOLOv5: "What is it, and where is it?" (Semantic identity + spatial coordinates)
- Male Fly Connectome: "How is it moving through time?" (Directional velocity + looming)

Produces the fused state vector s_t for behavioral policy training (PPO) in the pursuit arena.
Supports controlled ablations: Full Fusion vs YOLO-only vs Connectome-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np

from vision.embeddings import MotionEmbedding
from vision.yolo_interface import YOLODetection


class FusionMode(str, Enum):
    FULL = "full"                          # Connectome + YOLO
    CONNECTOME_ONLY = "connectome_only"    # Connectome motion only (no YOLO semantics)
    YOLO_ONLY = "yolo_only"                # YOLO object bounding box only (no connectome motion)
    RANDOM_BASELINE = "random_baseline"    # Random feature baseline


@dataclass
class FusedAgentState:
    mode: FusionMode
    features: np.ndarray          # 1D feature vector for reinforcement learning / controller
    target_visible: bool
    target_pos: tuple[float, float]
    target_velocity: tuple[float, float]
    speed: float
    is_looming: bool

    def describe(self) -> str:
        if not self.target_visible:
            return f"[{self.mode.value}] No target detected. Background optical flow: ({self.target_velocity[0]:+.2f}, {self.target_velocity[1]:+.2f})"
        loom_str = "LOOMING/EXPANDING" if self.is_looming else "receding/lateral"
        return (
            f"[{self.mode.value}] Target at ({self.target_pos[0]:+.2f}, {self.target_pos[1]:+.2f}) | "
            f"Connectome Velocity: ({self.target_velocity[0]:+.2f}, {self.target_velocity[1]:+.2f}) | "
            f"Speed: {self.speed:.2f} | Approach: {loom_str}"
        )


class StateFusionLayer:
    """
    Fuses YOLO object detections with biological connectome motion embeddings.
    """

    def __init__(self, mode: FusionMode = FusionMode.FULL):
        self.mode = mode

    def fuse(
        self,
        yolo_detections: list[YOLODetection],
        motion_embedding: MotionEmbedding,
    ) -> FusedAgentState:
        """
        Produce a fused state representation from YOLO and Connectome inputs.
        """
        has_target = len(yolo_detections) > 0
        if has_target:
            best_det = max(yolo_detections, key=lambda d: d.confidence)
            yolo_vec = best_det.to_feature_vector()  # (5D: [x, y, w, h, conf])
            target_pos = (best_det.center_x, best_det.center_y)
        else:
            yolo_vec = np.zeros(5, dtype=float)
            target_pos = (0.0, 0.0)

        motion_vec = motion_embedding.to_feature_vector()  # (6D: [vx, vy, speed, loom, on_pow, off_pow])
        target_velocity = (motion_embedding.vx, motion_embedding.vy)
        is_looming = motion_embedding.loom > 0.05

        # Cross-modal interaction feature: Alignment between target position and motion vector
        # (Is the object moving towards the center of gaze?)
        if has_target and motion_embedding.speed > 1e-4:
            pos_norm = np.array(target_pos)
            vel_norm = np.array(target_velocity) / (motion_embedding.speed + 1e-6)
            alignment = float(np.dot(pos_norm, vel_norm))
        else:
            alignment = 0.0

        # Construct feature vector based on ablation mode
        if self.mode == FusionMode.FULL:
            # 12-dimensional fused representation: [YOLO(5D), Connectome(6D), Alignment(1D)]
            features = np.concatenate([yolo_vec, motion_vec, [alignment]])

        elif self.mode == FusionMode.CONNECTOME_ONLY:
            # Mask out YOLO spatial coordinates and confidence
            features = np.concatenate([np.zeros(5), motion_vec, [0.0]])

        elif self.mode == FusionMode.YOLO_ONLY:
            # Mask out Connectome motion features
            features = np.concatenate([yolo_vec, np.zeros(6), [0.0]])

        elif self.mode == FusionMode.RANDOM_BASELINE:
            features = np.random.randn(12)

        else:
            raise ValueError(f"Unknown fusion mode: {self.mode}")

        return FusedAgentState(
            mode=self.mode,
            features=features,
            target_visible=has_target,
            target_pos=target_pos,
            target_velocity=target_velocity,
            speed=motion_embedding.speed,
            is_looming=is_looming,
        )
