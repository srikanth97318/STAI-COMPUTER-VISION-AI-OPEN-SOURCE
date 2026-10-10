"""Lightweight pose estimation using YOLO detections and geometric heuristics.

This module provides pose-related features without requiring PyTorch or additional
model files. It works with the existing OpenCV DNN YOLO detector and is designed
for CPU efficiency on laptops with limited RAM.

Note: This provides "lightweight" pose features rather than full body keypoint
detection. For full pose estimation, consider integrating MediaPipe or a pose
ONNX model when hardware permits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np


@dataclass
class Keypoint:
    """A single body keypoint with coordinates and confidence."""
    x: float
    y: float
    confidence: float


@dataclass
class PoseLandmarks:
    """Detected body landmarks/keypoints."""
    # Core keypoints: nose, shoulders, elbows, wrists, hips, knees, ankles
    nose: Optional[Keypoint] = None
    left_shoulder: Optional[Keypoint] = None
    right_shoulder: Optional[Keypoint] = None
    left_elbow: Optional[Keypoint] = None
    right_elbow: Optional[Keypoint] = None
    left_wrist: Optional[Keypoint] = None
    right_wrist: Optional[Keypoint] = None
    left_hip: Optional[Keypoint] = None
    right_hip: Optional[Keypoint] = None
    left_knee: Optional[Keypoint] = None
    right_knee: Optional[Keypoint] = None
    left_ankle: Optional[Keypoint] = None
    right_ankle: Optional[Keypoint] = None

    # Derived convenience properties
    shoulders: Tuple[Keypoint, Keypoint] = field(init=False, default=None)
    hips: Tuple[Keypoint, Keypoint] = field(init=False, default=None)

    def __post_init__(self):
        self.shoulders = (self.left_shoulder, self.right_shoulder)
        self.hips = (self.left_hip, self.right_hip)


@dataclass
class PoseEstimate:
    """Result of pose estimation for a single person."""
    person_id: int
    bounding_box: Tuple[float, float, float, float]  # x1, y1, x2, y2
    detection_confidence: float
    landmarks: PoseLandmarks
    pose_quality: float  # 0.0 = poor, 1.0 = good
    timestamp: float


class LightweightPoseEstimator:
    """Estimate pose features from YOLO detections using geometric heuristics.

    This estimator does NOT require PyTorch, MediaPipe, or additional model files.
    It derives pose features from the person bounding box and frame differences,
    providing enough information for temporal action recognition while being
    substantially more efficient than full pose models.
    """

    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold

    def estimate_from_detection(
        self,
        bbox: Tuple[float, float, float, float],
        frame: np.ndarray,
        timestamp: float = 0.0,
        person_id: int = 0,
    ) -> PoseEstimate:
        """Estimate pose from a YOLO detection bounding box.

        Args:
            bbox: (x1, y1, x2, y2) in pixel coordinates
            frame: The full video frame (numpy array)
            timestamp: Frame timestamp in seconds
            person_id: Unique identifier for this person track

        Returns:
            PoseEstimate with derived landmarks and quality score
        """
        x1, y1, x2, y2 = bbox
        frame_height, frame_width = frame.shape[:2]

        # Compute box dimensions
        box_width = max(x2 - x1, 1)
        box_height = max(y2 - y1, 1)
        box_center_x = (x1 + x2) / 2
        box_center_y = (y1 + y2) / 2

        # Derive keypoints using geometric heuristics:
        # - Shoulders: upper portion of the box, centered
        # - Hips: lower portion of the box, centered
        # - Elbows/wrists: extended from shoulders
        # - Knees/ankles: extended from hips

        # Shoulder width is typically ~0.6x body width, positioned at ~0.45 box height
        shoulder_width_ratio = 0.6
        shoulder_height_ratio = 0.45

        shoulder_half_width = box_width * shoulder_width_ratio / 2
        shoulder_y = box_height * shoulder_height_ratio

        # Left shoulder
        l_shoulder_x = box_center_x - shoulder_half_width
        l_shoulder_y = shoulder_y
        left_shoulder = Keypoint(
            x=max(0, min(l_shoulder_x, frame_width)),
            y=max(0, min(l_shoulder_y, frame_height)),
            confidence=0.7,  # Moderate confidence for heuristic-derived
        )

        # Right shoulder
        r_shoulder_x = box_center_x + shoulder_half_width
        r_shoulder_y = shoulder_y
        right_shoulder = Keypoint(
            x=max(0, min(r_shoulder_x, frame_width)),
            y=max(0, min(r_shoulder_y, frame_height)),
            confidence=0.7,
        )

        # Hips: wider than shoulders, at ~0.75 box height
        hip_height_ratio = 0.75
        hip_width_ratio = 0.8
        hip_half_width = box_width * hip_width_ratio / 2
        hip_y = box_height * hip_height_ratio

        # Left hip
        l_hip_x = box_center_x - hip_half_width
        l_hip_y = hip_y
        left_hip = Keypoint(
            x=max(0, min(l_hip_x, frame_width)),
            y=max(0, min(l_hip_y, frame_height)),
            confidence=0.7,
        )

        # Right hip
        r_hip_x = box_center_x + hip_half_width
        r_hip_y = hip_y
        right_hip = Keypoint(
            x=max(0, min(r_hip_x, frame_width)),
            y=max(0, min(r_hip_y, frame_height)),
            confidence=0.7,
        )

        # Elbows: extended from shoulders, roughly at shoulder y + 0.3*box_height
        elbow_y = shoulder_y + box_height * 0.3
        elbow_half_width = shoulder_half_width * 0.8

        left_elbow = Keypoint(
            x=max(0, min(box_center_x - elbow_half_width, frame_width)),
            y=max(0, min(elbow_y, frame_height)),
            confidence=0.6,
        )
        right_elbow = Keypoint(
            x=max(0, min(box_center_x + elbow_half_width, frame_width)),
            y=max(0, min(elbow_y, frame_height)),
            confidence=0.6,
        )

        # Wrists: further extended from elbows
        wrist_y = elbow_y + box_height * 0.3
        wrist_half_width = elbow_half_width * 1.2

        left_wrist = Keypoint(
            x=max(0, min(box_center_x - wrist_half_width, frame_width)),
            y=max(0, min(wrist_y, frame_height)),
            confidence=0.5,
        )
        right_wrist = Keypoint(
            x=max(0, min(box_center_x + wrist_half_width, frame_width)),
            y=max(0, min(wrist_y, frame_height)),
            confidence=0.5,
        )

        # Knees: below hips, at ~0.95 box height
        knee_y = box_height * 0.95
        knee_half_width = hip_half_width * 0.7

        left_knee = Keypoint(
            x=max(0, min(box_center_x - knee_half_width, frame_width)),
            y=max(0, min(knee_y, frame_height)),
            confidence=0.6,
        )
        right_knee = Keypoint(
            x=max(0, min(box_center_x + knee_half_width, frame_width)),
            y=max(0, min(knee_y, frame_height)),
            confidence=0.6,
        )

        # Ankles: below knees
        ankle_y = box_height * 1.1
        ankle_half_width = knee_half_width * 1.1

        left_ankle = Keypoint(
            x=max(0, min(box_center_x - ankle_half_width, frame_width)),
            y=max(0, min(ankle_y, frame_height)),
            confidence=0.5,
        )
        right_ankle = Keypoint(
            x=max(0, min(box_center_x + ankle_half_width, frame_width)),
            y=max(0, min(ankle_y, frame_height)),
            confidence=0.5,
        )

        # Nose: at the top of the bounding box, centered
        nose_x = box_center_x
        nose_y = y1 + box_height * 0.1
        nose = Keypoint(
            x=max(0, min(nose_x, frame_width)),
            y=max(0, min(nose_y, frame_height)),
            confidence=0.6,
        )

        # Build landmarks
        landmarks = PoseLandmarks(
            nose=keypoint_or_none(nose),
            left_shoulder=left_shoulder,
            right_shoulder=right_shoulder,
            left_elbow=left_elbow,
            right_elbow=right_elbow,
            left_wrist=left_wrist,
            right_wrist=right_wrist,
            left_hip=left_hip,
            right_hip=right_hip,
            left_knee=left_knee,
            right_knee=right_knee,
            left_ankle=left_ankle,
            right_ankle=right_ankle,
        )

        # Compute pose quality based on bounding box aspect ratio and completeness
        aspect_ratio = box_width / max(box_height, 1)
        # A typical standing person has aspect ratio ~0.5-0.7 (wider than tall)
        # Sitting/crouching has higher aspect ratio
        # Prone has very high aspect ratio
        if aspect_ratio < 1.0:
            # Taller than wide - likely standing
            pose_quality = min(1.0, 0.5 + 0.5 * (0.6 - abs(aspect_ratio - 0.6)) / 0.6)
        else:
            # Wider than tall - sitting, crouching, or prone
            pose_quality = min(1.0, 0.5 + 0.5 * (1.0 - abs(1.5 - aspect_ratio)) / 0.5)

        # Cap at reasonable values and ensure non-negative
        pose_quality = max(0.0, min(1.0, pose_quality))

        return PoseEstimate(
            person_id=person_id,
            bounding_box=bbox,
            detection_confidence=0.0,  # Will be set by the detector
            landmarks=landmarks,
            pose_quality=pose_quality,
            timestamp=timestamp,
        )


def keypoint_or_none(kp) -> Optional[object]:
    """Helper to handle None keypoints in dataclass defaults."""
    return kp


def draw_landmarks(
    frame: np.ndarray,
    landmarks: PoseLandmarks,
    color: Tuple[int, int, int] = (0, 255, 0),
    radius: int = 5,
) -> np.ndarray:
    """Draw pose landmarks on a frame.

    Args:
        frame: The video frame (numpy array, BGR format)
        landmarks: PoseLandmarks to draw
        color: BGR color tuple for drawing
        radius: Circle radius for keypoints

    Returns:
        Frame with pose landmarks drawn
    """
    import cv2

    # Draw keypoints
    keypoints = [
        ('nose', landmarks.nose),
        ('left_shoulder', landmarks.left_shoulder),
        ('right_shoulder', landmarks.right_shoulder),
        ('left_elbow', landmarks.left_elbow),
        ('right_elbow', landmarks.right_elbow),
        ('left_wrist', landmarks.left_wrist),
        ('right_wrist', landmarks.right_wrist),
        ('left_hip', landmarks.left_hip),
        ('right_hip', landmarks.right_hip),
        ('left_knee', landmarks.left_knee),
        ('right_knee', landmarks.right_knee),
        ('left_ankle', landmarks.left_ankle),
        ('right_ankle', landmarks.right_ankle),
    ]

    result = frame.copy()
    for name, kp in keypoints:
        if kp is not None and kp.confidence > 0.3:
            cx, cy = int(kp.x), int(kp.y)
            cv2.circle(result, (cx, cy), radius, color, -1)
            cv2.putText(
                result, name, (cx + 5, cy - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
            )

    # Draw connections
    connections = [
        ('nose', 'left_shoulder'),
        ('nose', 'right_shoulder'),
        ('left_shoulder', 'left_elbow'),
        ('right_shoulder', 'right_elbow'),
        ('left_elbow', 'left_wrist'),
        ('right_elbow', 'right_wrist'),
        ('left_shoulder', 'left_hip'),
        ('right_shoulder', 'right_hip'),
        ('left_hip', 'right_hip'),
        ('left_hip', 'left_knee'),
        ('right_hip', 'right_knee'),
        ('left_knee', 'left_ankle'),
        ('right_knee', 'right_ankle'),
    ]

    for p1_name, p2_name in connections:
        kp1 = getattr(landmarks, p1_name, None)
        kp2 = getattr(landmarks, p2_name, None)
        if kp1 is not None and kp2 is not None and kp1.confidence > 0.3 and kp2.confidence > 0.3:
            x1, y1 = int(kp1.x), int(kp1.y)
            x2, y2 = int(kp2.x), int(kp2.y)
            cv2.line(result, (x1, y1), (x2, y2), color, 2)

    return result


def draw_bounding_box_with_id(
    frame: np.ndarray,
    bbox: Tuple[float, float, float, float],
    person_id: int,
    confidence: float = 0.0,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Draw bounding box with person ID on the frame.

    Args:
        frame: The video frame
        bbox: (x1, y1, x2, y2) in pixel coordinates
        person_id: Track ID to display
        confidence: Detection confidence
        color: BGR color for the box

    Returns:
        Frame with bounding box and ID drawn
    """
    import cv2

    x1, y1, x2, y2 = bbox
    result = frame.copy()

    # Draw box
    cv2.rectangle(result, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    # Draw label background
    label = f"Person {person_id}"
    (label_w, label_h), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1,
    )
    cv2.rectangle(
        result,
        (int(x1), int(y1) - label_h - 8),
        (int(x1) + label_w, int(y1)),
        color,
        -1,
    )

    # Draw label text
    cv2.putText(
        result, label, (int(x1), int(y1) - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
    )

    # Draw confidence
    if confidence > 0:
        conf_text = f"Conf: {confidence:.0%}"
        (conf_w, conf_h), _ = cv2.getTextSize(
            conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1,
        )
        cv2.putText(
            result, conf_text,
            (int(x2) - conf_w, int(y1) + 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
        )

    return result