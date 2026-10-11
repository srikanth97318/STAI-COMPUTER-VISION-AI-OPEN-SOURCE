"""Lightweight multi-person tracking for STAI.

This module provides person tracking without requiring external dependencies like
deep-sort or byte-track. It is designed to work with the existing YOLO detection
pipeline and pose estimation module, running entirely on CPU.

Track IDs are persistent within a video stream but do NOT represent verified
real-world identities. IDs reset when the stream restarts or changes.

The tracker uses a simple IOU-based matching with a history buffer to maintain
identities across frames, even through temporary occlusions.

Deduplication is applied BEFORE tracking to prevent duplicate detections
from creating separate tracks for the same person.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
import numpy as np


def _next_track_id(tracks: Dict[int, TrackedPerson], last_id: int) -> int:
    """Get the next track ID, reusing recycled IDs from removed tracks first."""
    # Collect IDs of tracks that were removed (present in last_id tracking but not in current)
    # Strategy: find the highest currently-used ID, then reuse IDs below it that are free
    if not tracks:
        return max(last_id, 1)
    max_used = max(tracks.keys())
    # Check if any IDs from 1..max_used are available (not currently in use)
    used_set = set(tracks.keys())
    for i in range(1, max_used + 1):
        if i not in used_set:
            return i
    # All IDs 1..max_used are in use, assign a new one
    return max_used + 1


@dataclass
class TrackedPerson:
    """A person tracked across consecutive frames.

    Attributes:
        track_id: Persistent identifier for this track (within this video stream).
        bbox: Current bounding box (x1, y1, x2, y2) in pixel coordinates.
        age: Number of frames this track has been active.
        time_since_update: Frames since the track was last matched to a detection.
        history: Bounding box history for trajectory prediction.
        pose_estimate: Latest pose estimate associated with this track.
        pose_quality_history: History of pose quality scores.
        status: Current track status ("active", "lost", "removed").
    """
    track_id: int
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)
    age: int = 0
    time_since_update: int = 0
    history: deque = field(default_factory=lambda: deque(maxlen=5))
    pose_estimate: Optional[object] = None
    pose_quality_history: deque = field(default_factory=lambda: deque(maxlen=5))
    status: str = "active"


def iou(box_a: Tuple[float, float, float, float],
        box_b: Tuple[float, float, float, float]) -> float:
    """Compute Intersection over Union (IOU) between two bounding boxes.

    Args:
        box_a: (x1, y1, x2, y2) in pixel coordinates
        box_b: (x1, y1, x2, y2) in pixel coordinates

    Returns:
        IOU value in range [0, 1]. Higher means more overlap.
    """
    x1_a, y1_a, x2_a, y2_a = box_a
    x1_b, y1_b, x2_b, y2_b = box_b

    # Calculate intersection
    inter_x1 = max(x1_a, x1_b)
    inter_y1 = max(y1_a, y1_b)
    inter_x2 = min(x2_a, x2_b)
    inter_y2 = min(y2_a, y2_b)

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    # Calculate union
    area_a = max(0, x2_a - x1_a) * max(0, y2_a - y1_a)
    area_b = max(0, x2_b - x1_b) * max(0, y2_b - y1_b)
    union_area = area_a + area_b - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def deduplicate_detections(
    bboxes: List[Tuple[float, float, float, float, str, Tuple[int, int, int]]],
    iou_threshold: float = 0.5,
) -> List[Tuple[float, float, float, float, str, Tuple[int, int, int]]]:
    """Remove overlapping bounding boxes for the same person.

    This function prevents the tracker from creating separate tracks for
    multiple detections of the same person. It keeps the detection with
    larger area (typically higher confidence) when two boxes have high IoU.

    The function preserves the full 6-element tuple format:
    (x1, y1, x2, y2, label, color)

    Args:
        bboxes: List of (x1, y1, x2, y2, label, color) tuples.
        iou_threshold: Boxes with IoU above this value are considered duplicates.

    Returns:
        Deduplicated list of 6-element bounding box tuples (x1, y1, x2, y2, label, color).
    """
    if not bboxes:
        return []

    # Sort by box area (larger boxes first - typically higher confidence)
    bboxes_sorted = sorted(bboxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)

    kept: List[Tuple[float, float, float, float, str, Tuple[int, int, int]]] = []
    for bbox in bboxes_sorted:
        x1, y1, x2, y2, label, color = bbox
        is_duplicate = False
        for kept_bbox in kept:
            kx1, ky1, kx2, ky2, klabel, kcolor = kept_bbox
            iou_val = iou((x1, y1, x2, y2), (kx1, ky1, kx2, ky2))
            if iou_val > iou_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(bbox)

    return kept


class Tracker:
    """Lightweight multi-person tracker.

    Matches detections to existing tracks using IOU, assigns new track IDs for
    unmatched detections, and ages out tracks that haven't been matched for
    too long.

    Designed for CPU efficiency on laptops with limited RAM. Track IDs are
    stable within a single video stream but not globally unique across streams.

    Example:
        >>> tracker = Tracker(max_age=30, min_hits=1)
        >>> tracks = tracker.update(detections, pose_estimates, frame_timestamp)
    """

    def __init__(self,
                 max_age: int = 30,
                 min_hits: int = 1,
                 iou_threshold: float = 0.3):
        """Initialize the tracker.

        Args:
            max_age: Maximum frames a track can remain without a match before
                being removed.
            min_hits: Minimum number of detections required before a track is
                considered "confirmed" (not used in this lightweight implementation,
                but provided for API compatibility).
            iou_threshold: Minimum IOU for a detection to match an existing track.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.track_count = 0
        self.tracks: Dict[int, TrackedPerson] = {}

    def update(self,
               bboxes: List[Tuple[float, float, float, float]],
               pose_estimates: List[object],
               timestamp: float) -> Dict[int, TrackedPerson]:
        """Update tracks with new detections and pose estimates.

        Args:
            bboxes: List of bounding boxes (x1, y1, x2, y2) from YOLO detection.
            pose_estimates: List of pose estimates from the pose estimator.
            timestamp: Current frame timestamp in seconds.

        Returns:
            Dictionary of track_id -> TrackedPerson for active tracks.
        """
        # Step 1: Get existing track bounding boxes
        existing_tracks = list(self.tracks.values())

        if len(existing_tracks) == 0 and len(bboxes) == 0:
            # No tracks and no detections
            # Age out existing tracks
            for track in self.tracks.values():
                track.age += 1
                track.time_since_update += 1
                track.status = "lost" if track.time_since_update > self.max_age // 2 else "active"
            # Remove old tracks
            self.tracks = {
                tid: t for tid, t in self.tracks.items()
                if t.time_since_update <= self.max_age
            }
            return {tid: t for tid, t in self.tracks.items() if t.status == "active"}

        if len(existing_tracks) == 0:
            # No existing tracks, create new ones from detections
            for i, bbox in enumerate(bboxes):
                track_id = self._generate_track_id()
                self.tracks[track_id] = TrackedPerson(
                    track_id=track_id,
                    bbox=bbox,
                    age=1,
                    time_since_update=0,
                    pose_estimate=pose_estimates[i] if i < len(pose_estimates) else None,
                    pose_quality_history=[0.0],  # placeholder
                    status="active",
                )
            self.track_count = max(self.track_count, max(self.tracks.keys()) + 1 if self.tracks else 0)
            return {tid: t for tid, t in self.tracks.items() if t.status == "active"}

        # Step 2: Compute IOU matrix between existing tracks and new detections
        track_boxes = [t.bbox for t in existing_tracks]
        det_count = len(bboxes)
        track_count = len(existing_tracks)

        iou_matrix = np.zeros((track_count, det_count))
        for t_idx, track in enumerate(existing_tracks):
            for d_idx, bbox in enumerate(bboxes):
                iou_matrix[t_idx, d_idx] = iou(track.bbox, bbox)

        # Step 3: Greedy matching - match each track to the best detection
        # Use linear assignment logic: match highest IOU above threshold
        matched_track_indices = set()
        matched_det_indices = set()

        # Sort matches by IOU descending
        flat_indices = np.argwhere(iou_matrix >= self.iou_threshold)
        if len(flat_indices) > 0:
            # Sort by IOU value descending
            flat_indices = flat_indices[np.argsort(-iou_matrix[flat_indices[:, 0], flat_indices[:, 1]])]

            for t_idx, d_idx in flat_indices:
                if t_idx not in matched_track_indices and d_idx not in matched_det_indices:
                    # Match this track to this detection
                    matched_track_indices.add(t_idx)
                    matched_det_indices.add(d_idx)

                    # Update the track
                    track = existing_tracks[t_idx]
                    track.bbox = bboxes[d_idx]
                    track.age += 1
                    track.time_since_update = 0

                    # Update pose estimate if available
                    if d_idx < len(pose_estimates) and pose_estimates[d_idx] is not None:
                        track.pose_estimate = pose_estimates[d_idx]

                    # Update pose quality history
                    if d_idx < len(pose_estimates) and pose_estimates[d_idx] is not None:
                        # pose_quality is stored in the PoseEstimate, not in PoseLandmarks
                        q = pose_estimates[d_idx].pose_quality
                        if q is not None:
                            track.pose_quality_history.append(q)

        # Step 4: Update time_since_update for unmatched tracks
        for t_idx, track in enumerate(existing_tracks):
            if t_idx not in matched_track_indices:
                track.time_since_update += 1
                track.age += 1
                track.status = "lost" if track.time_since_update > self.max_age // 2 else "active"

        # Step 5: Create new tracks for unmatched detections
        for d_idx, bbox in enumerate(bboxes):
            if d_idx not in matched_det_indices:
                track_id = self._generate_track_id()
                self.tracks[track_id] = TrackedPerson(
                    track_id=track_id,
                    bbox=bbox,
                    age=1,
                    time_since_update=0,
                    pose_estimate=pose_estimates[d_idx] if d_idx < len(pose_estimates) else None,
                    pose_quality_history=[0.0],
                    status="active",
                )

        # Step 6: Age out and remove old tracks
        # Use strict < so a track is removed when time_since_update >= max_age
        self.tracks = {
            tid: t for tid, t in self.tracks.items()
            if t.time_since_update < self.max_age
        }

        # Increment track count for new IDs
        if self.tracks:
            self.track_count = max(self.track_count, max(self.tracks.keys()) + 1)
        else:
            self.track_count = 0

        return {tid: t for tid, t in self.tracks.items() if t.status == "active"}

    def _generate_track_id(self) -> int:
        """Generate a new unique track ID, reusing recycled IDs from removed tracks."""
        return _next_track_id(self.tracks, self.track_count)


# Convenience function for use in detect_final.py
def update_tracks(tracker: Tracker,
                  bboxes: List[Tuple[float, float, float, float]],
                  pose_estimates: List[object],
                  timestamp: float) -> Tuple[Dict[int, TrackedPerson], list]:
    """Update tracker and return formatted results for dashboard display.

    Args:
        tracker: Initialized Tracker instance.
        bboxes: List of YOLO detection bounding boxes.
        pose_estimates: List of pose estimates.
        timestamp: Current frame timestamp.

    Returns:
        Tuple of (active_tracks_dict, display_info_list).
        display_info_list contains dicts with: track_id, person_label, pose_available, quality.
    """
    active_tracks = tracker.update(bboxes, pose_estimates, timestamp)

    display_info = []
    for track_id, track in active_tracks.items():
        pose_available = track.pose_estimate is not None
        quality = "unknown"
        if pose_available and hasattr(track.pose_estimate, 'landmarks'):
            lp = track.pose_estimate.landmarks
            if hasattr(lp, 'pose_quality') and lp.pose_quality is not None:
                q = lp.pose_quality
                if q >= 0.7:
                    quality = "good"
                elif q >= 0.4:
                    quality = "fair"
                else:
                    quality = "poor"
            else:
                quality = "unknown"
        elif pose_available:
            quality = "unknown"

        display_info.append({
            "track_id": track_id,
            "person_label": f"Person #{track_id}",
            "pose_available": pose_available,
            "quality": quality,
            "age": track.age,
        })

    return active_tracks, display_info