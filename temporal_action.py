"""Temporal action recognition for STAI.

Analyzes a bounded sliding window of recent observations per track to infer
activities such as standing, sitting, walking, falling, etc. Designed for
CPU efficiency on laptops with limited RAM. Works with the existing tracking
and pose estimation modules.

Activity classes (in order of severity):
- STANDING: Person is upright, likely walking or standing still
- SITTING: Person's hips are higher than knees, torso relatively vertical
- WALKING: Forward center-of-mass displacement with alternating pose patterns
- RUNNING: Similar to walking but with higher velocity and larger displacement
- BENDING: torso inclination forward, knees may be flexed
- LYING_DOWN: Person's full height compressed, likely recumbent
- TRANSITIONING: Posture changing between standing and sitting (or vice versa)
- POSSIBLE_FALL: Rapid downward transition followed by low posture (not confirmed)
- UNKNOWN / UNCERTAIN: Insufficient evidence for any other class

Fall detection is a temporal event detector, NOT a rule that labels every
lying person as having fallen. A person lying down or exercising must not
automatically be classified as a fall. If evidence is insufficient,
report: 'Possible fall — requires review'
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Deque, Dict, Optional, Tuple, List
import numpy as np


class ActivityCategory(Enum):
    """Inferred activity categories."""
    STANDING = auto()
    SITTING = auto()
    WALKING = auto()
    RUNNING = auto()
    BENDING = auto()
    LYING_DOWN = auto()
    TRANSITIONING = auto()
    POSSIBLE_FALL = auto()
    UNKNOWN = auto()
    UNCERTAIN = auto()


# Reverse mapping for string conversion
_ACTIVITY_NAMES = {
    ActivityCategory.STANDING: "STANDING",
    ActivityCategory.SITTING: "SITTING",
    ActivityCategory.WALKING: "WALKING",
    ActivityCategory.RUNNING: "RUNNING",
    ActivityCategory.BENDING: "BENDING",
    ActivityCategory.LYING_DOWN: "LYING_DOWN",
    ActivityCategory.TRANSITIONING: "TRANSITIONING",
    ActivityCategory.POSSIBLE_FALL: "POSSIBLE_FALL",
    ActivityCategory.UNKNOWN: "UNKNOWN",
    ActivityCategory.UNCERTAIN: "UNCERTAIN",
}


def activity_to_string(cat: ActivityCategory) -> str:
    """Convert ActivityCategory to human-readable string."""
    return _ACTIVITY_NAMES.get(cat, "UNKNOWN")


def activity_from_string(s: str) -> ActivityCategory:
    """Convert string to ActivityCategory."""
    mapping = {v: k for k, v in _ACTIVITY_NAMES.items()}
    return mapping.get(s, ActivityCategory.UNKNOWN)


@dataclass
class ActivityObservation:
    """A single observation of activity from one frame/timestep.

    Contains the temporal features extracted from the track's history at
    this point in time.
    """
    timestamp: float
    # Temporal features
    center_displacement: float  # displacement from window start (pixels)
    bbox_height: float  # bounding box height (pixels)
    bbox_aspect_ratio: float  # width / height
    centroid_velocity: float  # pixels per second (magnitude)
    pose_quality: float  # 0.0 = poor, 1.0 = good
    pose_visibility: float  # average keypoint visibility 0-1
    # Posture features
    shoulder_hip_distance: float  # relative vertical distance
    is_vertical_torso: bool  # whether torso is approximately vertical
    # Derived
    frame_index: int


@dataclass
class ActivityResult:
    """Result of temporal action recognition.

    Attributes:
        activity: The inferred activity category.
        confidence: Confidence score (0-1) if calibrated, otherwise heuristic evidence score.
        start_timestamp: When the current activity began (seconds).
        last_update: When the activity was last observed (seconds).
        evidence_duration: How long the current activity has been sustained (seconds).
        pose_quality: Average pose quality during this period (0-1).
        explanation: Brief text explaining supporting observations.
        status: "preliminary", "stable", or "uncertain".
    """
    activity: ActivityCategory
    confidence: float  # 0-1 evidence score (not necessarily calibrated probability)
    start_timestamp: float
    last_update: float
    evidence_duration: float
    pose_quality: float
    explanation: str
    status: str  # "preliminary", "stable", "uncertain"


@dataclass
class ActivityTrackerState:
    """Internal state for tracking activity per person track."""

    # Bounded sliding window of recent observations
    window: Deque[ActivityObservation] = field(default_factory=lambda: deque(maxlen=20))

    # Tracking state
    current_activity: ActivityCategory = ActivityCategory.UNKNOWN
    current_confidence: float = 0.0
    activity_start_time: float = 0.0
    last_activity_change: float = 0.0
    activity_history: List[Tuple[ActivityCategory, float]] = field(
        default_factory=list,
    )

    # Hysteresis / stability tracking
    previous_activity: Optional[ActivityCategory] = None
    activity_stability_counter: int = 0
    min_stable_frames: int = 3

    # Fall detection state
    fall_triggered: bool = False
    fall_rapid_downward: bool = False
    fall_start_time: Optional[float] = None
    person_was_standing: bool = False

    # Observation window configuration
    observation_window_sec: float = 5.0  # How far back to look
    min_evidence_duration_sec: float = 2.0  # Minimum time before publishing
    hysteresis_threshold: int = 3  # Frames to require for label switch


class TemporalActionRecognizer:
    """Recognizes temporal activities from track histories.

    Uses a sliding window of recent observations and temporal features to
    infer activities such as standing, sitting, walking, falling, etc.

    Key design decisions:
    - Works with lightweight pose heuristics (no PyTorch/MediaPipe required)
    - Uses real timestamps from the video pipeline
    - Hysteresis prevents rapid label switching
    - Returns UNKNOWN/UNCERTAIN when evidence is insufficient
    - Fall detection is conservative (requires specific temporal pattern)
    """

    def __init__(
        self,
        observation_window_sec: float = 5.0,
        min_evidence_duration_sec: float = 2.0,
        hysteresis_threshold: int = 3,
        min_stable_frames: int = 3,
        fall_rapid_velocity: float = 150.0,  # px/sec threshold for "rapid downward"
        fall_min_downward_distance: float = 50.0,  # px minimum downward movement
    ):
        """Initialize the recognizer.

        Args:
            observation_window_sec: How far back to look for temporal features
            min_evidence_duration_sec: Minimum time current activity must be
                sustained before publishing
            hysteresis_threshold: Frames to require for a label switch
            min_stable_frames: Minimum consecutive frames at same activity
            fall_rapid_velocity: px/sec - how fast must someone fall down
            fall_min_downward_distance: px - minimum downward movement to trigger
        """
        self.observation_window_sec = observation_window_sec
        self.min_evidence_duration_sec = min_evidence_duration_sec
        self.hysteresis_threshold = hysteresis_threshold
        self.min_stable_frames = min_stable_frames
        self.fall_rapid_velocity = fall_rapid_velocity
        self.fall_min_downward_distance = fall_min_downward_distance

        # Per-track state (will be created on demand)
        self.states: Dict[int, ActivityTrackerState] = {}

    def _get_state(self, track_id: int) -> ActivityTrackerState:
        """Get or create the activity tracker state for a track."""
        if track_id not in self.states:
            self.states[track_id] = ActivityTrackerState()
        return self.states[track_id]

    def _extract_observation(
        self,
        bbox: Tuple[float, float, float, float],
        landmarks: Optional[object],
        timestamp: float,
        frame_index: int,
        frame_width: int,
        frame_height: int,
    ) -> ActivityObservation:
        """Extract temporal features from a detection for activity recognition.

        Args:
            bbox: (x1, y1, x2, y2) in pixel coordinates
            landmarks: Pose landmarks (optional - from pose estimator)
            timestamp: Current frame timestamp in seconds
            frame_index: Index of this frame in the video
            frame_width: Width of the full frame
            frame_height: Height of the full frame

        Returns:
            ActivityObservation with extracted temporal features
        """
        x1, y1, x2, y2 = bbox
        box_width = max(x2 - x1, 1)
        box_height = max(y2 - y1, 1)
        box_center_x = (x1 + x2) / 2
        box_center_y = (y1 + y2) / 2

        # Pose quality and visibility from landmarks if available
        pose_quality = 0.5  # default moderate
        pose_visibility = 0.5  # default moderate
        if landmarks is not None:
            # Try to get pose quality from the estimate
            if hasattr(landmarks, 'pose_quality') and landmarks.pose_quality is not None:
                pose_quality = landmarks.pose_quality
            # Average keypoint visibility
            if hasattr(landmarks, 'nose') and landmarks.nose is not None:
                pose_visibility = landmarks.nose.confidence
            if hasattr(landmarks, 'left_shoulder') and landmarks.left_shoulder is not None:
                pose_visibility = max(
                    pose_visibility,
                    landmarks.left_shoulder.confidence,
                )
            if hasattr(landmarks, 'right_shoulder') and landmarks.right_shoulder is not None:
                pose_visibility = max(
                    pose_visibility,
                    landmarks.right_shoulder.confidence,
                )
            # Compute shoulder-hip distance (relative to box height)
            if (hasattr(landmarks, 'left_shoulder') and landmarks.left_shoulder is not None and
                hasattr(landmarks, 'right_hip') and landmarks.right_hip is not None and
                hasattr(landmarks, 'left_hip') and landmarks.left_hip is not None):
                sh_dist = abs(landmarks.left_shoulder.y - landmarks.right_hip.y)
                shoulder_hip_distance = sh_dist / max(box_height, 1)
            else:
                shoulder_hip_distance = 0.5  # default moderate
        else:
            shoulder_hip_distance = 0.5  # default moderate (person standing)

        # Compute centroid velocity if we have history
        # (will be updated by the caller using history)
        centroid_velocity = 0.0

        # Determine if torso is approximately vertical
        # A vertical torso has similar x for shoulders and similar x for hips
        is_vertical_torso = True  # default assumption
        if (hasattr(landmarks, 'left_shoulder') and landmarks.left_shoulder is not None and
            hasattr(landmarks, 'right_shoulder') and landmarks.right_shoulder is not None and
            hasattr(landmarks, 'left_hip') and landmarks.left_hip is not None and
            hasattr(landmarks, 'right_hip') and landmarks.right_hip is not None):
            shoulder_width = abs(landmarks.left_shoulder.x - landmarks.right_shoulder.x)
            hip_width = abs(landmarks.left_hip.x - landmarks.right_hip.x)
            # If shoulders and hips have similar width, torso is likely vertical
            is_vertical_torso = (shoulder_width < box_width * 0.8 and
                                hip_width < box_width * 0.8)

        # Compute center displacement from window start
        # (will be updated by add_observation using history)

        return ActivityObservation(
            timestamp=timestamp,
            center_displacement=0.0,  # will be set by add_observation
            bbox_height=box_height,
            bbox_aspect_ratio=box_width / max(box_height, 1),
            centroid_velocity=centroid_velocity,
            pose_quality=pose_quality,
            pose_visibility=pose_visibility,
            shoulder_hip_distance=shoulder_hip_distance,
            is_vertical_torso=is_vertical_torso,
            frame_index=frame_index,
        )

    def add_observation(
        self,
        track_id: int,
        bbox: Tuple[float, float, float, float],
        landmarks: Optional[object],
        timestamp: float,
        frame_index: int,
        frame_width: int,
        frame_height: int,
    ) -> Optional[ActivityResult]:
        """Add a new observation and potentially return an activity result.

        Args:
            track_id: Persistent track identifier
            bbox: (x1, y1, x2, y2) in pixel coordinates
            landmarks: Pose landmarks (optional)
            timestamp: Current frame timestamp in seconds
            frame_index: Index of this frame in the video
            frame_width: Width of the full frame
            frame_height: Height of the full frame

        Returns:
            ActivityResult if enough evidence has been accumulated, or None
            if more observations are needed. Returns None when status is
            "uncertain" or "preliminary" if more data is still needed.
        """
        state = self._get_state(track_id)

        # Extract observation features
        obs = self._extract_observation(
            bbox, landmarks, timestamp, frame_index,
            frame_width, frame_height,
        )

        # Calculate center displacement from window start
        # Use the oldest observation in the window as reference
        if len(state.window) > 0:
            start_obs = state.window[0]
            obs.center_displacement = obs.timestamp - start_obs.timestamp
        else:
            obs.center_displacement = 0.0

        # Add observation to window
        state.window.append(obs)

        # Ensure window stays within observation window duration
        # Remove old observations at the front that are too old
        while (state.window and
               timestamp - state.window[0].timestamp > self.observation_window_sec):
            state.window.popleft()

        # If window is too small, return preliminary
        if len(state.window) < 3:
            return None  # Need more data

        # Compute derived features from the window
        result = self._analyze_window(state, timestamp, frame_width, frame_height)
        return result

    def _analyze_window(
        self,
        state: ActivityTrackerState,
        current_timestamp: float,
        frame_width: int,
        frame_height: int,
    ) -> Optional[ActivityResult]:
        """Analyze the observation window and return an activity result.

        Args:
            state: The tracker state for this track
            current_timestamp: Current time
            frame_width: Frame width (for velocity calculations)
            frame_height: Frame height (for velocity calculations)

        Returns:
            ActivityResult if enough evidence, None if more data needed.
        """
        window = state.window
        if len(window) < 3:
            return None

        # Get first and last observations in window
        start_obs = window[0]
        end_obs = window[-1]

        evidence_duration = end_obs.timestamp - start_obs.timestamp

        # --- Compute features from the window ---

        # 1. Center displacement (total movement)
        center_displacement = end_obs.center_displacement

        # 2. Average bbox height and aspect ratio
        avg_bbox_height = np.mean([o.bbox_height for o in window])
        avg_bbox_aspect_ratio = np.mean([o.bbox_aspect_ratio for o in window])

        # 3. Average centroid velocity (approximate from displacements)
        # Use simple displacement / time
        if len(window) > 1:
            total_time = window[-1].timestamp - window[0].timestamp
            if total_time > 0:
                # Approximate velocity from center displacement
                centroid_velocity = abs(end_obs.center_displacement) / total_time
            else:
                centroid_velocity = 0.0
        else:
            centroid_velocity = 0.0

        # 4. Average pose quality
        avg_pose_quality = np.mean([o.pose_quality for o in window])

        # 5. Average pose visibility
        avg_pose_visibility = np.mean([o.pose_visibility for o in window])

        # 6. Posture analysis
        # Determine dominant posture from last several observations
        recent_observations = list(window)[-5:]  # last 5

        # Count observations suggesting each posture type
        sitting_count = 0
        standing_count = 0
        lying_count = 0
        bending_count = 0

        for obs in recent_observations:
            # Sitting: higher aspect ratio (wider than tall), shoulder-hip distance small
            if obs.bbox_aspect_ratio > 1.2 and obs.shoulder_hip_distance < 0.6:
                sitting_count += 1
            # Standing: aspect ratio ~0.5-0.7 (taller than wide), vertical torso
            elif obs.bbox_aspect_ratio < 0.9 and obs.is_vertical_torso:
                standing_count += 1
            # Lying down: very high aspect ratio (very wide), or very low pose quality
            elif obs.bbox_aspect_ratio > 2.0:
                lying_count += 1
            # Bending: moderate aspect ratio, torso not vertical
            elif not obs.is_vertical_torso and 0.8 < obs.bbox_aspect_ratio < 1.5:
                bending_count += 1

        # --- Activity Classification ---

        # Determine the most likely activity
        counts = {
            ActivityCategory.SITTING: sitting_count,
            ActivityCategory.STANDING: standing_count,
            ActivityCategory.LYING_DOWN: lying_count,
            ActivityCategory.BENDING: bending_count,
        }
        # Add walking/running based on velocity and displacement
        walking_score = 0
        running_score = 0

        if centroid_velocity > 50:  # moving significantly
            walking_score = min(5, int(centroid_velocity / 30))
        if centroid_velocity > 100:  # moving fast
            running_score = min(5, int(centroid_velocity / 50))

        counts[ActivityCategory.WALKING] = counts.get(ActivityCategory.WALKING, 0) + walking_score
        counts[ActivityCategory.RUNNING] = counts.get(ActivityCategory.RUNNING, 0) + running_score

        # Find the activity with highest count
        if counts:
            best_activity = max(counts, key=counts.get)
            best_count = counts[best_activity]
        else:
            best_activity = ActivityCategory.UNKNOWN
            best_count = 0

        # --- Fall Detection ---
        # Fall detection: rapid downward transition + low posture afterwards
        fall_detected = False
        fall_explanation = ""

        # Check for rapid downward movement
        if (state.fall_rapid_downward and
            evidence_duration > 0.5):  # must persist briefly
            # Check if we went from standing-ish to very low
            if (avg_bbox_height > 100 and  # was relatively tall before
                end_obs.bbox_height < 50):  # now very short (lying down)
                # Check downward velocity
                height_change = end_obs.bbox_height - start_obs.bbox_height
                time_elapsed = end_obs.timestamp - start_obs.timestamp
                if time_elapsed > 0:
                    downward_velocity = height_change / time_elapsed
                    if downward_velocity < -self.fall_rapid_velocity:
                        fall_detected = True
                        fall_explanation = (
                            "Rapid downward body transition detected; "
                            "possible fall - requires human review."
                        )

        # --- Hysteresis / Stability Check ---

        # Has the activity changed recently?
        activity_changed = (
            best_activity != state.current_activity
            or evidence_duration < self.min_evidence_duration_sec
        )

        if activity_changed and best_count >= self.hysteresis_threshold:
            # New activity detected - check hysteresis
            # Require hysteresis_threshold consecutive frames at this activity
            # (already handled by window size, but check stability counter)
            if state.activity_stability_counter >= self.min_stable_frames:
                # Switch to new activity
                state.previous_activity = state.current_activity
                state.current_activity = best_activity
                state.current_confidence = min(best_count / len(window), 1.0)
                state.activity_start_time = start_obs.timestamp
                state.activity_stability_counter = 0
                state.last_activity_change = current_timestamp
            # else: keep current activity, increment counter
            state.activity_stability_counter += 1
        elif not activity_changed:
            # Same activity - increment stability counter
            state.activity_stability_counter += 1

        # --- Status Determination ---

        # If we have sufficient evidence duration and stability, mark as stable
        if (evidence_duration >= self.min_evidence_duration_sec and
            state.activity_stability_counter >= self.min_stable_frames and
            best_count >= self.hysteresis_threshold):
            status = "stable"
        elif evidence_duration > 0 and state.activity_stability_counter > 0:
            status = "preliminary"
        else:
            status = "uncertain"

        # --- Fall Override ---
        if fall_detected:
            # If fall detected, override to POSSIBLE_FALL with low confidence
            # but keep the original activity info in explanation
            actual_activity = ActivityCategory.POSSIBLE_FALL
            confidence = 0.4  # moderate confidence it's a fall
            explanation = fall_explanation
            status = "uncertain"  # falls always need review
        else:
            actual_activity = best_activity
            confidence = min(best_count / max(len(window), 1), 1.0)
            explanation = self._generate_explanation(
                actual_activity, recent_observations, avg_bbox_height,
                avg_pose_quality, centroid_velocity,
            )

        # Update activity history
        state.activity_history.append((
            actual_activity,
            current_timestamp,
        ))
        # Keep history manageable
        if len(state.activity_history) > 50:
            state.activity_history = state.activity_history[-50:]

        # Update fall detection state
        # If we just detected a fall, mark it; if person was standing before, reset
        if fall_detected:
            state.fall_triggered = True
        else:
            # Reset fall state if person is clearly standing/walking now
            if actual_activity in (ActivityCategory.STANDING, ActivityCategory.WALKING,
                                  ActivityCategory.RUNNING):
                state.fall_triggered = False

        # Clip confidence to 0-1 range
        confidence = max(0.0, min(1.0, confidence))

        # Generate start/last update timestamps
        start_ts = state.activity_start_time if state.activity_start_time > 0 else start_obs.timestamp
        last_update = window[-1].timestamp

        # Build explanation
        explanation_full = self._generate_explanation(
            actual_activity, recent_observations, avg_bbox_height,
            avg_pose_quality, centroid_velocity,
        )

        return ActivityResult(
            activity=actual_activity,
            confidence=confidence,
            start_timestamp=start_ts,
            last_update=last_update,
            evidence_duration=evidence_duration,
            pose_quality=avg_pose_quality,
            explanation=explanation_full,
            status=status,
        )

    def _generate_explanation(
        self,
        activity: ActivityCategory,
        recent_observations: List[ActivityObservation],
        avg_bbox_height: float,
        avg_pose_quality: float,
        centroid_velocity: float,
    ) -> str:
        """Generate a human-readable explanation of the activity inference.

        Args:
            activity: The inferred activity category
            recent_observations: Recent observations from the window
            avg_bbox_height: Average bounding box height
            avg_pose_quality: Average pose quality
            centroid_velocity: Estimated centroid velocity

        Returns:
            Brief text explaining supporting observations
        """
        parts = []

        # Activity-specific observations
        if activity == ActivityCategory.STANDING:
            if avg_bbox_height > 100:
                parts.append(f"person is upright ({avg_bbox_height:.0f}px tall)")
            if avg_pose_quality > 0.5:
                parts.append("pose quality good")
            if centroid_velocity < 20:
                parts.append("minimal movement")
            else:
                parts.append("some movement detected")

        elif activity == ActivityCategory.SITTING:
            if avg_bbox_height > 0 and avg_bbox_height < 80:
                parts.append(f"person is seated ({avg_bbox_height:.0f}px height)")
            if avg_pose_quality > 0.5:
                parts.append("pose quality moderate")
            parts.append("aspect ratio suggests sitting posture")

        elif activity == ActivityCategory.WALKING:
            if centroid_velocity > 50:
                parts.append(f"moving at {centroid_velocity:.0f}px/s")
            parts.append("center displacement indicates forward movement")
            if avg_pose_quality > 0.5:
                parts.append("pose quality moderate for walking analysis")

        elif activity == ActivityCategory.RUNNING:
            if centroid_velocity > 100:
                parts.append(f"fast movement at {centroid_velocity:.0f}px/s")
            parts.append("high velocity suggests running")

        elif activity == ActivityCategory.LYING_DOWN:
            if avg_bbox_height < 60:
                parts.append(f"person recumbent ({avg_bbox_height:.0f}px height)")
            parts.append("very wide aspect ratio suggests lying down")
            if avg_pose_quality < 0.5:
                parts.append("low pose quality - may be partially obscured")

        elif activity == ActivityCategory.BENDING:
            if recent_observations and not recent_observations[0].is_vertical_torso:
                parts.append("torso not vertical - bending posture")
            parts.append("moderate aspect ratio supports bending inference")

        elif activity == ActivityCategory.POSSIBLE_FALL:
            parts.append("rapid downward transition detected")

        elif activity == ActivityCategory.UNCERTAIN:
            parts.append("insufficient evidence for conclusive classification")
        else:
            parts.append(f"activity classified as {activity_to_string(activity)}")

        # Add pose quality note
        if avg_pose_quality > 0.7:
            parts.append("pose quality high - confidence increased")
        elif avg_pose_quality < 0.3:
            parts.append("pose quality low - confidence decreased")

        # Add evidence duration
        if recent_observations:
            dur = recent_observations[-1].timestamp - recent_observations[0].timestamp
            if dur > 0:
                parts.append(f"observed for {dur:.1f}s")

        return "; ".join(parts) if parts else "activity assessment in progress"

    def reset_track(self, track_id: int) -> None:
        """Reset the activity tracker state for a track.

        Call when a track is created or when the stream restarts.
        """
        if track_id in self.states:
            self.states[track_id] = ActivityTrackerState()

    def reset_all(self) -> None:
        """Reset all tracker states (e.g., on stream restart)."""
        self.states.clear()