"""STAI activity validation, camera health, and incident policy helpers.

This module deliberately has no model or network dependencies so policy and
validation remain usable when hosted inference is not configured.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
import sqlite3
import uuid


class ActivityCategory(str, Enum):
    NORMAL = "NORMAL"
    SOCIAL_INTERACTION = "SOCIAL_INTERACTION"
    POSSIBLE_FALL = "POSSIBLE_FALL"
    POSSIBLE_VIOLENCE = "POSSIBLE_VIOLENCE"
    CROWD_SURGE = "CROWD_SURGE"
    FIRE_OR_SMOKE = "FIRE_OR_SMOKE"
    OTHER_HAZARD = "OTHER_HAZARD"
    UNCERTAIN = "UNCERTAIN"


class VideoQuality(str, Enum):
    GOOD = "GOOD"
    BLURRY = "BLURRY"
    TOO_DARK = "TOO_DARK"
    OBSTRUCTED = "OBSTRUCTED"
    FROZEN = "FROZEN"
    UNKNOWN = "UNKNOWN"


class IncidentStatus(str, Enum):
    NEW = "NEW"
    UNDER_REVIEW = "UNDER_REVIEW"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"
    FALSE_ALARM = "FALSE_ALARM"


@dataclass(frozen=True)
class ActivityAssessment:
    activity_category: ActivityCategory
    description: str
    risk_level: str
    confidence: float
    evidence: list[str]
    requires_human_review: bool
    video_quality: VideoQuality

    @classmethod
    def validate(cls, value) -> "ActivityAssessment":
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError("assessment must be valid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("assessment must be an object")
        required = {"activity_category", "description", "risk_level", "confidence",
                    "evidence", "requires_human_review", "video_quality"}
        if not required.issubset(value):
            raise ValueError("assessment is missing required fields")
        try:
            category = ActivityCategory(value["activity_category"])
            quality = VideoQuality(value["video_quality"])
        except (TypeError, ValueError) as exc:
            raise ValueError("assessment contains an unsupported category or quality") from exc
        risk = value["risk_level"]
        confidence = value["confidence"]
        evidence = value["evidence"]
        if risk not in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}:
            raise ValueError("risk_level is invalid")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("confidence must be a finite number from 0 to 1")
        if not isinstance(value["description"], str) or not value["description"].strip():
            raise ValueError("description must be non-empty text")
        if not isinstance(value["requires_human_review"], bool):
            raise ValueError("requires_human_review must be boolean")
        if not isinstance(evidence, list) or len(evidence) > 12 or any(not isinstance(item, str) or not item.strip() for item in evidence):
            raise ValueError("evidence must be a list of short text cues")
        return cls(category, value["description"].strip()[:500], risk, float(confidence),
                   [item.strip()[:200] for item in evidence], value["requires_human_review"], quality)


@dataclass
class QualityThresholds:
    blur_variance_min: float = 35.0
    brightness_min: float = 28.0
    obstruction_std_min: float = 4.0
    frozen_similarity_min: float = 0.999
    frozen_repeats: int = 8


class CameraHealth:
    """Frame-quality checks with repetition history; static scenes alone do not freeze a feed."""
    def __init__(self, thresholds: QualityThresholds | None = None):
        self.thresholds = thresholds or QualityThresholds()
        self._fingerprints: deque[str] = deque(maxlen=self.thresholds.frozen_repeats)

    def inspect(self, frame, cv2_module) -> tuple[VideoQuality, str]:
        if frame is None or not getattr(frame, "size", 0):
            return VideoQuality.UNKNOWN, "Camera returned an invalid or missing frame."
        gray = cv2_module.cvtColor(frame, cv2_module.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        blur = float(cv2_module.Laplacian(gray, cv2_module.CV_64F).var())
        small = cv2_module.resize(gray, (32, 18), interpolation=cv2_module.INTER_AREA)
        fingerprint = hashlib.sha256(small.tobytes()).hexdigest()
        self._fingerprints.append(fingerprint)
        if len(self._fingerprints) == self._fingerprints.maxlen and len(set(self._fingerprints)) == 1:
            return VideoQuality.FROZEN, "The same sampled frame repeated across the camera health window."
        # A perfectly uniform dark or bright image often indicates a covered lens.
        if float(gray.std()) < self.thresholds.obstruction_std_min and (brightness < 12 or brightness > 243):
            return VideoQuality.OBSTRUCTED, "Frame is nearly uniform; the lens may be covered."
        if brightness < self.thresholds.brightness_min:
            return VideoQuality.TOO_DARK, f"Frame brightness is low ({brightness:.1f}/255)."
        if blur < self.thresholds.blur_variance_min:
            return VideoQuality.BLURRY, f"Frame sharpness is low (variance {blur:.1f})."
        return VideoQuality.GOOD, "Frame quality checks passed."


class IncidentStore:
    """Small local SQLite incident log with duplicate suppression and guarded status changes."""
    transitions = {
        IncidentStatus.NEW: {IncidentStatus.UNDER_REVIEW, IncidentStatus.ACKNOWLEDGED, IncidentStatus.RESOLVED, IncidentStatus.FALSE_ALARM},
        IncidentStatus.UNDER_REVIEW: {IncidentStatus.ACKNOWLEDGED, IncidentStatus.RESOLVED, IncidentStatus.FALSE_ALARM},
        IncidentStatus.ACKNOWLEDGED: {IncidentStatus.UNDER_REVIEW, IncidentStatus.RESOLVED, IncidentStatus.FALSE_ALARM},
        IncidentStatus.RESOLVED: set(), IncidentStatus.FALSE_ALARM: set(),
    }

    def __init__(self, path="stai_incidents.sqlite3", dedupe_seconds=120):
        self.path = path
        self.dedupe_seconds = dedupe_seconds
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS incidents (
                id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, camera_id TEXT NOT NULL,
                created_at TEXT NOT NULL, category TEXT NOT NULL, risk TEXT NOT NULL,
                description TEXT NOT NULL, evidence TEXT NOT NULL, quality TEXT NOT NULL,
                confidence REAL NOT NULL, status TEXT NOT NULL, telegram_status TEXT NOT NULL)""")

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        return db

    def record(self, assessment: ActivityAssessment, camera_id="camera-0", now=None):
        now = now or datetime.now(timezone.utc)
        stamp = now.isoformat()
        fingerprint = hashlib.sha256(f"{camera_id}|{assessment.activity_category.value}|{assessment.description[:120]}".encode()).hexdigest()
        with self._connect() as db:
            previous = db.execute("SELECT * FROM incidents WHERE fingerprint=? ORDER BY created_at DESC LIMIT 1", (fingerprint,)).fetchone()
            if previous:
                try:
                    age = (now - datetime.fromisoformat(previous["created_at"])).total_seconds()
                except ValueError:
                    age = self.dedupe_seconds + 1
                if 0 <= age < self.dedupe_seconds:
                    return dict(previous), False
            incident = {
                "id": str(uuid.uuid4()), "fingerprint": fingerprint, "camera_id": camera_id,
                "created_at": stamp, "category": assessment.activity_category.value,
                "risk": assessment.risk_level, "description": assessment.description,
                "evidence": json.dumps(assessment.evidence), "quality": assessment.video_quality.value,
                "confidence": assessment.confidence, "status": IncidentStatus.NEW.value,
                "telegram_status": "DISABLED",
            }
            db.execute("INSERT INTO incidents VALUES (:id,:fingerprint,:camera_id,:created_at,:category,:risk,:description,:evidence,:quality,:confidence,:status,:telegram_status)", incident)
            return incident, True

    def update_status(self, incident_id, status):
        try:
            target = IncidentStatus(status)
        except ValueError as exc:
            raise ValueError("unsupported incident status") from exc
        with self._connect() as db:
            row = db.execute("SELECT status FROM incidents WHERE id=?", (incident_id,)).fetchone()
            if not row:
                raise KeyError("incident not found")
            current = IncidentStatus(row["status"])
            if target not in self.transitions[current]:
                raise ValueError(f"cannot move incident from {current.value} to {target.value}")
            db.execute("UPDATE incidents SET status=? WHERE id=?", (target.value, incident_id))

    def set_telegram_status(self, incident_id, status):
        if status not in {"DISABLED", "SENT", "FAILED", "SUPPRESSED"}:
            raise ValueError("invalid notification status")
        with self._connect() as db:
            db.execute("UPDATE incidents SET telegram_status=? WHERE id=?", (status, incident_id))

    def list_incidents(self, limit=100):
        with self._connect() as db:
            return [dict(row) for row in db.execute("SELECT * FROM incidents ORDER BY created_at DESC LIMIT ?", (max(1, min(int(limit), 500)),))]


def should_notify(assessment: ActivityAssessment, quality_warning_only=False) -> bool:
    """Notifications require a reviewable possible hazard; ordinary/social activity is quiet."""
    if quality_warning_only or assessment.activity_category in {ActivityCategory.NORMAL, ActivityCategory.SOCIAL_INTERACTION, ActivityCategory.UNCERTAIN}:
        return False
    return assessment.requires_human_review and assessment.risk_level in {"HIGH", "CRITICAL"}


def build_telegram_message(incident: dict, dashboard_url="") -> str:
    """Construct plain-text safe Telegram content; no Markdown parsing is enabled."""
    timestamp = incident["created_at"].replace("T", " ")[:19] + " UTC"
    lines = [
        "STAI — HUMAN REVIEW REQUESTED", f"Incident: {incident['id']}",
        f"Camera: {incident['camera_id']}", f"Time: {timestamp}",
        f"Activity: {incident['category']}", f"Severity: {incident['risk']}",
        f"Video quality: {incident['quality']}",
        f"Model confidence: {incident['confidence']:.0%}",
        f"Assessment: {incident['description'][:400]}",
        "Evidence: " + "; ".join(json.loads(incident["evidence"])[:5]),
        "Please verify the camera feed. This is an AI assessment, not a confirmed emergency.",
    ]
    if dashboard_url.startswith(("https://", "http://")):
        lines.insert(2, f"Incident link: {dashboard_url.rstrip('/')}/incidents/{incident['id']}")
    return "\n".join(lines)


def deliver_telegram(message, bot_token, chat_id, enabled, http_post, timeout=5):
    """Send plain text when explicitly enabled; return a non-sensitive status."""
    if not enabled:
        return "DISABLED"
    if not bot_token or not chat_id:
        return "FAILED"
    try:
        response = http_post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            data={"chat_id": chat_id, "text": str(message)},
            timeout=timeout,
        )
        return "SENT" if response.status_code == 200 else "FAILED"
    except Exception:
        # Exception messages can contain the request URL, which contains the token.
        return "FAILED"


def build_activity_report(
    person_count: int,
    grid_counts: list | None,
    grid_n: int,
    camera_id: str = "camera-0",
    dashboard_url: str = "",
    anomaly_result=None,
    incident_records: list | None = None,
) -> str:
    """Build a readable Telegram activity report.

    Args:
        person_count: Total people currently detected.
        grid_counts: NxN list of lists of person counts per grid cell, or None.
        grid_n: Grid dimension (N x N).
        camera_id: Camera identifier string.
        dashboard_url: Optional base URL for incident links.
        anomaly_result: Latest AnomalyResult if available.
        incident_records: List of incident dicts requiring review.

    Returns:
        Plain-text message safe for Telegram (no Markdown parsing).
    """
    from stai_core import VideoQuality, IncidentStatus

    # Timestamp
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Crowd density status
    if grid_counts is not None and len(grid_counts) > 0:
        # Compute max density without numpy
        max_density = 0
        hot_cells = []
        for r in range(len(grid_counts)):
            for c in range(len(grid_counts[r])):
                count = grid_counts[r][c]
                if count > max_density:
                    max_density = count
                if count >= 2:
                    hot_cells.append((r, c))
        hot_cell_str = (
            ", ".join(f"row{r+1}/col{c+1}" for r, c in hot_cells) if hot_cells else "none"
        )
    else:
        max_density = 0
        hot_cell_str = "none"
    overcrowd = person_count >= 50  # threshold can be adjusted by caller

    # Activity summary from anomaly result if available
    activity_summary_parts = []
    if anomaly_result is not None:
        label = anomaly_result.activity_label
        activity_summary_parts.append(f"- {label.replace('_', ' ')}: 1")

    # Incidents requiring review
    pending_incidents = []
    if incident_records:
        for rec in incident_records:
            if rec.get("status") in {IncidentStatus.NEW.value, IncidentStatus.UNDER_REVIEW.value}:
                pending_incidents.append(
                    {
                        "label": rec.get("activity_label", "UNCLEAR"),
                        "risk": rec.get("risk", "LOW"),
                        "id": rec["id"],
                    }
                )

    # Build message lines
    lines = [
        "STAI — Activity Review",
        f"Camera: {camera_id}",
        f"Time: {timestamp}",
        f"People detected: {person_count}",
        f"Max density (grid): {max_density}",
        f"Hot cells: {hot_cell_str}",
    ]

    if overcrowd:
        lines.append("Crowd density: HIGH - overcrowd threshold exceeded")

    lines.append("Activity summary:")
    if activity_summary_parts:
        lines.extend(activity_summary_parts)
    else:
        lines.append("- No recent VLM activity classification")

    if pending_incidents:
        lines.append("Review required:")
        for inc in pending_incidents[:3]:
            risk_text = inc["risk"].upper()
            lines.append(f"- {inc['label']} (risk: {risk_text}) — ID: {inc['id'][:8]}...")
    else:
        lines.append("No incidents requiring review")

    lines.append(
        "Status: Automated analysis; not a confirmed emergency."
    )

    if dashboard_url.startswith(("https://", "http://")):
        lines.insert(2, f"Incident link: {dashboard_url.rstrip('/')}/incidents/")

    return "\n".join(lines)
