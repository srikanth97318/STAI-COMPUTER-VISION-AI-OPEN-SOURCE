"""Hugging Face vision inference and incident grouping for STAI."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
import os
import sqlite3
import uuid

ACTIVITY_LABELS = {
    "CALLING", "CLAPPING", "CYCLING", "DANCING", "DRINKING", "EATING",
    "FIGHTING", "HUGGING", "LAUGHING", "LISTENING_TO_MUSIC", "RUNNING",
    "SITTING", "SLEEPING", "TEXTING", "WORKING", "OTHER", "UNCLEAR",
}


@dataclass(frozen=True)
class AnomalyResult:
    activity_label: str
    anomaly_score: float | None
    score_type: str
    anomaly_detected: bool
    decision: str
    start_time_seconds: float | None
    end_time_seconds: float | None
    model_name: str
    requires_review: bool
    status: str
    video_quality: str
    explanation: str

    @classmethod
    def validate(cls, payload):
        if not isinstance(payload, dict):
            raise ValueError("anomaly result must be an object")
        required = {
            "activity_label", "anomaly_score", "score_type", "anomaly_detected", "decision",
            "start_time_seconds", "end_time_seconds", "model_name",
            "requires_review", "status", "video_quality", "explanation",
        }
        if required - payload.keys():
            raise ValueError("anomaly result is missing required fields")
        if payload["activity_label"] not in ACTIVITY_LABELS:
            raise ValueError("unsupported activity label")
        score = payload["anomaly_score"]
        if score is not None and (
            isinstance(score, bool)
            or not isinstance(score, (float, int))
            or not math.isfinite(score)
            or not 0 <= score <= 1
        ):
            raise ValueError("anomaly_score must be null or a finite score from 0 to 1")
        if payload["score_type"] not in {"ANOMALY_SCORE", "SIMILARITY_SCORE", "DISTANCE_SCORE"}:
            raise ValueError("unsupported score_type")
        if payload["decision"] not in {"NORMAL", "ANOMALOUS", "UNCERTAIN"}:
            raise ValueError("unsupported anomaly decision")
        if payload["status"] not in {"SUCCESS", "UNCERTAIN", "ERROR"}:
            raise ValueError("unsupported anomaly status")
        if not isinstance(payload["anomaly_detected"], bool) or not isinstance(payload["requires_review"], bool):
            raise ValueError("anomaly flags must be boolean")
        if payload["anomaly_detected"] != (payload["decision"] == "ANOMALOUS"):
            raise ValueError("anomaly_detected must agree with decision")

        expected_decision = (
            "ANOMALOUS" if payload["activity_label"] == "FIGHTING"
            else "UNCERTAIN" if payload["activity_label"] == "UNCLEAR"
            else "NORMAL"
        )
        if payload["decision"] != expected_decision:
            raise ValueError(
                f"activity label '{payload['activity_label']}' and decision '{payload['decision']}' disagree"
            )

        for field in ("start_time_seconds", "end_time_seconds"):
            value = payload[field]
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{field} must be null or a non-negative finite number")
        if (
            payload["start_time_seconds"] is not None
            and payload["end_time_seconds"] is not None
            and payload["start_time_seconds"] > payload["end_time_seconds"]
        ):
            raise ValueError("anomaly interval is reversed")
        for field in ("model_name", "video_quality", "explanation"):
            if not isinstance(payload[field], str):
                raise ValueError(f"{field} must be text")
        return cls(**{field: payload[field] for field in required})

    def as_dict(self):
        return {
            "activity_label": self.activity_label, "anomaly_score": self.anomaly_score,
            "score_type": self.score_type,
            "anomaly_detected": self.anomaly_detected, "decision": self.decision,
            "start_time_seconds": self.start_time_seconds, "end_time_seconds": self.end_time_seconds,
            "model_name": self.model_name, "requires_review": self.requires_review,
            "status": self.status, "video_quality": self.video_quality,
            "explanation": self.explanation,
        }


class HuggingFaceInferenceActionScorer:
    """Classify one representative frame with Hugging Face's image-classification API."""

    endpoint = "https://router.huggingface.co/hf-inference/models/{}"

    ACTIONS = {
        "calling": "CALLING", "clapping": "CLAPPING", "cycling": "CYCLING",
        "dancing": "DANCING", "drinking": "DRINKING", "eating": "EATING",
        "fighting": "FIGHTING", "hugging": "HUGGING", "laughing": "LAUGHING",
        "listening_to_music": "LISTENING_TO_MUSIC", "running": "RUNNING",
        "sitting": "SITTING", "sleeping": "SLEEPING", "texting": "TEXTING",
        "using_laptop": "WORKING",
    }

    def __init__(self, token=None, model="prithivMLmods/Human-Action-Recognition", timeout=45):
        if not model:
            raise ValueError("ANOMALY_MODEL_NAME must be a Hugging Face model ID")
        if float(timeout) <= 0:
            raise ValueError("Hugging Face request timeout must be positive")
        self.token = token or ""
        self.model_name = model
        self.timeout = float(timeout)

    @staticmethod
    def _normalize(name):
        return str(name).strip().lower().replace(" ", "_").replace("-", "_")

    @staticmethod
    def _prediction(result):
        if isinstance(result, list):
            if result and all(isinstance(item, dict) for item in result):
                candidates = {
                    item.get("label"): item.get("score", item.get("confidence"))
                    for item in result
                }
                return HuggingFaceInferenceActionScorer._prediction(candidates)
            if len(result) == 1:
                result = result[0]
        if isinstance(result, dict):
            candidates = result.get("confidences", result)
            if isinstance(candidates, list):
                candidates = {
                    item.get("label"): item.get("confidence")
                    for item in candidates if isinstance(item, dict)
                }
            if isinstance(candidates, dict):
                normalized = {}
                for name, confidence in candidates.items():
                    label = HuggingFaceInferenceActionScorer.ACTIONS.get(
                        HuggingFaceInferenceActionScorer._normalize(name)
                    )
                    if (
                        label
                        and isinstance(confidence, (int, float))
                        and not isinstance(confidence, bool)
                        and math.isfinite(confidence)
                    ):
                        normalized[label] = float(confidence)
                if normalized:
                    return max(normalized.items(), key=lambda item: item[1])
            label = result.get("label")
            if isinstance(label, str):
                mapped = HuggingFaceInferenceActionScorer.ACTIONS.get(
                    HuggingFaceInferenceActionScorer._normalize(label)
                )
                if mapped:
                    return mapped, None
        if isinstance(result, str):
            mapped = HuggingFaceInferenceActionScorer.ACTIONS.get(
                HuggingFaceInferenceActionScorer._normalize(result)
            )
            if mapped:
                return mapped, None
        raise ValueError("The Inference API returned an unrecognized classification result")

    def analyze_video(self, clip_or_frames) -> AnomalyResult:
        entries = list(clip_or_frames)
        valid_times = [
            float(row["timestamp_seconds"]) for row in entries
            if isinstance(row, dict) and isinstance(row.get("timestamp_seconds"), (int, float))
        ]
        start, end = (min(valid_times), max(valid_times)) if valid_times else (None, None)
        quality_values = [str(row.get("video_quality", "UNKNOWN")) for row in entries if isinstance(row, dict)]
        quality = next((value for value in quality_values if value != "GOOD"), "GOOD")
        try:
            if not entries:
                raise ValueError("no sampled frames were provided")
            if quality != "GOOD":
                return self._result(
                    "UNCLEAR", None, "UNCERTAIN", "UNCERTAIN", start, end, quality,
                    f"Frame quality is {quality}; action classification was skipped.",
                )
            if not self.token:
                return self._result(
                    "UNCLEAR", None, "UNCERTAIN", "ERROR", start, end, quality,
                    "HF_TOKEN is missing. Add a Hugging Face token with Inference Providers permission to .env.",
                )
            entry = entries[len(entries) // 2]
            frame = entry.get("frame")
            if frame is None or not getattr(frame, "size", 0):
                raise ValueError("selected frame is empty")

            import cv2
            height, width = frame.shape[:2]
            scale = min(640 / width, 640 / height, 1.0)
            if scale < 1:
                image = cv2.resize(
                    frame,
                    (max(1, int(width * scale)), max(1, int(height * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                image = frame
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                raise RuntimeError("Could not encode sampled frame")

            import requests
            response = requests.post(
                self.endpoint.format(self.model_name),
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": "image/jpeg",
                    "Accept": "application/json",
                },
                data=encoded.tobytes(),
                timeout=self.timeout,
            )
            if response.status_code == 401:
                raise RuntimeError("Hugging Face rejected HF_TOKEN; create a token with Inference Providers permission")
            if response.status_code == 403:
                raise RuntimeError("HF_TOKEN does not have permission to call Hugging Face Inference Providers")
            if response.status_code == 402:
                raise RuntimeError("Hugging Face Inference Provider credits are exhausted; wait for the monthly free credit reset")
            if response.status_code == 404:
                raise RuntimeError("This model has no active hf-inference image-classification deployment")
            if response.status_code == 410:
                raise RuntimeError("This model is deprecated and no longer supported by the Hugging Face Inference provider")
            if response.status_code == 429:
                raise RuntimeError("Hugging Face rate limit reached; retrying in 60 seconds")
            if response.status_code == 503:
                raise RuntimeError("Hugging Face model is loading or temporarily unavailable; retrying in 60 seconds")
            response.raise_for_status()
            result = response.json()
            label, score = self._prediction(result)
            decision = "ANOMALOUS" if label == "FIGHTING" else "NORMAL"
            confidence_text = f"; classifier confidence {score:.0%}" if score is not None else ""
            return AnomalyResult.validate({
                "activity_label": label,
                "anomaly_score": score,
                "score_type": "SIMILARITY_SCORE",
                "anomaly_detected": decision == "ANOMALOUS",
                "decision": decision,
                "start_time_seconds": start,
                "end_time_seconds": end,
                "model_name": self.model_name,
                "requires_review": label == "FIGHTING",
                "status": "SUCCESS",
                "video_quality": quality,
                "explanation": (
                    f"The image classifier identified {label.lower().replace('_', ' ')}"
                    f"{confidence_text}."
                ),
            })
        except Exception as exc:
            if isinstance(exc, RuntimeError):
                detail = str(exc)
            else:
                detail = f"Hugging Face image classification failed ({type(exc).__name__}: {str(exc)[:180]})."
            return self._result(
                "UNCLEAR", None, "UNCERTAIN", "ERROR", start, end, quality,
                detail,
            )

    def _result(self, label, score, decision, status, start, end, quality, explanation):
        return AnomalyResult.validate({
            "activity_label": label,
            "anomaly_score": score,
            "score_type": "ANOMALY_SCORE",
            "anomaly_detected": decision == "ANOMALOUS",
            "decision": decision,
            "start_time_seconds": start,
            "end_time_seconds": end,
            "model_name": self.model_name,
            "requires_review": label in {"FIGHTING", "UNCLEAR"},
            "status": status,
            "video_quality": quality,
            "explanation": explanation,
        })

    def missing_token_result(self):
        return self._result(
            "UNCLEAR", None, "UNCERTAIN", "ERROR", None, None, "UNKNOWN",
            "HF_TOKEN is missing. Add a Hugging Face token with Inference Providers permission to .env.",
        )


class AnomalyPersistence:
    """Require consecutive calibrated anomaly windows; uncertainty resets persistence."""

    def __init__(self, min_windows=2):
        if int(min_windows) < 1:
            raise ValueError("min_windows must be at least 1")
        self.min_windows = int(min_windows)
        self._count = 0
        self._start = None
        self._end = None

    def observe(self, result: AnomalyResult):
        if result is None or result.decision != "ANOMALOUS" or result.anomaly_score is None:
            self._count, self._start, self._end = 0, None, None
            return None
        if self._count == 0:
            self._start = result.start_time_seconds
        self._count += 1
        self._end = result.end_time_seconds
        if self._count < self.min_windows:
            return None
        return {
            "start_time_seconds": self._start,
            "end_time_seconds": self._end,
            "persistence_windows": self._count,
        }


class AnomalyIncidentStore:
    """SQLite persistence with adjacent-window grouping, isolated from existing incident schema."""

    def __init__(self, path="stai_incidents.sqlite3", merge_gap_seconds=30):
        self.path = path
        self.merge_gap_seconds = float(merge_gap_seconds)
        with sqlite3.connect(self.path, timeout=5) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS anomaly_incidents (
                id TEXT PRIMARY KEY, session_id TEXT NOT NULL, camera_id TEXT NOT NULL,
                created_at TEXT NOT NULL, start_time_seconds REAL NOT NULL, end_time_seconds REAL NOT NULL,
                activity_label TEXT NOT NULL, anomaly_score REAL NOT NULL, score_type TEXT NOT NULL, decision TEXT NOT NULL,
                model_name TEXT NOT NULL, model_explanation TEXT, video_quality TEXT NOT NULL, persistence_windows INTEGER NOT NULL,
                vlm_category TEXT, vlm_summary TEXT, vlm_risk TEXT, vlm_confidence REAL,
                requires_review INTEGER NOT NULL, status TEXT NOT NULL, telegram_status TEXT NOT NULL)""")
            columns = {row[1] for row in db.execute("PRAGMA table_info(anomaly_incidents)")}
            if "activity_label" not in columns:
                db.execute("ALTER TABLE anomaly_incidents ADD COLUMN activity_label TEXT NOT NULL DEFAULT 'UNCLEAR'")
            if "model_explanation" not in columns:
                db.execute("ALTER TABLE anomaly_incidents ADD COLUMN model_explanation TEXT")

    def record_window(self, result, persistence, camera_id, session_id):
        if result.decision != "ANOMALOUS" or result.anomaly_score is None:
            raise ValueError("only scored anomalous windows can create an anomaly incident")
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.path, timeout=5) as db:
            db.row_factory = sqlite3.Row
            previous = db.execute(
                "SELECT * FROM anomaly_incidents WHERE session_id=? AND camera_id=? "
                "ORDER BY end_time_seconds DESC LIMIT 1",
                (session_id, camera_id),
            ).fetchone()
            if previous and result.start_time_seconds is not None and previous["end_time_seconds"] is not None and result.start_time_seconds - previous["end_time_seconds"] <= self.merge_gap_seconds:
                db.execute(
                    """UPDATE anomaly_incidents SET end_time_seconds=?, anomaly_score=?,
                       model_explanation=?, persistence_windows=?, requires_review=1 WHERE id=?""",
                    (
                        max(previous["end_time_seconds"], result.end_time_seconds),
                        max(previous["anomaly_score"], result.anomaly_score),
                        result.explanation,
                        persistence["persistence_windows"],
                        previous["id"],
                    ),
                )
                record = dict(
                    db.execute("SELECT * FROM anomaly_incidents WHERE id=?", (previous["id"],)).fetchone()
                )
                return record, False
            record = {
                "id": str(uuid.uuid4()), "session_id": session_id, "camera_id": camera_id,
                "created_at": now,
                "start_time_seconds": persistence["start_time_seconds"],
                "end_time_seconds": persistence["end_time_seconds"],
                "anomaly_score": result.anomaly_score, "score_type": result.score_type,
                "activity_label": result.activity_label,
                "decision": result.decision, "model_name": result.model_name,
                "model_explanation": result.explanation,
                "video_quality": result.video_quality,
                "persistence_windows": persistence["persistence_windows"],
                "vlm_category": None, "vlm_summary": None, "vlm_risk": None,
                "vlm_confidence": None, "requires_review": 1, "status": "NEW",
                "telegram_status": "DISABLED",
            }
            db.execute(
                """INSERT INTO anomaly_incidents (
                    id,session_id,camera_id,created_at,start_time_seconds,end_time_seconds,
                    activity_label,anomaly_score,score_type,decision,model_name,model_explanation,video_quality,persistence_windows,
                    vlm_category,vlm_summary,vlm_risk,vlm_confidence,requires_review,status,telegram_status)
                VALUES (
                    :id,:session_id,:camera_id,:created_at,:start_time_seconds,:end_time_seconds,
                    :activity_label,:anomaly_score,:score_type,:decision,:model_name,:model_explanation,:video_quality,:persistence_windows,
                    :vlm_category,:vlm_summary,:vlm_risk,:vlm_confidence,:requires_review,:status,:telegram_status)""",
                record,
            )
            return record, True

    def attach_vlm(self, incident_id, assessment):
        with sqlite3.connect(self.path, timeout=5) as db:
            db.execute(
                """UPDATE anomaly_incidents SET vlm_category=?, vlm_summary=?, vlm_risk=?,
                   vlm_confidence=?, requires_review=? WHERE id=?""",
                (
                    assessment.activity_category.value, assessment.description,
                    assessment.risk_level, assessment.confidence,
                    int(assessment.requires_human_review), incident_id,
                ),
            )

    def get(self, incident_id):
        with sqlite3.connect(self.path, timeout=5) as db:
            db.row_factory = sqlite3.Row
            row = db.execute("SELECT * FROM anomaly_incidents WHERE id=?", (incident_id,)).fetchone()
            return dict(row) if row else None

    def set_telegram_status(self, incident_id, status):
        if status not in {"DISABLED", "SENT", "FAILED", "SUPPRESSED", "NOT_QUALIFIED"}:
            raise ValueError("invalid notification status")
        with sqlite3.connect(self.path, timeout=5) as db:
            db.execute("UPDATE anomaly_incidents SET telegram_status=? WHERE id=?", (status, incident_id))


def build_anomaly_telegram_message(record):
    start = record["start_time_seconds"]
    end = record["end_time_seconds"]
    summary = record.get("vlm_summary") or record.get("model_explanation") or "Human verification is required."
    category = record.get("vlm_category") or "UNKNOWN"
    risk = record.get("vlm_risk") or "UNASSESSED"
    confidence = record.get("vlm_confidence")
    confidence_text = f"{confidence:.0%}" if isinstance(confidence, (int, float)) else "unavailable"
    score = record.get("anomaly_score")
    score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "NA"
    return "\n".join([
        "STAI — ANOMALY REVIEW REQUESTED", f"Incident: {record['id']}",
        f"Camera: {record['camera_id']}", f"Event time: {record['created_at']}",
        f"Approximate clip interval: {start:.1f}s–{end:.1f}s",
        f"Observed activity: {record.get('activity_label', 'UNCLEAR')}",
        f"Action classifier confidence: {score_text} ({record['score_type']})",
        f"Decision: {record['decision']} after {record['persistence_windows']} persistent windows",
        f"Model: {record['model_name']}", f"Video quality: {record['video_quality']}",
        f"VLM category/risk: {category} / {risk}; model confidence {confidence_text}",
        f"Model assessment: {summary[:400]}",
        "A classifier result is not a confirmed emergency. Please verify the footage.",
    ])
