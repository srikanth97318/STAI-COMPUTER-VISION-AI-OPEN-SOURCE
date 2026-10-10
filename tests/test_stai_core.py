import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from stai_core import (
    ActivityAssessment,
    ActivityCategory,
    IncidentStatus,
    IncidentStore,
    VideoQuality,
    build_telegram_message,
    deliver_telegram,
    should_notify,
)


def assessment(category="NORMAL", risk="LOW", review=False, quality="GOOD"):
    return ActivityAssessment.validate({
        "activity_category": category,
        "description": "Possible observable event; verify with a human.",
        "risk_level": risk,
        "confidence": 0.72,
        "evidence": ["visible cue"],
        "requires_human_review": review,
        "video_quality": quality,
    })


class AssessmentTests(unittest.TestCase):
    def test_validates_schema_and_normalizes_types(self):
        result = ActivityAssessment.validate({
            "activity_category": "NORMAL", "description": "Walking.", "risk_level": "LOW",
            "confidence": 0, "evidence": [], "requires_human_review": False, "video_quality": "GOOD",
        })
        self.assertEqual(result.activity_category, ActivityCategory.NORMAL)

    def test_rejects_invalid_category_confidence_and_missing_fields(self):
        raw = {"activity_category": "CRIME"}
        with self.assertRaises(ValueError):
            ActivityAssessment.validate(raw)
        for confidence in (-0.1, 1.1, True, float("nan")):
            value = {"activity_category": "NORMAL", "description": "x", "risk_level": "LOW",
                     "confidence": confidence, "evidence": [], "requires_human_review": False, "video_quality": "GOOD"}
            with self.subTest(confidence=confidence), self.assertRaises(ValueError):
                ActivityAssessment.validate(value)

    def test_ordinary_social_and_quality_only_do_not_trigger_emergency_notification(self):
        self.assertFalse(should_notify(assessment("NORMAL", "CRITICAL", True)))
        self.assertFalse(should_notify(assessment("SOCIAL_INTERACTION", "HIGH", True)))
        self.assertFalse(should_notify(assessment("POSSIBLE_FALL", "HIGH", True), quality_warning_only=True))
        self.assertTrue(should_notify(assessment("POSSIBLE_FALL", "HIGH", True)))


class IncidentTests(unittest.TestCase):
    def test_deduplicates_and_guards_terminal_transitions(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as temp:
            store = IncidentStore(f"{temp}/incidents.db", dedupe_seconds=30)
            event = assessment("POSSIBLE_FALL", "HIGH", True)
            incident, created = store.record(event, now=start)
            repeated, repeated_created = store.record(event, now=start + timedelta(seconds=10))
            self.assertTrue(created)
            self.assertFalse(repeated_created)
            self.assertEqual(repeated["id"], incident["id"])
            store.update_status(incident["id"], IncidentStatus.ACKNOWLEDGED.value)
            store.update_status(incident["id"], IncidentStatus.RESOLVED.value)
            with self.assertRaises(ValueError):
                store.update_status(incident["id"], IncidentStatus.NEW.value)

    def test_telegram_message_labels_assessment_and_only_accepts_http_links(self):
        with tempfile.TemporaryDirectory() as temp:
            store = IncidentStore(f"{temp}/incidents.db")
            incident, _ = store.record(assessment("POSSIBLE_FALL", "HIGH", True))
            message = build_telegram_message(incident, "javascript:bad")
            self.assertIn("STAI — HUMAN REVIEW REQUESTED", message)
            self.assertIn("Model confidence: 72%", message)
            self.assertNotIn("javascript:", message)

    def test_telegram_delivery_is_opt_in_and_failures_are_contained(self):
        calls = []
        def post(*args, **kwargs):
            calls.append((args, kwargs))
            return type("Response", (), {"status_code": 200})()

        self.assertEqual(deliver_telegram("hello", "token", "chat", False, post), "DISABLED")
        self.assertEqual(calls, [])
        self.assertEqual(deliver_telegram("hello", "token", "chat", True, post), "SENT")
        self.assertEqual(calls[0][1]["timeout"], 5)
        self.assertEqual(deliver_telegram("hello", "token", "chat", True, lambda *a, **k: type("Response", (), {"status_code": 429})()), "FAILED")
        self.assertEqual(deliver_telegram("hello", "secret-token", "chat", True, lambda *a, **k: (_ for _ in ()).throw(TimeoutError("secret-token"))), "FAILED")


if __name__ == "__main__":
    unittest.main()
