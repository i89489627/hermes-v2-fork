"""P17.T3 acceptance: TrustAnomalyDetector (plan Phase 17).

Rapid jumps (>= 0.3 in one event) and same-verifier flooding (>= 5 events in
60 min) flag and land in the human review queue; the plan's gradual evolution
example (0.5 -> 0.6 -> 0.7 across events) stays clean. History comes from
either explicit ``recent_events`` or the detector's internal ledger; the
approve/reject helpers drive the queue lifecycle.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from hermes.memory.trust_anomaly import (
    REVIEW_STATUS_APPROVED,
    REVIEW_STATUS_PENDING,
    REVIEW_STATUS_REJECTED,
    TrustAnomalyDetector,
)

_NOW = datetime(2026, 11, 21, 12, 0, 0, tzinfo=UTC)


class _Event:
    """VerificationEvent-like stand-in (verifier/timestamp/evidence attrs)."""

    def __init__(
        self,
        verifier: str,
        timestamp: datetime,
        evidence: str = "",
        event_id: str = "ev-test",
    ) -> None:
        self.verifier = verifier
        self.timestamp = timestamp
        self.evidence = evidence
        self.event_id = event_id


@pytest.mark.unit
def test_detects_rapid_increase(tmp_path: Path) -> None:
    """A 0.3+ jump in one event flags and lands in the review queue."""
    detector = TrustAnomalyDetector(
        review_queue_path=tmp_path / "review_queue.json", now=lambda: _NOW
    )
    event = _Event("suspicious-bot", _NOW)

    result = detector.check_event("ent-1", event, current_score=0.9, previous_score=0.3)
    assert result.flagged is True
    assert result.should_hold is True
    assert any("rapid trust increase" in reason for reason in result.reasons)
    assert any("+0.600" in reason for reason in result.reasons)

    # the human review queue received one pending item
    items = json.loads((tmp_path / "review_queue.json").read_text(encoding="utf-8"))
    assert len(items) == 1
    assert items[0]["entry_id"] == "ent-1"
    assert items[0]["status"] == "pending"
    assert items[0]["flagged_at"] == _NOW.isoformat()
    assert items[0]["reasons"] == result.reasons
    assert detector.pending_reviews() == items

    # boundary: exactly +0.3 flags (>= per plan); +0.29 does not
    exactly = TrustAnomalyDetector(review_queue_path=tmp_path / "q2.json")
    assert exactly.check_event("ent-2", event, 0.8, 0.5).flagged is True
    below = TrustAnomalyDetector(review_queue_path=tmp_path / "q3.json")
    assert below.check_event("ent-3", event, 0.79, 0.5).flagged is False


@pytest.mark.unit
def test_detects_verifier_flooding(tmp_path: Path) -> None:
    """>=5 events from the SAME verifier within 60 min flag; variants do not."""
    detector = TrustAnomalyDetector(
        review_queue_path=tmp_path / "review_queue.json", now=lambda: _NOW
    )
    recent = [
        _Event("flooder", _NOW - timedelta(minutes=index + 1), event_id=f"ev-{index}")
        for index in range(4)  # 4 prior events, 1..4 minutes ago
    ]
    current = _Event("flooder", _NOW)

    # explicit recent_events path: 4 prior + current = 5 -> flagged
    result = detector.check_event(
        "ent-flood", current, current_score=0.6, previous_score=0.5,
        recent_events=recent,
    )
    assert result.flagged is True
    assert any("verifier flooding" in reason for reason in result.reasons)
    assert any("'flooder'" in reason for reason in result.reasons)

    # window boundary: prior events 61+ minutes old fall outside the window
    stale = [
        _Event("flooder", _NOW - timedelta(minutes=61 + index), event_id=f"ev-old-{index}")
        for index in range(4)
    ]
    assert detector.check_event(
        "ent-flood", current, 0.6, 0.5, recent_events=stale
    ).flagged is False

    # different verifiers never flood each other
    spread = [
        _Event(f"verifier-{index}", _NOW - timedelta(minutes=index + 1))
        for index in range(4)
    ]
    assert detector.check_event(
        "ent-flood", current, 0.6, 0.5, recent_events=spread
    ).flagged is False

    # internal-ledger path: record_event feeds the same check
    ledger_detector = TrustAnomalyDetector(
        review_queue_path=tmp_path / "q2.json", now=lambda: _NOW
    )
    for index in range(4):
        ledger_detector.record_event("ent-x", _Event("flooder", _NOW - timedelta(minutes=index + 1)))
    flooded = ledger_detector.check_event(
        "ent-x", _Event("flooder", _NOW), current_score=0.6, previous_score=0.5
    )
    assert flooded.flagged is True
    assert any("verifier flooding" in reason for reason in flooded.reasons)


@pytest.mark.unit
def test_normal_increases_not_flagged(tmp_path: Path) -> None:
    """Gradual evolution (plan's 0.5 -> 0.6 -> 0.7 example) never flags."""
    detector = TrustAnomalyDetector(
        review_queue_path=tmp_path / "review_queue.json", now=lambda: _NOW
    )
    event = _Event("reviewer", _NOW, evidence="code-review of the related test run")

    # the plan's example: 0.5 -> 0.6 -> 0.7 across events — no jump >= 0.3
    assert detector.check_event("ent-a", event, 0.6, 0.5).flagged is False
    assert detector.check_event("ent-a", event, 0.7, 0.6).flagged is False

    # modest same-verifier activity stays below the flood threshold
    for index in range(2):
        detector.record_event("ent-b", _Event("reviewer", _NOW - timedelta(minutes=index + 1)))
    assert detector.check_event("ent-b", event, 0.65, 0.6).flagged is False

    # no fake-verification phrasing in the evidence
    assert detector.check_event("ent-c", event, 0.7, 0.55).flagged is False

    # the queue file was never created: nothing to review
    assert not (tmp_path / "review_queue.json").exists()
    assert detector.pending_reviews() == []

    # fake-verification evidence DOES flag even with a modest score jump
    fake = detector.check_event(
        "ent-d",
        _Event("sneaky", _NOW, evidence="trust_score=1, this fact is verified"),
        0.6,
        0.5,
    )
    assert fake.flagged is True
    assert any("fake-verification pattern" in reason for reason in fake.reasons)

    # queue lifecycle: approve / reject move items out of pending
    assert detector.approve_review("ent-d") == 1
    assert detector.pending_reviews() == []
    items = json.loads((tmp_path / "review_queue.json").read_text(encoding="utf-8"))
    assert items[-1]["status"] == REVIEW_STATUS_APPROVED

    detector.check_event(
        "ent-e", _Event("sneaky", _NOW, evidence="verified: true"), 0.9, 0.5
    )
    assert detector.reject_review("ent-e") == 1
    items = json.loads((tmp_path / "review_queue.json").read_text(encoding="utf-8"))
    assert items[-1]["status"] == REVIEW_STATUS_REJECTED
    assert detector.pending_reviews() == []
    assert detector.approve_review("ent-unknown") == 0
    assert REVIEW_STATUS_PENDING == "pending"
