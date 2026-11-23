"""P17.T3 acceptance: anomaly detection wired into the real store (plan Phase 17).

The TrustAnomalyDetector plugs into ``verification.add_verification_event``
through the documented hook seam: a flagged event (rapid 0.3 -> 0.77 jump)
raises before anything is applied and lands in the human review queue; after
``approve_review`` the caller may re-apply, and ``reject_review`` marks the
held change rejected (revert guidance: the score is recomputed without the
event — trivially true here because the event never applied). Normal gradual
evolution is not flagged at all.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from hermes.memory.store import MemoryStore
from hermes.memory.trust_anomaly import (
    REVIEW_STATUS_APPROVED,
    REVIEW_STATUS_PENDING,
    REVIEW_STATUS_REJECTED,
    TrustAnomalyDetector,
)
from hermes.memory.trust_card import TrustCard, serialize
from hermes.memory.types import VerificationEvent
from hermes.memory.verification import (
    TrustAnomalyError,
    add_verification_event,
    clear_anomaly_hook,
    set_anomaly_hook,
)

_SEED_TIME = datetime(2026, 11, 20, 0, 0, 0, tzinfo=UTC)


def _seed_entry(store_root: Path, entry_id: str, value: float, body: str) -> None:
    """Write one entry file whose single agent-inferred event yields ``value``.

    (``compute_trust_score`` of a single event is its value — the weight
    cancels — so a value of 0.3 seeds a trust score of 0.3.)
    """
    event = VerificationEvent(
        event_id=f"ev-{uuid4().hex[:12]}",
        timestamp=_SEED_TIME,
        source="agent-inferred",
        verifier="migration",
        value=value,
    )
    card = TrustCard(
        id=entry_id,
        type="Project",
        created_at=_SEED_TIME,
        trust_score=value,
        verification_events=[event],
        context="global",
    )
    facts_dir = store_root / "global" / "facts"
    facts_dir.mkdir(parents=True, exist_ok=True)
    (facts_dir / f"{entry_id}.md").write_text(serialize(card, body), encoding="utf-8")


@pytest.mark.integration
def test_anomaly_triggers_review(tmp_path: Path) -> None:
    """Rapid jump flagged + queued + held; gradual evolution applies cleanly."""
    store = MemoryStore(root=tmp_path / "memory-repo")
    detector = TrustAnomalyDetector(
        review_queue_path=tmp_path / "review_queue.json",
    )
    _seed_entry(store.root, "ent-rapid", 0.3, "The API rate limit is 100 req/min.")
    _seed_entry(store.root, "ent-gradual", 0.5, "Deployments run on the staging lane.")
    _seed_entry(store.root, "ent-reject", 0.5, "The cache TTL is 300 seconds.")

    assert store.get("ent-rapid").trust_score == pytest.approx(0.3)

    # wire the documented hook seam (P17.T3): flagged -> hold the event
    set_anomaly_hook(
        lambda entry_id, event, previous, current: detector.check_event(
            entry_id, event, current, previous
        ).flagged
    )
    try:
        # NORMAL evolution: 0.5 -> 0.733 (jump 0.233 < 0.3) applies untouched
        updated = add_verification_event(
            store, "ent-gradual", source="file-read", verifier="test-hook", value=0.9
        )
        assert updated.trust_score == pytest.approx((0.5 * 0.5 + 0.7 * 0.9) / 1.2)
        assert detector.pending_reviews() == []

        # ANOMALY: 0.3 -> 0.767 (jump 0.467 >= 0.3) is held pending review
        with pytest.raises(TrustAnomalyError) as excinfo:
            add_verification_event(
                store, "ent-rapid", source="user-explicit", verifier="user", value=1.0
            )
        assert excinfo.value.entry_id == "ent-rapid"
        # the event never reached the entry: score unchanged, no new events
        assert store.get("ent-rapid").trust_score == pytest.approx(0.3)
        assert len(store.get("ent-rapid").verification_events) == 1

        pending = detector.pending_reviews()
        assert [item["entry_id"] for item in pending] == ["ent-rapid"]
        assert pending[0]["status"] == REVIEW_STATUS_PENDING
        assert any("rapid trust increase" in reason for reason in pending[0]["reasons"])
    finally:
        clear_anomaly_hook()

    # HUMAN REVIEW: approve -> the caller may re-apply the held event
    assert detector.approve_review("ent-rapid") == 1
    assert detector.pending_reviews() == []
    re_applied = add_verification_event(
        store, "ent-rapid", source="user-explicit", verifier="user", value=1.0
    )
    assert re_applied.trust_score == pytest.approx((0.5 * 0.3 + 1.0) / 1.5)

    # REJECT path: the held change is marked rejected; revert guidance is to
    # recompute without the event (trivially satisfied: it was never applied)
    set_anomaly_hook(
        lambda entry_id, event, previous, current: detector.check_event(
            entry_id, event, current, previous
        ).flagged
    )
    try:
        with pytest.raises(TrustAnomalyError):
            add_verification_event(
                store, "ent-reject", source="user-explicit", verifier="user", value=1.0
            )
        assert store.get("ent-reject").trust_score == pytest.approx(0.5)
    finally:
        clear_anomaly_hook()
    assert detector.reject_review("ent-reject") == 1
    queue = (tmp_path / "review_queue.json").read_text(encoding="utf-8")
    assert '"status": "' + REVIEW_STATUS_REJECTED + '"' in queue
    assert REVIEW_STATUS_APPROVED in queue
    assert store.get("ent-reject").trust_score == pytest.approx(0.5)
