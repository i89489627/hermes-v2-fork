"""P2.T1 acceptance: verification event tracking with trust score recompute.

Covers :func:`hermes.memory.verification.add_verification_event` on both
targets (store = persisted via ``record_event`` + one Git commit; entry =
in-memory recompute), the weighted-average score rule and the plan 2.4
weight ordering (user-explicit 1.0 / test-green 1.0 above llm 0.3).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from hermes.memory.entry import MemoryEntry
from hermes.memory.store import MemoryStore
from hermes.memory.trust_card import parse
from hermes.memory.types import SOURCE_WEIGHTS, VerificationEvent, VerificationSource
from hermes.memory.verification import add_verification_event


def _make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(root=tmp_path / "memory-repo")


def _fresh_entry(base: VerificationEvent, entry_id: str) -> MemoryEntry:
    """Entry whose history starts from a private copy of ``base``."""
    return MemoryEntry(
        id=entry_id,
        type="User",
        text="candidate fact",
        created_at=datetime.now(UTC),
        verification_events=[
            VerificationEvent(
                event_id=base.event_id,
                timestamp=base.timestamp,
                source=base.source,
                verifier=base.verifier,
                weight=base.weight,
                evidence=base.evidence,
                value=base.value,
            )
        ],
    )


@pytest.mark.unit
def test_add_event_updates_score(tmp_path: Path) -> None:
    """A recorded event recomputes the score and persists (store target)."""
    store = _make_store(tmp_path)
    entry = store.add("user prefers pnpm", type="User")
    assert len(entry.verification_events) >= 1  # every entry starts verified
    assert entry.trust_score == 1.0  # single agent-inferred (w=0.5, v=1.0) event

    updated = add_verification_event(
        store,
        entry.id,
        source="llm",
        verifier="extractor",
        evidence="spot check by model",
        value=0.5,
    )
    expected = (0.5 * 1.0 + 0.3 * 0.5) / (0.5 + 0.3)  # weighted avg: 0.8125
    assert updated.trust_score == pytest.approx(expected)

    reread = store.get(entry.id)
    assert reread is not None
    assert reread.trust_score == pytest.approx(expected)  # persisted
    assert [event.source for event in reread.verification_events] == [
        "agent-inferred",
        "llm",
    ]
    assert reread.last_verified_at == reread.verification_events[-1].timestamp

    path = tmp_path / "memory-repo" / "global" / "facts" / f"{entry.id}.md"
    card, _body = parse(path.read_text(encoding="utf-8"))
    assert card.trust_score == pytest.approx(expected)
    assert len(card.verification_events) == 2
    assert [commit.message for commit in store.git.log(path=path)][0] == (
        f"[memory] record-event {entry.id}"
    )

    # entry target: in-memory only — the store copy is untouched, no commit
    before_commits = len(store.git.log())
    add_verification_event(
        reread, source="test-green", verifier="pytest", evidence="suite green"
    )
    assert reread.trust_score == pytest.approx((0.5 + 0.15 + 1.0) / 1.8)
    assert store.get(entry.id) is not None
    assert store.get(entry.id).trust_score == pytest.approx(expected)  # type: ignore[union-attr]
    assert len(store.git.log()) == before_commits

    with pytest.raises(KeyError):
        add_verification_event(store, "2026-01-01-missing-entry-ffffffff")


@pytest.mark.unit
def test_multiple_events_weighted_avg(tmp_path: Path) -> None:
    """Score after N events is sum(w_i * v_i) / sum(w_i), persisted each time."""
    store = _make_store(tmp_path)
    entry = store.add("deploys are frozen on fridays", type="Project")
    assert entry.trust_score == pytest.approx(0.5 / 0.5)

    add_verification_event(
        store, entry.id, source="llm", verifier="extractor", value=0.0
    )
    expected_2 = (0.5 * 1.0 + 0.3 * 0.0) / (0.5 + 0.3)  # 0.625
    got = store.get(entry.id)
    assert got is not None and got.trust_score == pytest.approx(expected_2)

    add_verification_event(
        store, entry.id, source="test-green", verifier="ci", value=1.0
    )
    expected_3 = (0.5 * 1.0 + 0.3 * 0.0 + 1.0 * 1.0) / (0.5 + 0.3 + 1.0)  # 0.833...
    final = store.get(entry.id)
    assert final is not None and final.trust_score == pytest.approx(expected_3)
    assert len(final.verification_events) == 3
    assert all(event.weight == SOURCE_WEIGHTS[event.source] for event in final.verification_events)

    path = tmp_path / "memory-repo" / "global" / "facts" / f"{entry.id}.md"
    card, _body = parse(path.read_text(encoding="utf-8"))
    assert card.trust_score == pytest.approx(expected_3)


@pytest.mark.unit
def test_user_explicit_higher_than_llm() -> None:
    """Same confirming event value: user-explicit (1.0) outranks llm (0.3)."""
    timestamp = datetime.now(UTC)

    def base(event_id: str) -> VerificationEvent:
        return VerificationEvent(
            event_id=event_id,
            timestamp=timestamp,
            source="file-read",
            verifier="crawler",
            value=0.6,  # weak shared evidence
        )

    user_entry = _fresh_entry(base("ev-base-a"), "2026-01-01-entry-a")
    llm_entry = _fresh_entry(base("ev-base-b"), "2026-01-01-entry-b")

    add_verification_event(
        user_entry, source="user-explicit", verifier="user", evidence="user said so"
    )
    add_verification_event(
        llm_entry, source="llm", verifier="extractor", evidence="model guessed"
    )

    w_user = SOURCE_WEIGHTS["user-explicit"]
    w_llm = SOURCE_WEIGHTS["llm"]
    base_partial = SOURCE_WEIGHTS["file-read"] * 0.6
    expected_user = (base_partial + w_user * 1.0) / (SOURCE_WEIGHTS["file-read"] + w_user)
    expected_llm = (base_partial + w_llm * 1.0) / (SOURCE_WEIGHTS["file-read"] + w_llm)

    assert user_entry.trust_score == pytest.approx(expected_user)
    assert llm_entry.trust_score == pytest.approx(expected_llm)
    assert user_entry.trust_score > llm_entry.trust_score


@pytest.mark.unit
def test_test_green_highest_weight() -> None:
    """test-green carries the top weight (1.0) of all verification sources."""
    assert SOURCE_WEIGHTS["test-green"] == 1.0
    assert SOURCE_WEIGHTS["test-green"] == max(SOURCE_WEIGHTS.values())

    timestamp = datetime.now(UTC)
    scores: dict[VerificationSource, float] = {}
    for index, source in enumerate(sorted(SOURCE_WEIGHTS)):
        entry = _fresh_entry(
            VerificationEvent(
                event_id=f"ev-base-{index}",
                timestamp=timestamp,
                source="file-read",
                verifier="crawler",
                value=0.5,  # weak shared evidence
            ),
            f"2026-01-01-entry-{index}",
        )
        add_verification_event(entry, source=source, verifier="anchor", evidence="e")
        scores[source] = entry.trust_score

    # weighted average is monotone in the confirming event's weight
    by_weight = sorted(SOURCE_WEIGHTS, key=lambda s: SOURCE_WEIGHTS[s])
    for weaker, stronger in zip(by_weight, by_weight[1:], strict=False):
        assert scores[stronger] >= scores[weaker]
    assert scores["test-green"] == max(scores.values())
    assert scores["test-green"] == scores["user-explicit"]  # both weight 1.0
    assert scores["test-green"] > scores["tool-output"]
    assert scores["test-green"] > scores["llm"]
