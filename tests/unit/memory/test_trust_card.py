"""P1.T1 acceptance: TrustCard YAML frontmatter + trust scoring (plan Section 2.4).

``parse``/``serialize`` round-trip every field (datetimes as ISO 8601 with a
``Z`` suffix); ``compute_trust_score`` is the plan's weighted average with a
0.5 migration default for unverified entries; verification-event weights come
from the plan's source-weight table; ``MemoryEntry`` clamps trust_score to
[0, 1] and validates its fields.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from hermes.memory.entry import MemoryEntry
from hermes.memory.trust_card import TrustCard, compute_trust_score, parse, serialize
from hermes.memory.types import SOURCE_WEIGHTS, VerificationEvent

_UTC = UTC


def _card() -> TrustCard:
    """Fully-populated card exercising every serialized field."""
    created = datetime(2026, 11, 20, 10, 30, 0, tzinfo=_UTC)
    verified = datetime(2026, 11, 20, 11, 0, 0, 123456, tzinfo=_UTC)
    return TrustCard(
        id="mem-001",
        type="Project",
        created_at=created,
        last_verified_at=verified,
        trust_score=0.75,
        verification_events=[
            VerificationEvent(
                event_id="ev-1",
                timestamp=created,
                source="user-explicit",
                verifier="alice",
                evidence="stated in review",
            ),
            VerificationEvent(
                event_id="ev-2",
                timestamp=verified,
                source="test-green",
                verifier="ci",
                evidence="pytest tests/unit/memory",
                value=0.0,  # negative anchor: the red test disproves the claim
            ),
        ],
        context="session-42",
        scope=["python", "memory"],
        ttl_days=30,
        tags=["api", "design"],
        pii_checked=True,
        attachments=["notes/a.md"],
    )


@pytest.mark.unit
def test_parse_roundtrip() -> None:
    """parse -> serialize -> parse preserves every field incl. datetimes."""
    card = _card()
    body = "The parser **must** reject naive timestamps.\n\n- item 2\n"
    text = serialize(card, body)

    assert text.startswith("---\n"), "frontmatter block must be present"
    assert "2026-11-20T10:30:00Z" in text, "ISO 8601 with Z suffix"
    assert "2026-11-20T11:00:00.123456Z" in text, "microseconds survive the roundtrip"

    parsed_card, parsed_body = parse(text)
    assert parsed_card == card
    assert parsed_body == body

    # serialize is idempotent: parsing the rendered card twice is stable.
    text2 = serialize(parsed_card, parsed_body)
    assert parse(text2) == (parsed_card, parsed_body)


@pytest.mark.unit
def test_compute_trust_score_weighted_avg() -> None:
    """score = sum(w_i * v_i) / sum(w_i), clamped to [0, 1]."""
    now = datetime.now(_UTC)

    def ev(source: str, value: float = 1.0) -> VerificationEvent:
        return VerificationEvent(
            event_id=f"ev-{source}-{value}",
            timestamp=now,
            source=source,  # type: ignore[arg-type]
            verifier="tester",
            value=value,
        )

    mixed = [ev("user-explicit"), ev("tool-output", value=0.0)]
    expected = 1.0 / (1.0 + 0.85)  # (1.0*1.0 + 0.85*0.0) / (1.0 + 0.85)
    assert compute_trust_score(mixed) == pytest.approx(expected)

    assert compute_trust_score([ev("test-green")]) == 1.0
    assert compute_trust_score([ev("llm"), ev("code-review-human")]) == 1.0
    assert compute_trust_score([ev("file-read", value=0.0)]) == 0.0

    # Documented migration default for unverified (empty) histories.
    assert compute_trust_score([]) == 0.5


@pytest.mark.unit
def test_verification_event_weights() -> None:
    """Weights default to the plan's table; unknown sources are rejected."""
    assert SOURCE_WEIGHTS == {
        "user-explicit": 1.0,
        "test-green": 1.0,
        "code-review-human": 0.95,
        "tool-output": 0.85,
        "file-read": 0.7,
        "agent-inferred": 0.5,
        "llm": 0.3,
    }
    now = datetime.now(_UTC)
    for source, weight in SOURCE_WEIGHTS.items():
        event = VerificationEvent(
            event_id=f"ev-{source}",
            timestamp=now,
            source=source,
            verifier="tester",
            evidence="e",
        )
        assert event.weight == weight, f"{source} must default to {weight}"

    # Explicit weights override the table default.
    override = VerificationEvent(
        event_id="ev-x", timestamp=now, source="llm", verifier="v", weight=0.9
    )
    assert override.weight == 0.9

    # A test-red negative anchor is modelled as an event with value 0.0.
    anchor = VerificationEvent(
        event_id="ev-red",
        timestamp=now,
        source="test-green",
        verifier="ci",
        evidence="failing test",
        value=0.0,
    )
    assert anchor.value == 0.0

    with pytest.raises(ValueError):
        VerificationEvent(event_id="ev-bad", timestamp=now, source="test-red", verifier="ci")


@pytest.mark.unit
def test_memory_entry_invariants() -> None:
    """MemoryEntry clamps trust_score to [0, 1] and validates its fields."""
    now = datetime.now(_UTC)
    entry = MemoryEntry(
        id="mem-1",
        type="Reference",
        text="note",
        created_at=now,
        trust_score=1.7,
        verification_events=[
            VerificationEvent(
                event_id="ev-1", timestamp=now, source="file-read", verifier="agent"
            )
        ],
    )
    assert entry.trust_score == 1.0  # clamped into [0, 1]
    assert MemoryEntry(
        id="mem-2", type="Feedback", text="note", created_at=now, trust_score=-0.5
    ).trust_score == 0.0
    assert MemoryEntry(id="mem-3", type="User", text="note", created_at=now).trust_score == 0.5

    with pytest.raises(ValueError):
        MemoryEntry(id="mem-4", type="Bogus", text="t", created_at=now)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        # Naive datetime must be rejected.
        MemoryEntry(id="mem-5", type="User", text="t", created_at=datetime(2026, 1, 1))


@pytest.mark.unit
def test_compute_trust_score_zero_weight_degenerate() -> None:
    """Zero-weight histories fall back to the 0.5 migration default (P20.T6).

    Mutation-gate strengthening for the Tier-1 weighted-trust-score logic:
    the degenerate all-zero-weight branch (``total_weight <= 0``) and a
    three-source weighted mix pinning the exact numerator/denominator roles
    (``sum(w_i * v_i) / sum(w_i)``) — swapping weights with values, summing
    only values, or dropping the fallback all die here.
    """
    now = datetime.now(_UTC)

    def ev(source: str, value: float, weight: float | None = None) -> VerificationEvent:
        kwargs: dict[str, object] = {}
        if weight is not None:
            kwargs["weight"] = weight
        return VerificationEvent(
            event_id=f"ev-{source}-{value}-{weight}",
            timestamp=now,
            source=source,  # type: ignore[arg-type]
            verifier="tester",
            value=value,
            **kwargs,
        )

    # every weight explicitly 0.0: no usable evidence -> migration default
    zero_weighted = [
        ev("user-explicit", 1.0, weight=0.0),
        ev("test-green", 1.0, weight=0.0),
    ]
    assert compute_trust_score(zero_weighted) == 0.5
    # a single zero-weight event behaves identically
    assert compute_trust_score([ev("llm", 1.0, weight=0.0)]) == 0.5

    # three-source mix: numerator and denominator are exact
    mix = [
        ev("user-explicit", 1.0),  # weight 1.0
        ev("tool-output", 0.0),  # weight 0.85 (negative anchor)
        ev("file-read", 1.0),  # weight 0.7
    ]
    expected = (1.0 * 1.0 + 0.85 * 0.0 + 0.7 * 1.0) / (1.0 + 0.85 + 0.7)
    assert compute_trust_score(mix) == pytest.approx(expected)

    # an explicit non-default weight participates in both sums
    weighted = [
        ev("user-explicit", 0.5),
        ev("test-green", 1.0, weight=0.5),
    ]
    expected_weighted = (1.0 * 0.5 + 1.0 * 0.5) / (1.0 + 0.5)
    assert compute_trust_score(weighted) == pytest.approx(expected_weighted)


@pytest.mark.unit
def test_parse_rejects_invalid_frontmatter() -> None:
    """parse raises on missing required fields and malformed values (P20.T6).

    Mutation-gate strengthening for the Tier-1 parse contract: the three
    required fields are each enforced, unknown entry types are rejected, and
    list-valued frontmatter fields reject non-list data.
    """
    # text without a frontmatter block has no required fields at all
    with pytest.raises(ValueError, match="missing required field"):
        parse("just a body, no frontmatter block")

    def card_yaml(**overrides: object) -> str:
        fields: dict[str, object] = {
            "id": "mem-100",
            "type": "User",
            "created_at": "2026-11-20T10:30:00Z",
        }
        fields.update(overrides)
        kept = {key: value for key, value in fields.items() if value is not None}
        lines = [f"{key}: {value}" for key, value in kept.items()]
        return "---\n" + "\n".join(lines) + "\n---\n\nbody\n"

    # the minimal valid card parses
    card, body = parse(card_yaml())
    assert card.id == "mem-100" and card.type == "User" and body == "body\n"

    # each required field is enforced
    for missing in ("id", "type", "created_at"):
        with pytest.raises(ValueError, match=missing):
            parse(card_yaml(**{missing: None}))
    with pytest.raises(ValueError, match="unknown entry type"):
        parse(card_yaml(type="Bogus"))
    # list-valued fields reject scalars
    with pytest.raises(ValueError, match="scope"):
        parse(card_yaml(scope="not-a-list"))
    with pytest.raises(ValueError, match="verification_events"):
        parse(card_yaml(verification_events={"event_id": "ev-1"}))
