"""P20.T2 property: trust scores stay in [0, 1] and value-1.0 events never hurt.

For arbitrary verification-event histories (random sources from the plan
Section 2.4 table, random weights in [0, 1], random values in [0, 1], plus
the weight-derivation sentinel), :func:`compute_trust_score` always returns
a clamped float in the closed interval [0, 1] — including the documented
degenerate defaults (empty history -> 0.5, all-zero-weight history -> 0.5).

Appending invariants (:func:`add_verification_event` on the card):

- the task's condition: appending a value-1.0 event whose weight is at
  least the current weighted mean weight never decreases the score;
- the precise invariant (documented): appending a value-1.0 event NEVER
  decreases the score, for ANY weight in [0, 1]. Derivation: with
  ``S = sum(w_i * v_i)``, ``Tw = sum(w_i)`` (Tw > 0) and a new event of
  weight W, value 1.0::

      score_after >= score_before  <=>  (S + W) / (Tw + W) >= S / Tw
                                   <=>  W * (Tw - S) >= 0,

  and ``Tw - S >= 0`` always holds because every value ``v_i <= 1``. The
  task's weight condition is therefore a (conservative) sufficient subset;
  the test asserts the precise invariant and pins the task's condition with
  explicit examples. Degenerate histories keep the documented 0.5 default,
  which a value-1.0 event (score 1.0 or unchanged 0.5) can only reach or
  raise.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from hermes.memory.trust_card import TrustCard, add_verification_event, compute_trust_score
from hermes.memory.types import SOURCE_WEIGHTS, WEIGHT_FROM_SOURCE, VerificationEvent

_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

_SOURCE = st.sampled_from(sorted(SOURCE_WEIGHTS))
_WEIGHT = st.one_of(
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    st.just(WEIGHT_FROM_SOURCE),  # sentinel: derive the weight from the source
)
_VALUE = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

_EVENTS = st.lists(
    st.builds(
        VerificationEvent,
        event_id=st.text(min_size=1, max_size=12),
        timestamp=st.just(_NOW),
        source=_SOURCE,
        verifier=st.just("property-fake"),
        weight=_WEIGHT,
        value=_VALUE,
    ),
    max_size=8,
)


def _weighted_mean_weight(events: list[VerificationEvent]) -> float:
    """The weighted mean weight of a history (0.0 for the degenerate cases)."""
    total = sum(event.weight for event in events)
    if not events or total <= 0.0:
        return 0.0
    return sum(event.weight * event.weight for event in events) / total


@pytest.mark.property
@given(events=_EVENTS)
@example(events=[])
@example(events=[VerificationEvent("ev-0", _NOW, "user-explicit", "v", WEIGHT_FROM_SOURCE, value=0.0)])
@example(events=[VerificationEvent("ev-0", _NOW, "llm", "v", 0.0, value=1.0)])
@settings(max_examples=1000, deadline=None)
def test_score_always_in_range(events: list[VerificationEvent]) -> None:
    """compute_trust_score stays in [0, 1] for arbitrary event histories."""
    score = compute_trust_score(events)
    assert 0.0 <= score <= 1.0
    if not events:
        assert score == 0.5  # documented migration default
    if events and sum(event.weight for event in events) <= 0.0:
        assert score == 0.5  # documented degenerate default


@pytest.mark.property
@given(events=_EVENTS, new_weight=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
@example(events=[], new_weight=0.0)
@example(events=[], new_weight=1.0)
@example(
    events=[VerificationEvent("ev-0", _NOW, "user-explicit", "v", WEIGHT_FROM_SOURCE, value=0.0)],
    new_weight=1.0,
)  # weight >= the current weighted mean weight (task condition, non-vacuous)
@example(
    events=[VerificationEvent("ev-0", _NOW, "llm", "v", 0.3, value=1.0)],
    new_weight=0.3,
)
@settings(max_examples=1000, deadline=None)
def test_add_event_never_decreases_if_higher_weight(
    events: list[VerificationEvent], new_weight: float
) -> None:
    """Appending a value-1.0 event never decreases the trust score.

    The precise invariant (see the module docstring derivation) holds for
    ANY new weight; the task's sufficient condition — new weight at least
    the current weighted mean weight — is covered by construction and by
    the pinned examples.
    """
    score_before = compute_trust_score(events)
    card = TrustCard(
        id="prop-card",
        type="User",
        created_at=_NOW,
        trust_score=score_before,
        verification_events=list(events),
    )
    new_event = VerificationEvent(
        event_id="ev-new",
        timestamp=_NOW,
        source="llm",
        verifier="property-fake",
        weight=new_weight,
        value=1.0,
    )

    add_verification_event(card, new_event)

    assert 0.0 <= card.trust_score <= 1.0
    assert card.trust_score >= score_before - 1e-9
    # the task's stated condition is a subset of the precise invariant;
    # verify it holds exactly as stated whenever it applies
    if new_weight >= _weighted_mean_weight(events):
        assert card.trust_score >= score_before - 1e-9
