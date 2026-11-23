"""TopicDetector: classify conversation topic shifts for virtual sessions (plan P7.T3).

Phase 7: "TopicDetector: detect topic shifts in conversation (main model
classification). Every N messages, compare vs previous topic. On topic
shift: create virtual session boundary." The detector is the classification
SEAM the session manager (P7.T4) consumes: every ``max_messages`` messages
it classifies the current conversation topic and decides whether the
conversation moved off the previous topic.

Model-injected classification (the extractor/gate pattern, plan r3)
------------------------------------------------------------------------------
``topic_model`` is the injected main-model callable (``prompt -> raw
completion``) — the same single-model architecture as
:class:`hermes.memory.extractor.ExtractorAgent` and
:class:`hermes.memory.gating.CBDCGate`. The prompt asks the model to
"classify the conversation topic in <=6 words"; the model's output IS the
topic (normalized whitespace), and the RULES only handle the comparison
against ``prev_topic``. When no model is injected, the default
classification path is fully LOCAL and deterministic — no LLM call, no
network, no agent stack import: :func:`_lexical_topic` reduces the recent
messages to their most frequent non-stopword terms (top 6, frequency then
first-occurrence order) — the lexical topic of the conversation.

The comparison — the same for both classification paths (P7.T3 design:
"comparing recent messages against prev_topic") — is character-trigram
TF-IDF cosine similarity between the CONTENT of the recent messages (their
non-stopword terms joined in order, the lexical content profile) and
``prev_topic``, reusing
:class:`hermes.memory.conflict_detector.TfidfTrigramEmbedder` (the P8.T1
default embedder — deterministic, pure python). ``shifted`` when the
similarity falls below ``shift_threshold`` (default 0.3). Comparing the
message CONTENT rather than the topic label keeps the decision robust even
when an injected model labels the same subject with lexically different
words, and when one-off chatter words dilute the 6-word topic label.

Decision + confidence
------------------------------------------------------------------------------
``shifted = similarity < shift_threshold``. ``confidence`` is confidence IN
THE REPORTED DECISION: ``1 - similarity`` (clamped to [0, 1]) for a shift —
the plan's "confidence = 1 - similarity" — and ``similarity`` for a
continuation (a clear continuation at similarity 0.8 must not report a
confidence of 0.2). With no ``prev_topic`` (session start) there is nothing
to compare: no shift is possible and confidence is 1.0.

Latency: the default path is pure string + dict arithmetic (sub-millisecond
at plan scale) — the <500ms per-detection acceptance holds trivially.
Injected-model latency is the model's; the detector never retries and never
blocks beyond the model call itself.

An injected ``similarity_fn`` (``(content_profile, previous_topic) -> float
in [0, 1]``) replaces the trigram cosine for the comparison; non-finite
results degrade to 0.0 (treated as a full shift) and out-of-range values
are clamped, mirroring the gate's NaN guard.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

from hermes.memory.conflict_detector import TfidfTrigramEmbedder

__all__ = ["DEFAULT_SHIFT_THRESHOLD", "ShiftResult", "TopicDetector"]

logger = logging.getLogger(__name__)

#: Similarity below which the conversation is deemed off the previous topic.
DEFAULT_SHIFT_THRESHOLD: Final[float] = 0.3

#: The topic is at most this many words (plan prompt: "in <=6 words").
_TOPIC_MAX_WORDS: Final[int] = 6

#: Classification prompt for the injected model (plan P7.T3 wording).
TOPIC_SYSTEM_PROMPT: Final[str] = """\
You are the hermes-v2 topic detector (plan Phase 7). Classify the \
conversation topic of the recent messages in <=6 words. Return ONLY the \
topic phrase — no prose, no punctuation, no markdown."""

#: Prompt framing around the message list.
_MESSAGES_HEADER: Final[str] = "Recent messages (oldest first):"
_MESSAGES_FOOTER: Final[str] = "Respond with ONLY the topic (<= 6 words)."

_WORD_RE: Final[re.Pattern[str]] = re.compile(r"[a-z0-9']+")

#: Function words carry no topic signal; the lexical topic is built from
#: content words only. Kept deliberately small — over-filtering would strip
#: real subject words ("new", "old", "first", ... matter for topics).
_STOPWORDS: Final[frozenset[str]] = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "so", "as",
    "at", "by", "for", "from", "in", "into", "of", "off", "on", "onto", "out",
    "over", "to", "up", "with", "about", "after", "before", "between", "during",
    "is", "are", "was", "were", "be", "been", "being", "am", "do", "does",
    "did", "done", "has", "have", "had", "will", "would", "shall", "should",
    "can", "could", "may", "might", "must", "it", "its", "this", "that",
    "these", "those", "there", "here", "they", "them", "their", "we", "us",
    "our", "you", "your", "i", "me", "my", "he", "she", "his", "her", "what",
    "which", "who", "whom", "how", "when", "where", "why", "not", "no", "yes",
    "again", "just", "now", "still", "also", "too", "very", "really", "some",
    "any", "all", "one", "two", "get", "got", "gets", "need", "needs", "want",
    "wants", "let", "lets", "please", "thanks", "thank", "okay", "ok", "hey",
})


@dataclass(frozen=True)
class ShiftResult:
    """Outcome of one :meth:`TopicDetector.detect_shift` call (plan P7.T3).

    ``shifted`` — the conversation moved off the previous topic;
    ``topic`` — the classified current topic (model output when a model is
    injected, otherwise the deterministic lexical topic);
    ``confidence`` — confidence in the reported decision, in [0, 1];
    ``latency_ms`` — wall time of the whole detection.
    """

    shifted: bool
    topic: str
    confidence: float
    latency_ms: float


def _content_words(messages: Sequence[str]) -> list[str]:
    """Casefolded non-stopword terms of ``messages``, in message order."""
    words: list[str] = []
    for message in messages:
        for word in _WORD_RE.findall(message.casefold()):
            if word not in _STOPWORDS:
                words.append(word)
    return words


def _lexical_topic(messages: Sequence[str]) -> str:
    """Deterministic lexical topic: top content words of ``messages``.

    Words are casefolded, ranked by (descending frequency, first occurrence)
    and capped at ``_TOPIC_MAX_WORDS`` — a stable, LLM-free stand-in for the
    model classification.
    """
    counts: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    for position, word in enumerate(_content_words(messages), start=1):
        counts[word] = counts.get(word, 0) + 1
        if word not in first_seen:
            first_seen[word] = position
    ranked = sorted(counts, key=lambda word: (-counts[word], first_seen[word]))
    return " ".join(ranked[:_TOPIC_MAX_WORDS])


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine of two sparse vectors (both L2-normalized by the embedder)."""
    if not a or not b:
        return 0.0
    if len(b) < len(a):
        a, b = b, a
    return sum(weight * b.get(feature, 0.0) for feature, weight in a.items())


class TopicDetector:
    """Detect topic shifts for virtual session boundaries (plan P7.T3).

    ``topic_model`` — injected main-model callable (``prompt -> raw
    completion``); ``None`` wires the deterministic lexical default (no LLM).
    ``max_messages`` caps how many recent messages feed the classification.
    ``similarity_fn`` — optional ``(content_profile, previous_topic) ->
    float`` replacement for the trigram cosine comparison (``content_profile``
    is the messages' non-stopword terms joined in order). ``shift_threshold``
    is the similarity bar below which the topic counts as shifted (plan
    P7.T3 design: 0.3).
    """

    def __init__(
        self,
        topic_model: Callable[[str], str] | None = None,
        max_messages: int = 10,
        similarity_fn: Callable[[str, str], float] | None = None,
        shift_threshold: float = DEFAULT_SHIFT_THRESHOLD,
    ) -> None:
        if max_messages < 1:
            raise ValueError(f"max_messages must be >= 1, got {max_messages}")
        if not 0.0 <= shift_threshold <= 1.0:
            raise ValueError(
                f"shift_threshold must be in [0, 1], got {shift_threshold}"
            )
        self._topic_model = topic_model
        self._max_messages = max_messages
        self._similarity_fn = similarity_fn
        self._shift_threshold = shift_threshold

    # ── classification ─────────────────────────────────────────────────────
    def _build_prompt(self, messages: Sequence[str]) -> str:
        """The classification prompt: rules + the recent messages, oldest first."""
        rendered = "\n".join(f"- {message}" for message in messages)
        return (
            f"{TOPIC_SYSTEM_PROMPT}\n\n"
            f"{_MESSAGES_HEADER}\n{rendered}\n\n{_MESSAGES_FOOTER}"
        )

    def _classify(self, messages: Sequence[str]) -> str:
        """Current topic: the injected model's output, else the lexical topic.

        Empty/whitespace model output degrades to the lexical fallback (the
        decision must never depend on a silent model).
        """
        if self._topic_model is not None:
            raw = self._topic_model(self._build_prompt(messages))
            if isinstance(raw, str) and raw.strip():
                return " ".join(raw.split())
            logger.warning("topic detector: empty model output; using lexical fallback")
        return _lexical_topic(messages)

    # ── comparison ─────────────────────────────────────────────────────────
    def _default_similarity(self, candidate: str, previous: str) -> float:
        """Character-trigram TF-IDF cosine (P8.T1 embedder, deterministic)."""
        embedder = TfidfTrigramEmbedder()
        embedder.fit([candidate, previous])
        return _cosine(embedder.embed(candidate), embedder.embed(previous))

    def _similarity(self, candidate: str, previous: str) -> float:
        """Similarity in [0, 1] between the content profile and the previous topic."""
        if self._similarity_fn is not None:
            value = float(self._similarity_fn(candidate, previous))
        else:
            value = self._default_similarity(candidate, previous)
        if value != value:  # NaN guard (mirrors the gate's non-finite handling)
            return 0.0
        return min(1.0, max(0.0, value))

    # ── public API (plan Phase 7) ──────────────────────────────────────────
    def detect_shift(
        self,
        recent_messages: Sequence[str],
        prev_topic: str | None = None,
    ) -> ShiftResult:
        """Classify the current topic and compare the conversation with ``prev_topic``.

        Uses the last ``max_messages`` non-blank messages. The comparison is
        the trigram cosine between the messages' lexical content profile
        (their non-stopword terms, joined) and ``prev_topic`` — the P7.T3
        design's "comparing recent messages against prev_topic" — so the
        decision stays robust even when an injected model labels the same
        subject with lexically different words. With no ``prev_topic`` (or
        no usable content) no shift is possible: ``shifted=False`` at
        confidence 1.0. Otherwise ``shifted`` is ``similarity <
        shift_threshold`` and ``confidence`` expresses the reported decision
        (``1 - similarity`` for a shift — the plan's formula —
        ``similarity`` for a continuation).
        """
        start = time.perf_counter()
        messages = [m for m in recent_messages if isinstance(m, str) and m.strip()]
        messages = messages[-self._max_messages :]
        topic = self._classify(messages)

        profile = " ".join(_content_words(messages))
        if prev_topic is None or not prev_topic.strip() or not profile.strip():
            return ShiftResult(
                shifted=False, topic=topic, confidence=1.0, latency_ms=self._elapsed_ms(start)
            )

        similarity = self._similarity(profile, prev_topic.strip())
        shifted = similarity < self._shift_threshold
        confidence = (1.0 - similarity) if shifted else similarity
        return ShiftResult(
            shifted=shifted,
            topic=topic,
            confidence=min(1.0, max(0.0, confidence)),
            latency_ms=self._elapsed_ms(start),
        )

    @staticmethod
    def _elapsed_ms(start: float) -> float:
        return (time.perf_counter() - start) * 1000.0
