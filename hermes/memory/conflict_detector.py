"""ConflictDetector: pre-write semantic conflict check (plan P8.T1).

Before a candidate memory is written, the existing entries are scanned for a
*potential conflict*: the entry most similar to the new text above the
threshold (plan Section 8: similarity >0.85) is reported as the conflicting
entry, classified as either a ``duplicate`` (same fact restated) or a
``contradiction`` (the fact is negated/reversed).

Embedding backend — IMPORTANT deviation from the plan, documented
------------------------------------------------------------------
The plan (Section 8) specifies ``sentence-transformers`` embeddings.
``sentence-transformers`` is NOT installed locally (it pulls the heavy torch
dependency), so this module defines an :class:`EmbeddingBackend` protocol with
two implementations:

- :class:`TfidfTrigramEmbedder` — the DEFAULT and what the tests use.
  Deterministic, pure-python: character-trigram TF-IDF vectors (sublinear TF,
  smoothed IDF over the candidate corpus, per-word ``^``/``$`` boundary
  sentinels) compared with cosine similarity.
- :class:`SentenceTransformerEmbedder` — optional; lazy-imports
  ``sentence_transformers`` on first use and raises
  :class:`EmbeddingBackendUnavailableError` with a clear message when the
  package is unavailable.

Embeddings are **sparse vectors** (``dict[feature, weight]``): trigram keys for
the TF-IDF backend, dimension-index keys for the dense backend, so one cosine
implementation serves both.

Contradiction classification — the plan assigns it to the main model
(Section 8: "For contradiction detection (vs duplicate): uses main model").
Main-model contradiction classification lands with the reviewer pipeline
(P8.T2 wires it); this module's heuristic is the LOCAL/NO-LLM FALLBACK PATH
per plan 5.5 ("Conflict detection → fallback to embeddings-only, skip
contradiction detection") — realized here as a conservative lexical-cue
heuristic instead of a full skip, so the reviewer still receives the richer
``duplicate``/``contradiction`` hint when no LLM is available.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, Protocol

__all__ = [
    "CONFLICT_THRESHOLD",
    "ANTONYM_PAIRS",
    "NEGATION_CUES",
    "ConflictDetector",
    "ConflictKind",
    "ConflictResult",
    "EmbeddingBackend",
    "EmbeddingBackendUnavailableError",
    "SentenceTransformerEmbedder",
    "TfidfTrigramEmbedder",
]

#: Similarity above which two entries are a potential conflict (plan P8.T1).
CONFLICT_THRESHOLD: float = 0.85

#: Kind of a detected conflict; ``None`` when no conflict was found.
ConflictKind = Literal["duplicate", "contradiction"]

#: Sparse embedding vector: feature key -> weight.
Vector = dict[str, float]


class EmbeddingBackendUnavailableError(RuntimeError):
    """Raised when the configured embedding backend cannot be loaded."""


class EmbeddingBackend(Protocol):
    """Text-embedding backend used by :class:`ConflictDetector`.

    ``fit`` learns corpus statistics (IDF for the TF-IDF backend; a no-op for
    dense neural embeddings). ``embed`` returns the sparse vector of one text;
    weights are L2-normalized by convention, though the detector's cosine
    similarity does not rely on it.
    """

    def fit(self, texts: Iterable[str]) -> None:
        """Learn corpus statistics from ``texts`` (call before ``embed``)."""
        ...

    def embed(self, text: str) -> Vector:
        """Return the embedding vector of ``text`` (after :meth:`fit`)."""
        ...


# ── default backend: char-trigram TF-IDF ────────────────────────────────────

_WORD_RE = re.compile(r"[a-z0-9]+")


def _trigrams(text: str) -> list[str]:
    """Character trigrams of ``text`` (normalized), one word at a time.

    The text is casefolded, punctuation is dropped and every word is embedded
    between ``^``/``$`` sentinels before trigram extraction, so word-boundary
    morphology ("friday" vs "fridays") shares most of its trigrams instead of
    being diluted by cross-word trigrams.
    """
    out: list[str] = []
    for word in _WORD_RE.findall(text.casefold()):
        padded = f"^{word}$"
        if len(padded) == 3:  # single-char word: "^a$" is exactly one trigram
            out.append(padded)
            continue
        out.extend(padded[i : i + 3] for i in range(len(padded) - 2))
    return out


class TfidfTrigramEmbedder:
    """Deterministic pure-python embedder (plan P8.T1 default backend).

    ``fit`` computes smoothed IDF over the corpus trigrams:
    ``idf(t) = ln((N + 1) / (df(t) + 1)) + 1``. ``embed`` builds a sublinear-TF
    (``1 + ln(count)``) TF-IDF vector, L2-normalized. Trigrams never seen in
    the fitted corpus are ignored (they carry no corpus-relative weight); when
    ``embed`` is called before any ``fit``, a single-document fit is performed
    so every observed trigram gets the same IDF and the vector degrades to
    plain TF. Pure python, fully deterministic, no third-party dependencies.
    """

    def __init__(self) -> None:
        self._idf: dict[str, float] | None = None

    def fit(self, texts: Iterable[str]) -> None:
        """Compute IDF over the trigram document frequency of ``texts``."""
        doc_freq: dict[str, int] = {}
        count = 0
        for text in texts:
            count += 1
            for trigram in set(_trigrams(text)):
                doc_freq[trigram] = doc_freq.get(trigram, 0) + 1
        self._idf = {
            trigram: math.log((count + 1) / (df + 1)) + 1.0
            for trigram, df in doc_freq.items()
        }

    def embed(self, text: str) -> Vector:
        """L2-normalized sparse TF-IDF trigram vector of ``text``."""
        if self._idf is None:
            self.fit([text])
        assert self._idf is not None  # set by the branch above (mypy narrowing)
        idf = self._idf
        counts: dict[str, int] = {}
        for trigram in _trigrams(text):
            counts[trigram] = counts.get(trigram, 0) + 1
        vector = {
            trigram: (1.0 + math.log(n)) * idf[trigram]
            for trigram, n in counts.items()
            if trigram in idf
        }
        norm = math.sqrt(sum(weight * weight for weight in vector.values()))
        if norm == 0.0:
            return {}
        return {trigram: weight / norm for trigram, weight in vector.items()}


class SentenceTransformerEmbedder:
    """Optional dense backend lazy-loading ``sentence_transformers`` (plan P8.T1).

    Import and model load happen on first :meth:`embed` call; when the
    ``sentence_transformers`` package is not installed an
    :class:`EmbeddingBackendUnavailableError` with an actionable message is
    raised (never a bare ImportError). Dense model output is projected into a
    sparse vector keyed by dimension index so the shared cosine applies.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model: object | None = None

    def fit(self, texts: Iterable[str]) -> None:
        """No-op: dense sentence embeddings need no corpus fitting."""
        return None

    def _ensure_model(self) -> object:
        if self._model is None:
            try:
                from sentence_transformers import (  # noqa: PLC0415 (lazy import)
                    SentenceTransformer,
                )
            except ImportError as exc:
                raise EmbeddingBackendUnavailableError(
                    "sentence-transformers is not installed; install it with "
                    "'pip install sentence-transformers' or use the default "
                    "TfidfTrigramEmbedder (ConflictDetector(embedder=None))"
                ) from exc
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed(self, text: str) -> Vector:
        """Dense embedding of ``text`` (dimension-index keyed sparse vector)."""
        model = self._ensure_model()
        encode = getattr(model, "encode", None)
        if encode is None:  # pragma: no cover - defensive
            raise EmbeddingBackendUnavailableError(
                "configured sentence-transformers model has no encode() method"
            )
        raw = encode(text)
        return {str(index): float(value) for index, value in enumerate(raw)}


def _cosine(a: Vector, b: Vector) -> float:
    """Cosine similarity of two sparse vectors (0.0 when either is zero)."""
    if not a or not b:
        return 0.0
    dot = sum(weight * b[key] for key, weight in a.items() if key in b)
    norm_a = math.sqrt(sum(weight * weight for weight in a.values()))
    norm_b = math.sqrt(sum(weight * weight for weight in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:  # pragma: no cover - guarded by empty check
        return 0.0
    return dot / (norm_a * norm_b)


# ── contradiction heuristic (plan 5.5 local fallback; P8.T2 owns the LLM path) ─

#: Negation/contrast cues; a cue present on exactly one side of a
#: high-similarity pair signals a contradiction (asymmetric negation).
NEGATION_CUES: tuple[str, ...] = (
    "not",
    "never",
    "no longer",
    "instead",
    "rather than",
    "cannot",
    "without",
)

#: Small bundled antonym table: word pairs whose co-occurrence across a
#: high-similarity pair signals a reversed fact.
ANTONYM_PAIRS: tuple[tuple[str, str], ...] = (
    ("enabled", "disabled"),
    ("enable", "disable"),
    ("true", "false"),
    ("add", "remove"),
    ("start", "stop"),
    ("allow", "deny"),
    ("allow", "forbid"),
    ("increase", "decrease"),
    ("open", "close"),
    ("show", "hide"),
    ("always", "never"),
    ("on", "off"),
    ("yes", "no"),
    ("require", "optional"),
    ("pnpm", "npm"),
    ("yarn", "npm"),
    ("global", "local"),
    ("include", "exclude"),
    ("max", "min"),
    ("first", "last"),
    ("before", "after"),
)


def _tokens(text: str) -> set[str]:
    """Word tokens of ``text`` (casefolded, punctuation stripped)."""
    return set(_WORD_RE.findall(text.casefold()))


def _phrases(text: str) -> set[str]:
    """Multi-word cue phrases present in the normalized text."""
    normalized = " " + " ".join(_WORD_RE.findall(text.casefold())) + " "
    return {cue for cue in NEGATION_CUES if " " in cue and f" {cue} " in normalized}


def _is_contradiction(new_text: str, existing_text: str) -> bool:
    """Heuristic contradiction check between two high-similarity texts.

    Fires when (a) a negation cue appears on exactly one side, or (b) one
    antonym-pair member appears on each side. Purely lexical — advisory only;
    the reviewer pipeline (P8.T2) owns main-model classification.
    """
    new_tokens = _tokens(new_text)
    existing_tokens = _tokens(existing_text)
    new_phrases = _phrases(new_text)
    existing_phrases = _phrases(existing_text)

    for cue in NEGATION_CUES:
        if " " in cue:
            if (cue in new_phrases) != (cue in existing_phrases):
                return True
        elif (cue in new_tokens) != (cue in existing_tokens):
            return True
    for left, right in ANTONYM_PAIRS:
        if (left in new_tokens and right in existing_tokens) or (
            right in new_tokens and left in existing_tokens
        ):
            return True
    return False


@dataclass(frozen=True)
class ConflictResult:
    """Outcome of one :meth:`ConflictDetector.check_conflict` call (plan P8.T1).

    ``conflicting_entry_id`` is the id of the most similar existing entry and
    ``kind`` its classification; both are ``None`` when no existing entry
    reaches the similarity threshold. ``similarity`` always carries the best
    cosine similarity seen (0.0 when ``existing`` is empty).
    """

    is_conflict: bool
    similarity: float
    conflicting_entry_id: str | None
    kind: ConflictKind | None = None


class ConflictDetector:
    """Pre-write conflict check over existing memory entries (plan P8.T1).

    ``embedder`` defaults to :class:`TfidfTrigramEmbedder` (deterministic,
    dependency-free — see module docstring for the deviation from the plan's
    sentence-transformers); ``threshold`` defaults to 0.85 (plan Section 8).
    """

    def __init__(
        self,
        embedder: EmbeddingBackend | None = None,
        threshold: float = CONFLICT_THRESHOLD,
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"threshold must lie in (0, 1], got {threshold!r}")
        self._embedder = embedder if embedder is not None else TfidfTrigramEmbedder()
        self.threshold = threshold

    def check_conflict(
        self,
        new_entry: str,
        existing: list[tuple[str, str]],
    ) -> ConflictResult:
        """Find the existing entry most similar to ``new_entry`` (plan P8.T1).

        ``existing`` carries ``(entry_id, text)`` pairs. The embedder is fitted
        on the candidate corpus (new text + all existing texts) so IDF weights
        reflect this comparison; the best cosine similarity above
        ``self.threshold`` yields a conflict, classified ``contradiction`` when
        the heuristic fires on the winning pair, else ``duplicate``.
        """
        if not existing:
            return ConflictResult(
                is_conflict=False, similarity=0.0, conflicting_entry_id=None
            )
        self._embedder.fit([new_entry, *(text for _id, text in existing)])
        new_vector = self._embedder.embed(new_entry)
        best_id: str | None = None
        best_text: str | None = None
        best_similarity = 0.0
        for entry_id, text in existing:
            similarity = _cosine(new_vector, self._embedder.embed(text))
            if similarity > best_similarity:
                best_similarity = similarity
                best_id = entry_id
                best_text = text
        if best_id is None or best_text is None or best_similarity <= self.threshold:
            return ConflictResult(
                is_conflict=False,
                similarity=best_similarity,
                conflicting_entry_id=None,
            )
        kind: ConflictKind = (
            "contradiction" if _is_contradiction(new_entry, best_text) else "duplicate"
        )
        return ConflictResult(
            is_conflict=True,
            similarity=best_similarity,
            conflicting_entry_id=best_id,
            kind=kind,
        )
