"""Dynamic skill bundle suggestion (plan P13.T6).

Phase 13's learning loop watches how skills are used together: when the same
set of skills keeps firing in the same sessions, the user is offered a bundle
so the workflow becomes one deliberate unit instead of an accidental pattern.

Algorithm (simple, deterministic, documented)
---------------------------------------------
1. Co-occurrence counting: every session in the usage ledger contributes its
   (deduplicated) skill-id set; each unordered pair inside one session adds 1
   to that pair's count.
2. Thresholding: pairs whose count reached ``threshold`` become edges of the
   co-occurrence graph.
3. Grouping: bundles are the connected components of that graph with at
   least two skills (singletons are not bundles).
4. Reporting: a bundle's ``co_occurrences`` is the minimum pairwise count
   over all pairs inside it — the weakest link, i.e. every pair in the
   bundle co-occurred at least that many times.

``propose_bundles`` filters out bundles the user already has
(``known_bundles``), so only genuinely new suggestions reach the prompt.
Output is deterministic: skills tuples are sorted and bundles are returned in
ascending skills order.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

__all__ = ["BundleSuggester", "BundleSuggestion"]


@dataclass(frozen=True)
class BundleSuggestion:
    """A proposed bundle: its skills (sorted) and their co-occurrence strength."""

    skills: tuple[str, ...]
    co_occurrences: int


class BundleSuggester:
    """Pairwise co-occurrence analysis over per-session skill usage (plan P13.T6)."""

    def __init__(self, threshold: int = 3) -> None:
        if isinstance(threshold, bool) or not isinstance(threshold, int):
            raise TypeError(f"threshold must be an int, got {type(threshold).__name__}")
        if threshold < 1:
            raise ValueError(f"threshold must be >= 1, got {threshold}")
        self.threshold: int = threshold

    # ── analysis ──────────────────────────────────────────────────────────

    def analyze(self, usage_ledger: Sequence[Sequence[str]]) -> list[BundleSuggestion]:
        """Bundles over the co-occurrence graph of ``usage_ledger``.

        ``usage_ledger`` holds one entry per session (the skill ids fired in
        it; duplicates within one session count once). Returns the connected
        components (size >= 2) of the thresholded pair-co-occurrence graph,
        sorted ascending by skills tuple. An empty ledger or no qualifying
        pairs yields ``[]``.
        """
        counts = self._pair_counts(usage_ledger)
        adjacency: dict[str, set[str]] = {}
        for (first, second), count in counts.items():
            if count >= self.threshold:
                adjacency.setdefault(first, set()).add(second)
                adjacency.setdefault(second, set()).add(first)
        seen: set[str] = set()
        suggestions: list[BundleSuggestion] = []
        for start in sorted(adjacency):
            if start in seen:
                continue
            component = self._component(start, adjacency)
            seen.update(component)
            if len(component) < 2:
                continue
            members = sorted(component)
            # Every pair inside a component is an edge (count >= threshold),
            # so the weakest pairwise link is the bundle's strength.
            strength = min(
                counts[(first, second)]
                for index, first in enumerate(members)
                for second in members[index + 1 :]
            )
            suggestions.append(
                BundleSuggestion(skills=tuple(members), co_occurrences=strength)
            )
        suggestions.sort(key=lambda suggestion: suggestion.skills)
        return suggestions

    def propose_bundles(
        self,
        usage_ledger: Sequence[Sequence[str]],
        known_bundles: Sequence[Iterable[str]],
    ) -> list[BundleSuggestion]:
        """:meth:`analyze` minus the bundles the user already has.

        ``known_bundles`` holds the skill sets of existing bundles (order of
        elements irrelevant — compared as ``frozenset``).
        """
        known = {frozenset(bundle) for bundle in known_bundles}
        return [
            suggestion
            for suggestion in self.analyze(usage_ledger)
            if frozenset(suggestion.skills) not in known
        ]

    # ── internals ─────────────────────────────────────────────────────────

    def _pair_counts(
        self, usage_ledger: Sequence[Sequence[str]]
    ) -> dict[tuple[str, str], int]:
        """Unordered-pair co-occurrence counts, canonicalized (a < b per key)."""
        counts: dict[tuple[str, str], int] = {}
        for index, session in enumerate(usage_ledger):
            if isinstance(session, str):
                raise TypeError(
                    f"usage_ledger[{index}] must be a sequence of skill ids, not a bare str"
                )
            skills: set[str] = set()
            for skill_id in session:
                if not isinstance(skill_id, str):
                    raise TypeError(
                        f"usage_ledger[{index}] must contain only str, "
                        f"got {type(skill_id).__name__}"
                    )
                skills.add(skill_id)
            ordered = sorted(skills)
            for position, first in enumerate(ordered):
                for second in ordered[position + 1 :]:
                    key = (first, second)
                    counts[key] = counts.get(key, 0) + 1
        return counts

    @staticmethod
    def _component(start: str, adjacency: dict[str, set[str]]) -> set[str]:
        """Connected component containing ``start`` (deterministic BFS order)."""
        component = {start}
        queue = [start]
        while queue:
            current = queue.pop()
            for neighbor in sorted(adjacency[current]):
                if neighbor not in component:
                    component.add(neighbor)
                    queue.append(neighbor)
        return component
