from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from llm_kee.experience.hashing import canonical_hash, seal_model
from llm_kee.experience.models import (
    ExperienceCandidate,
    ExperienceCandidateStatus,
    ExperienceKind,
    ImmutableMemorySnapshot,
    MemoryRevisionOperation,
    MemoryRevisionProposal,
)


@dataclass(frozen=True)
class ReconciliationResult:
    revisions: tuple[MemoryRevisionProposal, ...]
    snapshot: ImmutableMemorySnapshot


class MemoryReconciler:
    """Coordinate new experience against the whole immutable memory snapshot."""

    def reconcile_batch(
        self,
        candidates: Iterable[ExperienceCandidate],
        snapshot: ImmutableMemorySnapshot,
    ) -> ReconciliationResult:
        pending = tuple(candidates)
        if any(
            candidate.tenant_id != snapshot.tenant_id
            or candidate.domain != snapshot.domain
            or candidate.parent_memory_hash != snapshot.snapshot_hash
            for candidate in pending
        ):
            raise ValueError("broad batch must share the same immutable memory snapshot and scope")

        current = snapshot
        revisions: list[MemoryRevisionProposal] = []
        for candidate in pending:
            revision, current = self._reconcile_one(candidate, current)
            revisions.append(revision)
        return ReconciliationResult(revisions=tuple(revisions), snapshot=current)

    def _reconcile_one(
        self,
        candidate: ExperienceCandidate,
        current: ImmutableMemorySnapshot,
    ) -> tuple[MemoryRevisionProposal, ImmutableMemorySnapshot]:
        duplicate = next(
            (
                existing
                for existing in current.experiences
                if self._signature(existing) == self._signature(candidate)
            ),
            None,
        )
        conflicts = [
            existing
            for existing in current.experiences
            if self._normalized(existing.condition) == self._normalized(candidate.condition)
            and self._normalized(existing.action) == self._normalized(candidate.action)
            and self._normalized(existing.expected_outcome)
            != self._normalized(candidate.expected_outcome)
        ]

        if duplicate:
            operation = MemoryRevisionOperation.NO_CHANGE
            target = None
            candidate_ids: list[str] = []
            updated = current.experiences
            rationale = "Equivalent conditional experience already exists in the current snapshot."
        elif conflicts:
            operation = MemoryRevisionOperation.QUALIFY
            target = conflicts[0].id
            candidate_ids = [candidate.id]
            updated = current.experiences + (self._with_status(candidate),)
            rationale = "Conflicting outcome retained as a bounded qualification pending independent evaluation."
        else:
            operation = MemoryRevisionOperation.ADD
            target = None
            candidate_ids = [candidate.id]
            updated = current.experiences + (self._with_status(candidate),)
            rationale = "New scoped conditional experience has no duplicate or direct contradiction."

        next_snapshot = ImmutableMemorySnapshot.create(
            current.tenant_id,
            current.domain,
            current.memory_binding,
            updated,
        )
        revision_identity = canonical_hash(
            {
                "candidate_hash": candidate.candidate_hash,
                "parent_memory_hash": current.snapshot_hash,
                "proposed_snapshot_hash": next_snapshot.snapshot_hash,
                "operation": str(operation),
            }
        )
        revision = seal_model(
            MemoryRevisionProposal,
            "revision_hash",
            id=f"memory_revision_{revision_identity[:24]}",
            tenant_id=current.tenant_id,
            domain=current.domain,
            memory_binding=current.memory_binding,
            parent_memory_hash=current.snapshot_hash,
            candidate_ids=candidate_ids,
            operation=operation,
            target_experience_id=target,
            conflicts_with=[item.id for item in conflicts],
            rationale=rationale,
            proposed_snapshot_hash=next_snapshot.snapshot_hash,
        )
        return revision, next_snapshot

    @staticmethod
    def _normalized(value: str) -> str:
        return " ".join(value.lower().split())

    def _signature(self, value: ExperienceCandidate) -> tuple[str, str, str, str]:
        return (
            self._normalized(value.condition),
            self._normalized(value.action),
            self._normalized(value.expected_outcome),
            str(value.kind),
        )

    @staticmethod
    def _with_status(candidate: ExperienceCandidate) -> ExperienceCandidate:
        payload = candidate.model_dump(mode="json", exclude={"candidate_hash"})
        payload["status"] = ExperienceCandidateStatus.RECONCILED
        return seal_model(ExperienceCandidate, "candidate_hash", **payload)
