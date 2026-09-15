from __future__ import annotations

from dataclasses import dataclass

from llm_kee.experience.hashing import canonical_hash, seal_model
from llm_kee.experience.curriculum import CurriculumPlanner, PracticeCoordinator
from llm_kee.experience.distiller import ExperienceDistiller
from llm_kee.experience.evaluator import ExperienceEvaluator
from llm_kee.experience.models import (
    ActorLearningHandoff,
    ExperienceCandidate,
    ExperienceLearningBatch,
    ImmutableMemorySnapshot,
)
from llm_kee.experience.reconciler import MemoryReconciler, ReconciliationResult


@dataclass(frozen=True)
class BroadLearningResult:
    handoffs: tuple[ActorLearningHandoff, ...]
    candidates: tuple[ExperienceCandidate, ...]
    reconciliation: ReconciliationResult


class GovernedExperienceLearning:
    """KEE v0.2's proposal-only learning boundary.

    This façade deliberately exposes no publish, permission, policy or external
    action method. NOX owns execution and the release pipeline.
    """

    def __init__(self) -> None:
        self.distiller = ExperienceDistiller()
        self.reconciler = MemoryReconciler()
        self.evaluator = ExperienceEvaluator()
        self.curriculum = CurriculumPlanner()
        self.practice = PracticeCoordinator()

    def process_broad_batch(
        self,
        handoffs: list[ActorLearningHandoff],
        snapshot: ImmutableMemorySnapshot,
    ) -> BroadLearningResult:
        if not handoffs:
            raise ValueError("broad learning batch cannot be empty")
        if any(
            handoff.tenant_id != snapshot.tenant_id
            or handoff.domain != snapshot.domain
            or handoff.parent_memory_hash != snapshot.snapshot_hash
            for handoff in handoffs
        ):
            raise ValueError("all broad branches must begin from the same immutable memory snapshot")

        # Distillation is branch-local against the same frozen snapshot. Only
        # after all branches produce valid handoffs do revisions merge in a
        # deterministic, sequential order.
        ordered_handoffs = tuple(sorted(handoffs, key=lambda item: item.id))
        ordered = tuple(
            candidate
            for handoff in ordered_handoffs
            for candidate in self.distiller.distill(handoff)
        )
        result = self.reconciler.reconcile_batch(ordered, snapshot)
        return BroadLearningResult(
            handoffs=ordered_handoffs,
            candidates=ordered,
            reconciliation=result,
        )

    def build_atomic_batch(
        self,
        handoff: ActorLearningHandoff,
        snapshot: ImmutableMemorySnapshot,
    ) -> ExperienceLearningBatch:
        """Build the retry-safe unit submitted to the NOX State Engine."""
        result = self.process_broad_batch([handoff], snapshot)
        identity = canonical_hash(
            {
                "handoff_hash": handoff.handoff_hash,
                "memory_binding": snapshot.memory_binding,
                "parent_memory_hash": snapshot.snapshot_hash,
            }
        )
        return seal_model(
            ExperienceLearningBatch,
            "batch_hash",
            id=f"experience_batch_{identity[:24]}",
            handoff_id=handoff.id,
            tenant_id=handoff.tenant_id,
            domain=handoff.domain,
            memory_binding=snapshot.memory_binding,
            candidates=list(result.candidates),
            revisions=list(result.reconciliation.revisions),
            final_snapshot_hash=result.reconciliation.snapshot.snapshot_hash,
        )
