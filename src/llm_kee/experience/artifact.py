from __future__ import annotations

from llm_kee.experience.hashing import canonical_hash
from llm_kee.experience.models import (
    ExperienceEvaluation,
    ExperienceEvaluationDecision,
    FrozenMemoryArtifact,
    ImmutableMemorySnapshot,
    MemoryArtifactStatus,
)


DISTILLER_PROMPT_HASH = canonical_hash("kee-experience-distiller/1")
RECONCILER_PROMPT_HASH = canonical_hash("kee-memory-reconciler/1")


class MemoryArtifactFreezer:
    """Freeze a candidate artifact; activation remains exclusively in NOX."""

    def freeze_candidate(
        self,
        *,
        snapshot: ImmutableMemorySnapshot,
        parent_memory_hash: str,
        evaluations: list[ExperienceEvaluation],
        evaluator_bundle_hash: str,
        schema_hash: str,
        retrieval_version: str,
        parent_artifact_id: str | None = None,
    ) -> FrozenMemoryArtifact:
        if not snapshot.experiences or not evaluations:
            raise ValueError("freezing memory requires experience and independent evaluation")
        if any(
            item.tenant_id != snapshot.tenant_id
            or item.domain != snapshot.domain
            or item.evaluator_bundle_hash != evaluator_bundle_hash
            for item in evaluations
        ):
            raise ValueError("evaluation scope or evaluator bundle does not match memory")
        accepted = all(
            item.decision == ExperienceEvaluationDecision.ACCEPTED for item in evaluations
        )
        auto_eligible = accepted and all(item.automatic_release_eligible for item in evaluations)
        identity = canonical_hash(
            {
                "snapshot_hash": snapshot.snapshot_hash,
                "parent_memory_hash": parent_memory_hash,
                "evaluation_hashes": sorted(item.evaluation_hash for item in evaluations),
                "schema_hash": schema_hash,
                "retrieval_version": retrieval_version,
            }
        )
        artifact_id = f"memory_artifact_{identity[:24]}"
        return FrozenMemoryArtifact(
            id=artifact_id,
            tenant_id=snapshot.tenant_id,
            domain=snapshot.domain,
            memory_binding=snapshot.memory_binding,
            parent_artifact_id=parent_artifact_id,
            parent_memory_hash=parent_memory_hash,
            artifact_uri=f"nox-memory://candidate/{artifact_id}",
            artifact_hash=snapshot.snapshot_hash,
            schema_hash=schema_hash,
            experience_ids=[item.id for item in snapshot.experiences],
            experiences=list(snapshot.experiences),
            distiller_prompt_hash=DISTILLER_PROMPT_HASH,
            reconciler_prompt_hash=RECONCILER_PROMPT_HASH,
            evaluator_bundle_hash=evaluator_bundle_hash,
            retrieval_version=retrieval_version,
            status=MemoryArtifactStatus.CANDIDATE,
            automatic_release_eligible=auto_eligible,
            created_at=max(item.evaluated_at for item in evaluations),
        )
