from __future__ import annotations

from llm_kee.experience.hashing import canonical_hash, seal_model
from llm_kee.experience.models import (
    ActorHandoffMode,
    ActorLearningHandoff,
    CausalStatus,
    ExperienceCandidate,
    ExperienceEvaluation,
    ExperienceEvaluationDataset,
    ExperienceEvaluationDecision,
    MemoryRevisionOperation,
    MemoryRevisionProposal,
)


EVALUATOR_BUNDLE = {
    "boundary": "kee-experience-evaluator/1",
    "causal_claim_rule": "mechanism-needs-independent-support/1",
    "negative_transfer_limit": 0.01,
    "scope_rule": "tenant-domain-exact/1",
}
EVALUATOR_BUNDLE_HASH = canonical_hash(EVALUATOR_BUNDLE)


class ExperienceEvaluator:
    """Evaluate memory content independently from the actor that authored it."""

    def evaluate(
        self,
        revision: MemoryRevisionProposal,
        candidates: list[ExperienceCandidate],
        dataset: ExperienceEvaluationDataset,
        source_handoffs: list[ActorLearningHandoff],
    ) -> ExperienceEvaluation:
        relevant = [candidate for candidate in candidates if candidate.id in set(revision.candidate_ids)]
        if revision.operation != MemoryRevisionOperation.NO_CHANGE and len(relevant) != len(revision.candidate_ids):
            raise ValueError("evaluation is missing a revision candidate")
        if any(
            candidate.tenant_id != revision.tenant_id or candidate.domain != revision.domain
            for candidate in relevant
        ):
            raise ValueError("revision candidates must remain in the same tenant and domain")
        handoffs = {handoff.id: handoff for handoff in source_handoffs}
        if any(candidate.handoff_id not in handoffs for candidate in relevant):
            raise ValueError("evaluation is missing the experience-owning actor handoff")
        if any(
            handoff.tenant_id != revision.tenant_id or handoff.domain != revision.domain
            for handoff in handoffs.values()
        ):
            raise ValueError("actor handoffs must remain in the revision scope")

        cross_tenant = sum(
            case.tenant_id != revision.tenant_id or case.domain != revision.domain
            for case in dataset.cases
        )
        in_scope = [
            case
            for case in dataset.cases
            if case.tenant_id == revision.tenant_id and case.domain == revision.domain
        ]
        direct_support = sum(case.supports_candidate for case in in_scope)
        counterexamples = sum(case.is_counterexample for case in in_scope)
        in_scope_count = len(in_scope)
        candidate_success = sum(case.candidate_succeeded for case in in_scope) / max(1, in_scope_count)
        parent_success = sum(case.parent_succeeded for case in in_scope) / max(1, in_scope_count)
        negative_transfer = sum(case.negative_transfer for case in in_scope) / max(1, in_scope_count)
        has_mechanism_support = any(case.causal_mechanism_verified for case in in_scope)
        unsupported_causal = sum(
            candidate.causal_status != CausalStatus.OBSERVED_ASSOCIATION and not has_mechanism_support
            for candidate in relevant
        )

        concerns: list[str] = []
        if direct_support == 0:
            concerns.append("No independent case directly supports the candidate experience.")
        if unsupported_causal:
            concerns.append("A causal claim lacks independent mechanism or intervention support.")
        if cross_tenant:
            concerns.append("Evaluation dataset contains records outside the candidate scope.")
        if candidate_success < parent_success:
            concerns.append("Candidate task success is below the frozen parent memory.")
        if negative_transfer > 0.01:
            concerns.append("Negative transfer exceeds the one-percent safety bound.")
        if counterexamples:
            concerns.append("Counterexamples require an explicit boundary or qualification.")

        unsafe = bool(cross_tenant) or candidate_success < parent_success or negative_transfer > 0.01
        incomplete = direct_support == 0 or bool(unsupported_causal) or bool(counterexamples)
        if unsafe:
            decision = ExperienceEvaluationDecision.REJECTED
        elif incomplete:
            decision = ExperienceEvaluationDecision.NEEDS_EXPERIMENT
        else:
            decision = ExperienceEvaluationDecision.ACCEPTED
        auto_eligible = (
            revision.operation != MemoryRevisionOperation.NO_CHANGE
            and decision == ExperienceEvaluationDecision.ACCEPTED
            and bool(relevant)
            and all(
                handoffs[candidate.handoff_id].promotion_eligible
                and handoffs[candidate.handoff_id].handoff_mode == ActorHandoffMode.SAME_SESSION
                for candidate in relevant
            )
        )

        evaluated_at = max(handoff.created_at for handoff in source_handoffs)
        identity = canonical_hash(
            {
                "revision_hash": revision.revision_hash,
                "dataset_hash": dataset.dataset_hash,
                "evaluator_bundle_hash": EVALUATOR_BUNDLE_HASH,
            }
        )
        return seal_model(
            ExperienceEvaluation,
            "evaluation_hash",
            id=f"experience_evaluation_{identity[:24]}",
            revision_id=revision.id,
            tenant_id=revision.tenant_id,
            domain=revision.domain,
            evaluator_bundle_hash=EVALUATOR_BUNDLE_HASH,
            dataset_hash=dataset.dataset_hash,
            direct_support_count=direct_support,
            counterexample_count=counterexamples,
            regression_case_count=len(dataset.cases),
            candidate_task_success_rate=candidate_success,
            parent_task_success_rate=parent_success,
            negative_transfer_rate=negative_transfer,
            unsupported_causal_claims=unsupported_causal,
            cross_tenant_violations=cross_tenant,
            decision=decision,
            concerns=concerns,
            automatic_release_eligible=auto_eligible,
            evaluated_at=evaluated_at,
        )
