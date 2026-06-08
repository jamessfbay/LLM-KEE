from llm_kee.models import FailureRecord, ImprovementAction, ImprovementProposal, ImprovementReview
from llm_kee.models.base import now_utc
from llm_kee.storage import KEEStore


class ImprovementService:
    def __init__(self, store: KEEStore) -> None:
        self.store = store

    def propose_for_failure(self, failure_id: str) -> ImprovementProposal:
        failure = self.store.failure_records.get(failure_id)
        if not failure:
            raise ValueError(f"Failure not found: {failure_id}")
        improvement_type = self._improvement_type(failure)
        action = self._action_for_failure(failure, improvement_type)
        self.store.improvement_actions.upsert(action)
        proposal = ImprovementProposal(
            improvement_type=improvement_type,
            title=f"{improvement_type}: {failure.summary}",
            description=self._description(failure, improvement_type),
            failure_ids=[failure.id],
            intent_ids=[failure.intent_id] if failure.intent_id else [],
            actions=[action],
            status="pending_review",
            rationale=f"Generated from failure type {failure.failure_type}.",
        )
        return self.store.improvement_proposals.upsert(proposal)

    def review(self, proposal_id: str, approve: bool, notes: str | None = None) -> ImprovementProposal:
        proposal = self.store.improvement_proposals.get(proposal_id)
        if not proposal:
            raise ValueError(f"Improvement proposal not found: {proposal_id}")
        proposal.status = "approved" if approve else "rejected"
        proposal.updated_at = now_utc()
        proposal = self.store.improvement_proposals.upsert(proposal)
        review = ImprovementReview(
            improvement_id=proposal.id,
            decision=proposal.status,
            notes=notes,
        )
        self.store.improvement_reviews.upsert(review)
        return proposal

    def _improvement_type(self, failure: FailureRecord) -> str:
        mapping = {
            "missing_source": "add_source_ingestion",
            "low_evidence": "request_human_review",
            "conflicting_answer": "add_evaluation_case",
            "wrong_skill_route": "adjust_skill_routing",
            "workflow_step_error": "revise_prompt",
            "judge_disagreement": "add_evaluation_case",
        }
        return mapping.get(failure.failure_type, "request_human_review")

    def _action_for_failure(self, failure: FailureRecord, improvement_type: str) -> ImprovementAction:
        descriptions = {
            "add_source_ingestion": "Add or configure source ingestion for the missing data.",
            "add_skill": "Add a new skill definition for the unsupported task.",
            "adjust_skill_routing": "Adjust task-to-skill routing for this intent.",
            "revise_prompt": "Revise prompt or workflow step instructions for the failing step.",
            "add_evaluation_case": "Add an evaluation case that captures this failure.",
            "request_human_review": "Route the issue to human review before changing the system.",
        }
        return ImprovementAction(
            action_type=improvement_type,
            target_type=failure.target_type,
            target_id=failure.target_id,
            description=descriptions.get(improvement_type, "Review and improve the agent behavior."),
            payload={
                "failure_id": failure.id,
                "failure_type": failure.failure_type,
                "artifact_id": failure.artifact_id,
                "proposal_id": failure.proposal_id,
                "workflow_run_id": failure.workflow_run_id,
                "action_run_id": failure.action_run_id,
            },
        )

    def _description(self, failure: FailureRecord, improvement_type: str) -> str:
        return (
            f"Proposed {improvement_type} because LLM-KEE recorded "
            f"{failure.failure_type}: {failure.summary}"
        )
