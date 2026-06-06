from llm_kee.models import (
    ActionArtifact,
    FeedbackType,
    LearningSignal,
    ProposalType,
    TargetType,
    UpdateProposal,
    WorkflowRun,
)


class ProposalGenerator:
    def from_signal(self, signal: LearningSignal) -> UpdateProposal:
        payload = signal.payload
        feedback_type = payload.get("feedback_type")
        target_type = payload.get("target_type", TargetType.CLAIM)
        target_id = payload.get("target_id")
        new_value = payload.get("new_value") or {}

        proposal_type = self._proposal_type(feedback_type, target_type)
        evidence_ids = new_value.get("evidence_ids") or payload.get("evidence_ids") or []
        title = f"{proposal_type} for {target_type}"

        return UpdateProposal(
            proposal_type=proposal_type,
            target_type=target_type,
            target_id=target_id,
            title=title,
            rationale=signal.summary,
            evidence_ids=evidence_ids,
            source_signal_ids=[signal.id],
            proposed_change={
                "old_value": payload.get("old_value"),
                "new_value": new_value,
                "comment": payload.get("comment"),
            },
            confidence=0.65 if evidence_ids else 0.45,
        )

    def from_action_artifact(self, artifact: ActionArtifact) -> UpdateProposal:
        return UpdateProposal(
            proposal_type=ProposalType.UPDATE_CLAIM,
            target_type=TargetType.CLAIM,
            target_id=artifact.id,
            title=f"Review action artifact: {artifact.title}",
            rationale=f"Action artifact {artifact.id} produced update-ready intelligence.",
            evidence_ids=artifact.evidence_ids,
            proposed_change={
                "artifact_type": artifact.artifact_type,
                "content": artifact.content,
            },
            confidence=artifact.confidence,
        )

    def from_workflow_run(self, run: WorkflowRun) -> UpdateProposal:
        evidence_ids = [
            evidence_id
            for step in run.steps
            for evidence_id in ((step.output or {}).get("evidence_ids") or [])
        ]
        return UpdateProposal(
            proposal_type=ProposalType.PATTERN_PROPOSAL,
            target_type=TargetType.PATTERN,
            target_id=run.id,
            title=f"Review workflow pattern: {run.task_type}",
            rationale=f"Workflow run {run.id} completed and can be reviewed as a reusable pattern.",
            evidence_ids=evidence_ids,
            proposed_change={
                "workflow_run_id": run.id,
                "task_type": run.task_type,
                "steps": [step.model_dump(mode="json") for step in run.steps],
                "output": run.output,
            },
            confidence=0.7 if run.status == "completed" else 0.4,
        )

    def _proposal_type(self, feedback_type: str | None, target_type: str) -> ProposalType:
        if feedback_type == FeedbackType.MISSING_EVIDENCE:
            return ProposalType.ADD_EVIDENCE
        if feedback_type == FeedbackType.DUPLICATE and target_type == TargetType.ENTITY:
            return ProposalType.MERGE_ENTITY
        if target_type == TargetType.RELATION:
            return ProposalType.UPDATE_RELATION
        if target_type == TargetType.SCHEMA:
            return ProposalType.SCHEMA_CHANGE
        if feedback_type == FeedbackType.REJECTION:
            return ProposalType.RETIRE_CLAIM
        return ProposalType.UPDATE_CLAIM
