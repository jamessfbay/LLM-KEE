from typing import Any

from llm_kee.models import FailureCluster, FailureRecord
from llm_kee.storage import KEEStore


class FailureDetector:
    def __init__(self, store: KEEStore) -> None:
        self.store = store

    def record(self, payload: dict[str, Any]) -> FailureRecord:
        failure_type = payload.get("failure_type") or self._infer_failure_type(payload)
        failure = FailureRecord(
            failure_type=failure_type,
            summary=payload.get("summary") or self._summary(failure_type, payload),
            severity=payload.get("severity") or self._severity(failure_type),
            intent_id=payload.get("intent_id"),
            target_type=payload.get("target_type"),
            target_id=payload.get("target_id"),
            artifact_id=payload.get("artifact_id"),
            proposal_id=payload.get("proposal_id"),
            workflow_run_id=payload.get("workflow_run_id"),
            action_run_id=payload.get("action_run_id"),
            evidence_ids=list(payload.get("evidence_ids") or []),
            payload=payload,
            status=payload.get("status") or "open",
        )
        failure = self.store.failure_records.upsert(failure)
        self._upsert_cluster(failure)
        return failure

    def _infer_failure_type(self, payload: dict[str, Any]) -> str:
        if payload.get("workflow_step_error") or payload.get("error"):
            return "workflow_step_error"
        if payload.get("judge_disagreement"):
            return "judge_disagreement"
        if payload.get("wrong_skill_route"):
            return "wrong_skill_route"
        if payload.get("conflict") or payload.get("decision") == "conflict":
            return "conflicting_answer"
        if payload.get("missing_source") or payload.get("source_missing"):
            return "missing_source"
        if payload.get("artifact_id") and not payload.get("evidence_ids"):
            return "low_evidence"
        return "low_evidence"

    def _summary(self, failure_type: str, payload: dict[str, Any]) -> str:
        target = payload.get("target_id") or payload.get("artifact_id") or payload.get("proposal_id") or "unknown"
        return f"{failure_type} detected for {target}"

    def _severity(self, failure_type: str) -> str:
        if failure_type in {"conflicting_answer", "judge_disagreement", "workflow_step_error"}:
            return "high"
        if failure_type in {"missing_source", "wrong_skill_route"}:
            return "medium"
        return "low"

    def _upsert_cluster(self, failure: FailureRecord) -> None:
        clusters = [
            cluster
            for cluster in self.store.failure_clusters.list()
            if cluster.failure_type == failure.failure_type and cluster.status == "open"
        ]
        if clusters:
            cluster = clusters[0]
            if failure.id not in cluster.failure_ids:
                cluster.failure_ids.append(failure.id)
        else:
            cluster = FailureCluster(
                failure_type=failure.failure_type,
                failure_ids=[failure.id],
                summary=f"Open cluster for {failure.failure_type}",
                suggested_improvement_type=self.suggested_improvement_type(failure.failure_type),
            )
        self.store.failure_clusters.upsert(cluster)

    def suggested_improvement_type(self, failure_type: str) -> str:
        mapping = {
            "missing_source": "add_source_ingestion",
            "low_evidence": "request_human_review",
            "conflicting_answer": "add_evaluation_case",
            "wrong_skill_route": "adjust_skill_routing",
            "workflow_step_error": "revise_prompt",
            "judge_disagreement": "add_evaluation_case",
        }
        return mapping.get(failure_type, "request_human_review")
