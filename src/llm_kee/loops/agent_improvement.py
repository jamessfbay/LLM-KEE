from typing import Any

from llm_kee.failures import FailureDetector
from llm_kee.improvements import ImprovementService
from llm_kee.intent import IntentDetector
from llm_kee.models import AgentImprovementCycle
from llm_kee.storage import KEEStore


class AgentImprovementLoop:
    def __init__(
        self,
        store: KEEStore,
        intent_detector: IntentDetector,
        failure_detector: FailureDetector,
        improvement_service: ImprovementService,
    ) -> None:
        self.store = store
        self.intent_detector = intent_detector
        self.failure_detector = failure_detector
        self.improvement_service = improvement_service

    def run(self, payload: dict[str, Any]) -> AgentImprovementCycle:
        intent = self.intent_detector.detect_conversation(payload)
        failure_payloads = payload.get("failures")
        if not failure_payloads:
            failure_payloads = [payload]
        failures = []
        improvements = []
        for failure_payload in failure_payloads:
            record_payload = dict(failure_payload)
            record_payload.setdefault("intent_id", intent.id)
            failure = self.failure_detector.record(record_payload)
            failures.append(failure)
            improvements.append(self.improvement_service.propose_for_failure(failure.id))
        cycle = AgentImprovementCycle(
            conversation_intent_id=intent.id,
            failure_ids=[failure.id for failure in failures],
            improvement_proposal_ids=[proposal.id for proposal in improvements],
            status="pending_review",
        )
        return self.store.agent_improvement_cycles.upsert(cycle)
