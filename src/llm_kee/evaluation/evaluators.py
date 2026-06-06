from llm_kee.models import (
    EvaluationDecision,
    EvaluationResult,
    EvaluatorType,
    ProposalType,
    UpdateProposal,
)
from llm_kee.config import JudgeConfig


class RuleEngine:
    evaluator_type = EvaluatorType.RULE_ENGINE
    evaluator_name = "rule_engine"

    def evaluate(self, proposal: UpdateProposal) -> EvaluationResult:
        concerns: list[str] = []
        if not proposal.rationale.strip():
            concerns.append("Proposal rationale is required.")
        if proposal.confidence < 0.2:
            concerns.append("Proposal confidence is too low.")
        decision = EvaluationDecision.FAIL if concerns else EvaluationDecision.PASS
        return EvaluationResult(
            proposal_id=proposal.id,
            evaluator_type=self.evaluator_type,
            evaluator_name=self.evaluator_name,
            score=0.2 if concerns else 0.9,
            decision=decision,
            concerns=concerns,
        )


class EvidenceChecker:
    evaluator_type = EvaluatorType.EVIDENCE_CHECKER
    evaluator_name = "evidence_checker"

    def evaluate(self, proposal: UpdateProposal) -> EvaluationResult:
        high_risk = proposal.proposal_type in {
            ProposalType.CREATE_CLAIM,
            ProposalType.UPDATE_CLAIM,
            ProposalType.UPDATE_RELATION,
            ProposalType.SCHEMA_CHANGE,
        }
        if high_risk and not proposal.evidence_ids:
            return EvaluationResult(
                proposal_id=proposal.id,
                evaluator_type=self.evaluator_type,
                evaluator_name=self.evaluator_name,
                score=0.25,
                decision=EvaluationDecision.NEED_MORE_EVIDENCE,
                concerns=["Evidence-backed proposals require at least one evidence ID."],
                recommended_changes=["Attach source evidence before approval."],
            )
        return EvaluationResult(
            proposal_id=proposal.id,
            evaluator_type=self.evaluator_type,
            evaluator_name=self.evaluator_name,
            score=0.85,
            decision=EvaluationDecision.PASS,
        )


class ConflictChecker:
    evaluator_type = EvaluatorType.CONFLICT_CHECKER
    evaluator_name = "conflict_checker"

    def evaluate(self, proposal: UpdateProposal) -> EvaluationResult:
        text = " ".join(
            str(value).lower() for value in proposal.proposed_change.values() if value is not None
        )
        if "conflict" in text or "contradict" in text:
            return EvaluationResult(
                proposal_id=proposal.id,
                evaluator_type=self.evaluator_type,
                evaluator_name=self.evaluator_name,
                score=0.35,
                decision=EvaluationDecision.CONFLICT,
                concerns=["Potential contradiction detected from proposal text."],
            )
        return EvaluationResult(
            proposal_id=proposal.id,
            evaluator_type=self.evaluator_type,
            evaluator_name=self.evaluator_name,
            score=0.8,
            decision=EvaluationDecision.PASS,
        )


class BehaviorSignalEvaluator:
    evaluator_type = EvaluatorType.BEHAVIOR_SIGNAL
    evaluator_name = "behavior_signal"

    def evaluate(self, proposal: UpdateProposal) -> EvaluationResult:
        score = min(1.0, max(0.0, proposal.confidence + 0.1))
        return EvaluationResult(
            proposal_id=proposal.id,
            evaluator_type=self.evaluator_type,
            evaluator_name=self.evaluator_name,
            score=score,
            decision=EvaluationDecision.PASS if score >= 0.5 else EvaluationDecision.REVIEW,
        )


class DeterministicLLMJudge:
    evaluator_type = EvaluatorType.LLM_JUDGE

    def __init__(self, config: JudgeConfig | None = None) -> None:
        self.config = config or JudgeConfig(name="deterministic_mock")
        self.evaluator_name = self.config.name

    def evaluate(self, proposal: UpdateProposal) -> EvaluationResult:
        has_change = bool(proposal.proposed_change.get("new_value"))
        has_reason = len(proposal.rationale) >= 12
        base_score = 0.8 if has_change and has_reason else 0.45
        score = self._provider_adjusted_score(base_score)
        concerns = []
        if not has_change:
            concerns.append("No proposed new value was supplied.")
        if not has_reason:
            concerns.append("Rationale is too short to judge semantic quality.")
        return EvaluationResult(
            proposal_id=proposal.id,
            evaluator_type=self.evaluator_type,
            evaluator_name=self.evaluator_name,
            weight=self.config.weight,
            score=score,
            decision=EvaluationDecision.PASS if score >= 0.7 else EvaluationDecision.REVIEW,
            concerns=concerns,
        )

    def _provider_adjusted_score(self, base_score: float) -> float:
        # Deterministic provider-specific variance lets tests exercise cross-judge aggregation
        # without calling external model APIs.
        name = f"{self.config.provider}:{self.config.model}:{self.config.name}".lower()
        adjustment = ((sum(ord(char) for char in name) % 7) - 3) / 100
        return min(1.0, max(0.0, base_score + adjustment))
