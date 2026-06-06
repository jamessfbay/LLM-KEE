from llm_kee.models import (
    AggregatedEvaluation,
    EvaluationDecision,
    LearningDecision,
    LearningDecisionType,
)


class LearningGate:
    def decide(self, aggregate: AggregatedEvaluation) -> LearningDecision:
        if aggregate.recommendation == EvaluationDecision.FAIL:
            return LearningDecision(
                proposal_id=aggregate.proposal_id,
                decision=LearningDecisionType.REJECT,
                reason="One or more evaluators failed the proposal.",
                final_score=aggregate.final_score,
            )
        if aggregate.recommendation == EvaluationDecision.CONFLICT:
            return LearningDecision(
                proposal_id=aggregate.proposal_id,
                decision=LearningDecisionType.CONFLICT_REVIEW,
                reason="Conflict checker found a potential contradiction.",
                final_score=aggregate.final_score,
                required_actions=["Review conflicting graph records."],
            )
        if aggregate.recommendation == EvaluationDecision.NEED_MORE_EVIDENCE:
            return LearningDecision(
                proposal_id=aggregate.proposal_id,
                decision=LearningDecisionType.NEED_MORE_EVIDENCE,
                reason="Evidence checker requires stronger source support.",
                final_score=aggregate.final_score,
                required_actions=["Attach evidence IDs or source references."],
            )
        if aggregate.final_score >= 0.85 and aggregate.agreement_level >= 0.8:
            return LearningDecision(
                proposal_id=aggregate.proposal_id,
                decision=LearningDecisionType.AUTO_APPLY,
                reason="Evaluators agree with high confidence.",
                final_score=aggregate.final_score,
            )
        return LearningDecision(
            proposal_id=aggregate.proposal_id,
            decision=LearningDecisionType.PENDING_REVIEW,
            reason="Proposal is plausible but needs human review.",
            final_score=aggregate.final_score,
        )
