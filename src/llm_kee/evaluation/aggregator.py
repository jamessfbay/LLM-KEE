from collections import Counter

from llm_kee.models import AggregatedEvaluation, EvaluationDecision, EvaluationResult


class EvaluationAggregator:
    def aggregate(self, proposal_id: str, results: list[EvaluationResult]) -> AggregatedEvaluation:
        if not results:
            return AggregatedEvaluation(
                proposal_id=proposal_id,
                final_score=0.0,
                agreement_level=0.0,
                recommendation=EvaluationDecision.REVIEW,
                evaluator_count=0,
                concerns=["No evaluator results were produced."],
            )

        total_weight = sum(result.weight for result in results) or len(results)
        final_score = sum(result.score * result.weight for result in results) / total_weight
        decisions = [result.decision for result in results]
        most_common, count = Counter(decisions).most_common(1)[0]
        agreement = count / len(results)
        concerns = [concern for result in results for concern in result.concerns]

        recommendation = EvaluationDecision.PASS
        if EvaluationDecision.FAIL in decisions:
            recommendation = EvaluationDecision.FAIL
        elif EvaluationDecision.CONFLICT in decisions:
            recommendation = EvaluationDecision.CONFLICT
        elif EvaluationDecision.NEED_MORE_EVIDENCE in decisions:
            recommendation = EvaluationDecision.NEED_MORE_EVIDENCE
        elif final_score < 0.7 or EvaluationDecision.REVIEW in decisions:
            recommendation = EvaluationDecision.REVIEW
        else:
            recommendation = most_common

        return AggregatedEvaluation(
            proposal_id=proposal_id,
            final_score=final_score,
            agreement_level=agreement,
            recommendation=recommendation,
            evaluator_count=len(results),
            concerns=concerns,
        )
