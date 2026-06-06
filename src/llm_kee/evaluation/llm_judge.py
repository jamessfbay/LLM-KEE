import json
import os

from llm_kee.config import JudgeConfig
from llm_kee.evaluation.evaluators import DeterministicLLMJudge
from llm_kee.models import EvaluationDecision, EvaluationResult, EvaluatorType, UpdateProposal


class LLMJudge:
    evaluator_type = EvaluatorType.LLM_JUDGE

    def __init__(self, config: JudgeConfig) -> None:
        self.config = config
        self.evaluator_name = config.name
        self.mock = DeterministicLLMJudge(config)

    def evaluate(self, proposal: UpdateProposal) -> EvaluationResult:
        if self.config.provider == "mock":
            return self.mock.evaluate(proposal)
        if self.config.provider == "openai":
            return self._evaluate_openai(proposal)
        return EvaluationResult(
            proposal_id=proposal.id,
            evaluator_type=self.evaluator_type,
            evaluator_name=self.evaluator_name,
            weight=self.config.weight,
            score=0.5,
            decision=EvaluationDecision.REVIEW,
            concerns=[f"LLM judge provider is not implemented: {self.config.provider}"],
        )

    def _evaluate_openai(self, proposal: UpdateProposal) -> EvaluationResult:
        api_key_env = self.config.api_key_env or "OPENAI_API_KEY"
        api_key = os.getenv(api_key_env)
        if not api_key:
            result = self.mock.evaluate(proposal)
            result.decision = EvaluationDecision.REVIEW
            result.concerns.append(
                f"OpenAI judge fallback: missing API key environment variable {api_key_env}."
            )
            return result
        try:
            from openai import OpenAI
        except ImportError:
            result = self.mock.evaluate(proposal)
            result.decision = EvaluationDecision.REVIEW
            result.concerns.append("OpenAI judge fallback: install llm-kee[llm] to enable OpenAI.")
            return result

        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict knowledge-governance judge. "
                            "Return JSON with score, decision, concerns, recommended_changes."
                        ),
                    },
                    {"role": "user", "content": json.dumps(proposal.model_dump(mode="json"))},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            payload = json.loads(content)
            score = float(payload.get("score", 0.5))
            decision = payload.get("decision", "review")
            if decision not in {item.value for item in EvaluationDecision}:
                decision = "review"
            return EvaluationResult(
                proposal_id=proposal.id,
                evaluator_type=self.evaluator_type,
                evaluator_name=self.evaluator_name,
                weight=self.config.weight,
                score=max(0.0, min(1.0, score)),
                decision=EvaluationDecision(decision),
                concerns=list(payload.get("concerns") or []),
                recommended_changes=list(payload.get("recommended_changes") or []),
            )
        except Exception as exc:
            result = self.mock.evaluate(proposal)
            result.decision = EvaluationDecision.REVIEW
            result.concerns.append(f"OpenAI judge fallback after provider error: {exc}")
            return result
