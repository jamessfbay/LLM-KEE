from llm_kee.config import EvaluationConfig, JudgeConfig, Settings
from llm_kee.models import (
    EvaluationDecision,
    FeedbackType,
    TargetType,
    UserFeedback,
)
from llm_kee.services import KEEEngine


def test_openai_judge_without_api_key_falls_back_to_review(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    engine = KEEEngine(
        Settings(
            workspace=tmp_path,
            evaluation=EvaluationConfig(
                judges=[
                    JudgeConfig(
                        name="openai_judge",
                        provider="openai",
                        model="gpt-4.1-mini",
                        api_key_env="OPENAI_API_KEY",
                    )
                ]
            ),
        )
    )
    feedback = UserFeedback(
        target_type=TargetType.CLAIM,
        target_id="claim_1",
        feedback_type=FeedbackType.CORRECTION,
        new_value={"text": "A requires C.", "evidence_ids": ["ev_1"]},
        comment="Corrected by user.",
    )

    _, proposal = engine.accept_feedback(feedback)
    results, _ = engine.run_evaluations(proposal)
    llm_result = next(result for result in results if result.evaluator_name == "openai_judge")

    assert llm_result.decision == EvaluationDecision.REVIEW
    assert any("missing API key" in concern for concern in llm_result.concerns)


def test_disabled_judge_is_skipped(tmp_path):
    engine = KEEEngine(
        Settings(
            workspace=tmp_path,
            evaluation=EvaluationConfig(
                judges=[
                    JudgeConfig(name="disabled", provider="mock", enabled=False),
                ]
            ),
        )
    )

    assert all(getattr(evaluator, "evaluator_name", None) != "disabled" for evaluator in engine.evaluators)
