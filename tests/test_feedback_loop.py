from llm_kee.config import Settings
from llm_kee.models import (
    EvaluatorType,
    FeedbackType,
    LearningDecisionType,
    ProposalStatus,
    TargetType,
    UserFeedback,
)
from llm_kee.services import KEEEngine


def test_feedback_creates_proposal_and_needs_evidence(tmp_path):
    engine = KEEEngine(Settings(workspace=tmp_path))
    feedback = UserFeedback(
        target_type=TargetType.CLAIM,
        target_id="claim_1",
        feedback_type=FeedbackType.CORRECTION,
        old_value={"text": "A requires B"},
        new_value={"text": "A requires C"},
        comment="The source says C.",
    )

    saved_feedback, proposal = engine.accept_feedback(feedback)
    results, aggregate = engine.run_evaluations(proposal)
    decision = engine.store.decisions.list()[0]

    assert saved_feedback.status == "proposed"
    assert proposal.target_id == "claim_1"
    assert len(results) == 7
    assert sum(result.evaluator_type == EvaluatorType.LLM_JUDGE for result in results) == 3
    assert aggregate.proposal_id == proposal.id
    assert decision.decision == LearningDecisionType.NEED_MORE_EVIDENCE
    assert engine.store.proposals.get(proposal.id).status == ProposalStatus.NEED_MORE_EVIDENCE


def test_evidence_backed_feedback_moves_to_review_or_approval(tmp_path):
    engine = KEEEngine(Settings(workspace=tmp_path))
    feedback = UserFeedback(
        target_type=TargetType.CLAIM,
        target_id="claim_2",
        feedback_type=FeedbackType.CORRECTION,
        old_value={"text": "A requires B"},
        new_value={"text": "A requires C", "evidence_ids": ["ev_1"]},
        comment="The source says C.",
    )

    _, proposal = engine.accept_feedback(feedback)
    _, aggregate = engine.run_evaluations(proposal)
    decision = engine.store.decisions.list()[0]

    assert aggregate.final_score >= 0.7
    assert decision.decision in {
        LearningDecisionType.AUTO_APPLY,
        LearningDecisionType.PENDING_REVIEW,
    }


def test_reusable_trace_creates_pattern(tmp_path):
    from llm_kee.models import ReasoningStep, ReasoningTrace

    engine = KEEEngine(Settings(workspace=tmp_path))
    trace = ReasoningTrace(
        question="Which policy affects Project Alpha?",
        final_answer="Policy SB 330 affects Project Alpha.",
        reasoning_steps=[
            ReasoningStep(order=1, description="Find policy claims for Project Alpha.")
        ],
        used_claim_ids=["claim_1"],
        used_evidence_ids=["ev_1"],
        confidence=0.8,
        reusable=True,
    )

    engine.save_trace(trace)

    patterns = engine.store.patterns.list()
    assert len(patterns) == 1
    assert patterns[0].source_trace_ids == [trace.id]


def test_settings_can_configure_multiple_llm_judges(tmp_path):
    from llm_kee.config import EvaluationConfig, JudgeConfig

    settings = Settings(
        workspace=tmp_path,
        evaluation=EvaluationConfig(
            judges=[
                JudgeConfig(name="openai_judge", provider="mock", model="gpt-4.1-mini"),
                JudgeConfig(name="claude_judge", provider="mock", model="claude-3-5-sonnet"),
                JudgeConfig(name="disabled_judge", enabled=False),
            ]
        ),
    )
    engine = KEEEngine(settings)

    judge_names = [
        evaluator.evaluator_name
        for evaluator in engine.evaluators
        if getattr(evaluator, "evaluator_type", None) == EvaluatorType.LLM_JUDGE
    ]

    assert judge_names == ["openai_judge", "claude_judge"]
