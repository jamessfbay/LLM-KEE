from llm_kee.config import EvaluationConfig, Settings
from llm_kee.models import AggregatedEvaluation, EvaluationDecision, LearningDecisionType, ProposalStatus, ProposalType, TargetType, UpdateProposal
from llm_kee.services import KEEEngine
from llm_kee.evaluation import EvidenceChecker


def test_screening_cannot_auto_approve_by_default(tmp_path) -> None:
    engine = KEEEngine(Settings(workspace=tmp_path))
    aggregate = AggregatedEvaluation(
        proposal_id="proposal",
        final_score=1,
        agreement_level=1,
        recommendation=EvaluationDecision.PASS,
        evaluator_count=5,
    )

    assert engine.gate.decide(aggregate).decision == LearningDecisionType.PENDING_REVIEW


def test_nox_mode_blocks_direct_apply_even_for_approved_proposal(tmp_path) -> None:
    engine = KEEEngine(Settings(workspace=tmp_path, runtime_mode="nox_adapter", evaluation=EvaluationConfig()))
    proposal = UpdateProposal(
        proposal_type=ProposalType.UPDATE_CLAIM,
        target_type=TargetType.CLAIM,
        title="Candidate only",
        rationale="NOX must authorize this change.",
        proposed_change={"new_value": "candidate"},
        status=ProposalStatus.APPROVED,
    )

    result = engine.safe_apply.apply(proposal)

    assert result["status"] == "rejected"
    assert "NOX owns" in result["message"]
    assert proposal.status == ProposalStatus.APPROVED


def test_nox_mode_blocks_local_proposal_approval(tmp_path) -> None:
    engine = KEEEngine(Settings(workspace=tmp_path, runtime_mode="nox_adapter"))
    proposal = UpdateProposal(
        proposal_type=ProposalType.UPDATE_CLAIM,
        target_type=TargetType.CLAIM,
        title="Candidate only",
        rationale="NOX owns activation.",
    )

    try:
        engine.approve_proposal(proposal)
    except RuntimeError as exc:
        assert "does not allow KEE" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("NOX mode must reject local KEE approval")


def test_evidence_checker_requires_resolved_immutable_binding() -> None:
    proposal = UpdateProposal(
        proposal_type=ProposalType.UPDATE_CLAIM,
        target_type=TargetType.CLAIM,
        title="Bound evidence",
        rationale="The source contains the proposed value.",
        evidence_ids=["ev"],
        proposed_change={"new_value": "value"},
    )
    checker = EvidenceChecker(lambda _id: {"valid": True})

    result = checker.evaluate(proposal)

    assert result.decision == EvaluationDecision.PASS
    assert result.evidence_refs == ["ev"]
    assert result.score_semantics == "ordinal_screening_not_probability"


def test_engine_marks_accepted_position_bound_kg_evidence_valid(tmp_path) -> None:
    class AcceptedEvidenceClient:
        def get_object(self, object_type, object_id):  # type: ignore[no-untyped-def]
            assert (object_type, object_id) == ("evidence", "ev")
            return {
                "verification": {
                    "valid": True,
                    "review_state": "approved",
                    "evidence": [
                        {
                            "source_content_hash": "sha256",
                            "quote_start": 4,
                            "quote_end": 12,
                        }
                    ],
                }
            }

    engine = KEEEngine(Settings(workspace=tmp_path))
    engine.kg_client = AcceptedEvidenceClient()  # type: ignore[assignment]

    assert engine._resolve_evidence("ev")["valid"] is True  # type: ignore[index]
