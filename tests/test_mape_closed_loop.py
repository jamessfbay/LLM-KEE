from llm_kee.config import Settings
from llm_kee.models import LearningSignal, ProposalStatus, SignalType
from llm_kee.services import KEEEngine


def test_graph_conflict_signal_runs_action_and_creates_gated_proposal(tmp_path):
    engine = KEEEngine(Settings(workspace=tmp_path))
    signal = LearningSignal(
        signal_type=SignalType.GRAPH_CONFLICT,
        source_id="claim_1",
        summary="Conflict detected for claim status.",
        payload={"target_id": "claim_1", "evidence_ids": ["ev_1"], "conflict": "old status contradicts new source"},
        priority=9,
    )

    cycle = engine.run_mape_cycle([signal])

    assert cycle.action_run_ids
    assert cycle.artifact_ids
    assert cycle.proposal_ids
    assert cycle.evaluation_ids
    assert cycle.decision_ids
    proposal = engine.store.proposals.get(cycle.proposal_ids[0])
    assert proposal.status in {ProposalStatus.CONFLICT_REVIEW, ProposalStatus.PENDING_REVIEW}
    assert engine.store.action_runs.get(cycle.action_run_ids[0]).action_type == "detect_missing_or_conflicting_information"


def test_new_source_signal_runs_intelligence_pack_and_proposal(tmp_path):
    engine = KEEEngine(Settings(workspace=tmp_path))
    signal = LearningSignal(
        signal_type=SignalType.NEW_SOURCE,
        source_id="doc_1.md",
        summary="New source document discovered.",
        payload={"target_id": "doc_1.md", "evidence_ids": ["ev_1"]},
        priority=8,
    )

    cycle = engine.run_mape_cycle([signal])

    assert cycle.proposal_ids
    action_run = engine.store.action_runs.get(cycle.action_run_ids[0])
    assert action_run.action_type == "generate_intelligence_pack"


def test_mape_missing_evidence_signal_needs_more_evidence(tmp_path):
    engine = KEEEngine(Settings(workspace=tmp_path))
    signal = LearningSignal(
        signal_type=SignalType.LOW_CONFIDENCE_CLAIM,
        source_id="claim_2",
        summary="Low confidence claim has no evidence.",
        payload={"target_id": "claim_2"},
        priority=8,
    )

    cycle = engine.run_mape_cycle([signal])

    proposal = engine.store.proposals.get(cycle.proposal_ids[0])
    assert proposal.status == ProposalStatus.NEED_MORE_EVIDENCE
