from llm_kee.config import Settings
from llm_kee.models import LearningSignal, ProposalStatus, SignalType
from llm_kee.services import KEEEngine


def test_intent_detector_maps_permit_timeline_risk_and_fallback(tmp_path):
    engine = KEEEngine(Settings(workspace=tmp_path))

    permit = engine.detect_intent({"utterance": "Find permit history for this parcel."})
    timeline = engine.detect_intent({"utterance": "Rebuild the approval timeline."})
    risk = engine.detect_intent({"utterance": "What is the approval risk?"})
    fallback = engine.detect_intent({"utterance": "Please help with this unusual task."})

    assert permit.intent_type == "permit_history_lookup"
    assert timeline.intent_type == "timeline_reconstruction"
    assert risk.intent_type == "approval_risk_analysis"
    assert fallback.intent_type == "general_knowledge_task"


def test_failure_and_improvement_graph_create_reviewable_proposal(tmp_path):
    engine = KEEEngine(Settings(workspace=tmp_path))

    missing_source = engine.record_failure(
        {"failure_type": "missing_source", "summary": "Permit source is unavailable."}
    )
    wrong_route = engine.record_failure(
        {"failure_type": "wrong_skill_route", "summary": "Risk task used evidence lookup only."}
    )
    missing_source_proposal = engine.propose_improvement(missing_source.id)
    wrong_route_proposal = engine.propose_improvement(wrong_route.id)

    assert missing_source_proposal.improvement_type == "add_source_ingestion"
    assert missing_source_proposal.status == "pending_review"
    assert wrong_route_proposal.improvement_type == "adjust_skill_routing"
    assert engine.store.skills.list()
    skill_count = len(engine.store.skills.list())

    approved = engine.approve_improvement(wrong_route_proposal.id)

    assert approved.status == "approved"
    assert len(engine.store.skills.list()) == skill_count
    assert engine.store.improvement_reviews.list()[-1].decision == "approved"


def test_knowledge_evolution_loop_wraps_mape_cycle(tmp_path):
    engine = KEEEngine(Settings(workspace=tmp_path))
    signal = LearningSignal(
        signal_type=SignalType.NEW_SOURCE,
        source_id="source_1",
        summary="New project source appeared.",
        payload={"evidence_ids": ["ev_1"]},
        priority=8,
    )

    cycle = engine.run_knowledge_evolution([signal])

    assert cycle.mape_cycle_id
    assert cycle.proposal_ids
    assert cycle.evaluation_ids
    assert cycle.decision_ids
    proposal = engine.store.proposals.get(cycle.proposal_ids[0])
    assert proposal.status in {ProposalStatus.PENDING_REVIEW, ProposalStatus.APPROVED}


def test_skill_selection_loop_records_intent_workflow_and_evaluation(tmp_path):
    engine = KEEEngine(Settings(workspace=tmp_path))

    cycle = engine.run_skill_selection(
        {
            "task_type": "timeline_reconstruction",
            "question": "Rebuild the project timeline.",
            "target_id": "project_1",
            "evidence_ids": ["ev_1"],
        }
    )

    intent = engine.store.task_intents.get(cycle.task_intent_id)
    assert intent.intent_type == "timeline_reconstruction"
    assert cycle.workflow_run_id
    assert cycle.evaluation_record_ids
    assert cycle.status == "completed"


def test_agent_improvement_loop_detects_failure_and_proposes_improvement(tmp_path):
    engine = KEEEngine(Settings(workspace=tmp_path))

    cycle = engine.run_agent_improvement(
        {
            "utterance": "Why can't you find the permit history?",
            "failure_type": "missing_source",
            "summary": "Permit data source is missing.",
            "target_id": "project_1",
        }
    )

    intent = engine.store.conversation_intents.get(cycle.conversation_intent_id)
    failure = engine.store.failure_records.get(cycle.failure_ids[0])
    proposal = engine.store.improvement_proposals.get(cycle.improvement_proposal_ids[0])

    assert intent.intent_type == "permit_history_lookup"
    assert failure.failure_type == "missing_source"
    assert proposal.improvement_type == "add_source_ingestion"
    assert cycle.status == "pending_review"
