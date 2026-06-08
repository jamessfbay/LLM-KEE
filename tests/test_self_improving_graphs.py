from llm_kee.config import Settings
from llm_kee.models import (
    ConversationIntent,
    FailureRecord,
    ImprovementAction,
    ImprovementProposal,
    IntentPattern,
    TaskIntent,
)
from llm_kee.services import KEEEngine
from llm_kee.storage import KEEStore


def test_intent_failure_improvement_models_round_trip_through_json_store(tmp_path):
    store = KEEStore(tmp_path)
    pattern = store.intent_patterns.upsert(
        IntentPattern(
            intent_type="permit_history_lookup",
            description="Find permit records.",
            trigger_terms=["permit"],
        )
    )
    task_intent = store.task_intents.upsert(
        TaskIntent(
            intent_type="permit_history_lookup",
            task_type="permit_history_lookup",
            matched_pattern_ids=[pattern.id],
        )
    )
    conversation_intent = store.conversation_intents.upsert(
        ConversationIntent(
            intent_type="permit_history_lookup",
            utterance="Find permit history.",
            matched_pattern_ids=[pattern.id],
        )
    )
    failure = store.failure_records.upsert(
        FailureRecord(
            failure_type="missing_source",
            summary="Permit source is missing.",
            intent_id=conversation_intent.id,
        )
    )
    action = store.improvement_actions.upsert(
        ImprovementAction(
            action_type="add_source_ingestion",
            description="Add permit source ingestion.",
        )
    )
    proposal = store.improvement_proposals.upsert(
        ImprovementProposal(
            improvement_type="add_source_ingestion",
            title="Add permit ingestion",
            description="Permit source is missing.",
            failure_ids=[failure.id],
            intent_ids=[task_intent.id, conversation_intent.id],
            actions=[action],
        )
    )

    reloaded = KEEStore(tmp_path)
    assert reloaded.intent_patterns.get(pattern.id).intent_type == "permit_history_lookup"
    assert reloaded.task_intents.get(task_intent.id).matched_pattern_ids == [pattern.id]
    assert reloaded.conversation_intents.get(conversation_intent.id).utterance == "Find permit history."
    assert reloaded.failure_records.get(failure.id).failure_type == "missing_source"
    assert reloaded.improvement_actions.get(action.id).action_type == "add_source_ingestion"
    assert reloaded.improvement_proposals.get(proposal.id).status == "pending_review"


def test_default_intent_patterns_are_registered(tmp_path):
    engine = KEEEngine(Settings(workspace=tmp_path))

    pattern_ids = {pattern.id for pattern in engine.store.intent_patterns.list()}

    assert "permit_history_lookup" in pattern_ids
    assert "approval_risk_analysis" in pattern_ids
