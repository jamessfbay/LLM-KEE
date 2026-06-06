from llm_kee.config import Settings
from llm_kee.models import (
    ChangeSet,
    ClaimNode,
    EntityNode,
    EvidenceNode,
    LearningSignal,
    RelationEdge,
    SignalType,
    SourceRecord,
)
from llm_kee.services import KEEEngine
from llm_kee.storage import KEEStore


def test_five_graph_models_round_trip_through_json_store(tmp_path):
    store = KEEStore(tmp_path)
    source = store.sources.upsert(SourceRecord(title="Source A", source_type="memo"))
    evidence = store.evidence_nodes.upsert(
        EvidenceNode(source_id=source.id, quote="A requires C.", confidence=0.9)
    )
    claim = store.claim_nodes.upsert(
        ClaimNode(subject="A", predicate="requires", object="C", evidence_ids=[evidence.id])
    )
    entity = store.entity_nodes.upsert(EntityNode(entity_type="concept", canonical_name="A"))
    relation = store.relation_edges.upsert(
        RelationEdge(subject_id=entity.id, predicate="supports", object_id=claim.id)
    )

    reloaded = KEEStore(tmp_path)
    assert reloaded.sources.get(source.id).title == "Source A"
    assert reloaded.evidence_nodes.get(evidence.id).quote == "A requires C."
    assert reloaded.claim_nodes.get(claim.id).evidence_ids == [evidence.id]
    assert reloaded.entity_nodes.get(entity.id).canonical_name == "A"
    assert reloaded.relation_edges.get(relation.id).object_id == claim.id


def test_default_skills_plan_and_run_workflow(tmp_path):
    engine = KEEEngine(Settings(workspace=tmp_path))

    workflow = engine.plan_workflow(
        {"task_type": "timeline_reconstruction", "target_id": "target_1", "evidence_ids": ["ev_1"]}
    )
    run = engine.run_workflow(workflow)

    assert workflow.skill_sequence == [
        "rag_evidence_retrieval",
        "event_sourcing_timeline",
        "evidence_evaluation",
    ]
    assert run.status == "completed"
    assert [step.status for step in run.steps] == ["completed", "completed", "completed"]
    assert run.steps[0].output["evidence_ids"] == ["ev_1"]


def test_action_run_creates_workflow_and_artifact(tmp_path):
    engine = KEEEngine(Settings(workspace=tmp_path))

    run = engine.run_action(
        "detect_missing_or_conflicting_information",
        {"target_id": "target_1", "evidence_ids": ["ev_1"], "question": "Find conflicts."},
    )

    assert run.status == "completed"
    assert run.workflow_run_id
    assert len(run.artifact_ids) == 1
    artifact = engine.store.action_artifacts.get(run.artifact_ids[0])
    assert artifact.artifact_type == "detect_missing_or_conflicting_information"
    assert engine.validate_artifact(artifact.id).valid is True


def test_evolution_event_supersedes_previous_version(tmp_path):
    engine = KEEEngine(Settings(workspace=tmp_path))
    first = engine.create_evolution_event(
        ChangeSet(
            target_type="claim",
            target_id="claim_1",
            operation="create",
            after={"text": "A requires B."},
            reason="Initial extraction.",
            evidence_ids=["ev_1"],
        )
    )
    second = engine.create_evolution_event(
        ChangeSet(
            target_type="claim",
            target_id="claim_1",
            operation="update",
            before={"text": "A requires B."},
            after={"text": "A requires C."},
            reason="New evidence supersedes old claim.",
            evidence_ids=["ev_2"],
        )
    )

    versions = engine.store.knowledge_versions.list()
    assert first.to_version_id != second.to_version_id
    assert sorted(version.version for version in versions) == [1, 2]
    assert engine.store.knowledge_versions.get(first.to_version_id).status == "superseded"
    assert engine.evolution.history("claim_1")[-1].reason == "New evidence supersedes old claim."


def test_mape_cycle_records_control_loop(tmp_path):
    engine = KEEEngine(Settings(workspace=tmp_path))
    signal = LearningSignal(
        signal_type=SignalType.GRAPH_CONFLICT,
        summary="Conflict detected.",
        priority=9,
    )

    cycle = engine.run_mape_cycle([signal])

    assert cycle.analysis["high_priority_count"] == 1
    assert cycle.plan["requires_review"] is True
    assert cycle.status == "completed"
