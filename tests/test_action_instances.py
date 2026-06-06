from llm_kee.config import Settings
from llm_kee.services import KEEEngine


def test_action_instances_and_artifacts_can_be_listed(tmp_path):
    engine = KEEEngine(Settings(workspace=tmp_path))

    run = engine.run_action(
        "rebuild_timeline",
        {
            "target_id": "demo_project_alpha",
            "question": "Rebuild the timeline.",
            "evidence_ids": ["ev_1", "ev_2"],
        },
    )

    runs = engine.store.action_runs.list()
    artifacts = engine.store.action_artifacts.list()
    run_artifacts = [
        artifact
        for artifact in artifacts
        if artifact.action_run_id == run.id
    ]

    assert [item.id for item in runs] == [run.id]
    assert len(run_artifacts) == 1
    assert run_artifacts[0].artifact_type == "rebuild_timeline"
    assert run_artifacts[0].evidence_ids == ["ev_1", "ev_2"]
