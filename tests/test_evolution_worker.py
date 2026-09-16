import json

import pytest

from llm_kee.experience.evolution_worker import (
    EvolutionConfig,
    EvolutionWorker,
    SandboxPracticeRunner,
)
from llm_kee.experience.hashing import seal_model
from llm_kee.experience.curriculum import practice_binding_ref
from llm_kee.experience.models import (
    ActorBinding,
    ActorHandoffMode,
    ActorLearningHandoff,
    ImmutableMemorySnapshot,
    TaskVerdict,
)
from llm_kee.experience.service import GovernedExperienceLearning


def source_handoff(
    parent_hash: str,
    handoff_id: str = "handoff-source",
    practice_ref: str | None = None,
) -> ActorLearningHandoff:
    return seal_model(
        ActorLearningHandoff,
        "handoff_hash",
        id=handoff_id,
        tenant_id="tenant-a",
        domain="financial_disclosure",
        subject_id="issuer-a",
        decision_id=f"decision-{handoff_id}",
        episode_id=f"episode-{handoff_id}",
        source_run_id=f"run-{handoff_id}",
        actor_binding=ActorBinding(
            provider="test",
            model="actor-v1",
            prompt_hash="a" * 64,
            toolset_hash="b" * 64,
            protocol_schema_hash="c" * 64,
        ),
        handoff_mode=ActorHandoffMode.SAME_SESSION,
        verdict=TaskVerdict.PASS,
        action_refs=["internal-report", *([practice_ref] if practice_ref else [])],
        effect_proof_refs=["proof"],
        verifier_report_hash="d" * 64,
        outcome_hash="e" * 64,
        evidence_refs=["sec:official:fixture"],
        conclusions=["Use the verified internal-report procedure for this bounded filing type."],
        uncertainties=[],
        parent_memory_hash=parent_hash,
        promotion_eligible=True,
        created_at="2026-09-15T12:00:00Z",
    )


def test_evolution_worker_waits_for_independent_practice_then_freezes_artifact() -> None:
    snapshot = ImmutableMemorySnapshot.create(
        "tenant-a", "financial_disclosure", "kee-financial_disclosure"
    )
    source = source_handoff(snapshot.snapshot_hash)
    batch = GovernedExperienceLearning().build_atomic_batch(source, snapshot)
    events = [
        {"sequence": 1, "event_type": "experience.handoff_ready", "subject_id": source.id, "data": source.model_dump(mode="json")},
        {"sequence": 2, "event_type": "experience.distilled", "subject_id": batch.candidates[0].id, "data": batch.candidates[0].model_dump(mode="json")},
        {"sequence": 3, "event_type": "experience.reconciled", "subject_id": batch.revisions[0].id, "data": batch.revisions[0].model_dump(mode="json")},
    ]
    submitted: list[tuple[str, dict]] = []

    def request(method: str, path: str, payload: dict | None) -> dict:
        if method == "GET" and "experience-events" in path:
            return {"items": events, "next": None}
        if method == "GET" and "/active?" in path:
            return {
                "artifact_hash": snapshot.snapshot_hash,
                "artifact": {
                    "id": f"memory-genesis-{snapshot.snapshot_hash[:24]}",
                    "tenant_id": snapshot.tenant_id,
                    "domain": snapshot.domain,
                    "memory_binding": snapshot.memory_binding,
                    "artifact_uri": "nox://memory/genesis",
                    "experiences": [],
                },
            }
        assert method == "POST" and payload is not None
        submitted.append((path, payload))
        kind = (
            "experience.evaluated" if "experience-evaluations" in path else
            "curriculum.planned" if "curriculum-plans" in path else
            "practice.completed" if "practice-runs" in path else
            "memory.candidate_created"
        )
        events.append({"sequence": len(events) + 1, "event_type": kind, "subject_id": payload["id"], "data": payload})
        return {"record": payload}

    def practice_runner(plan: dict, task: dict) -> ActorLearningHandoff:
        return source_handoff(
            snapshot.snapshot_hash,
            "handoff-practice",
            practice_binding_ref(plan["id"], task["id"]),
        )

    worker = EvolutionWorker(
        EvolutionConfig(
            "http://state", "evaluator", snapshot.tenant_id, snapshot.domain,
            snapshot.memory_binding, "f" * 64,
        ),
        request=request,
        practice_runner=practice_runner,
    )
    assert worker.run_once() == 3  # bootstrap evaluation, curriculum, completed practice
    assert not any("memory-artifacts" in path for path, _ in submitted)
    assert worker.run_once() == 2  # independent evaluation and immutable artifact
    artifacts = [payload for path, payload in submitted if "memory-artifacts" in path]
    assert len(artifacts) == 1
    assert artifacts[0]["automatic_release_eligible"] is True


def test_evolution_outputs_are_retry_deterministic() -> None:
    snapshot = ImmutableMemorySnapshot.create("tenant-a", "financial_disclosure", "memory-a")
    source = source_handoff(snapshot.snapshot_hash)
    batch = GovernedExperienceLearning().build_atomic_batch(source, snapshot)
    worker = EvolutionWorker(
        EvolutionConfig("http://state", "token", "tenant-a", "financial_disclosure", "memory-a", "f" * 64),
        request=lambda *_: {},
    )
    dataset = worker._bootstrap_dataset(batch.revisions[0], source)
    first = worker.evaluator.evaluate(batch.revisions[0], list(batch.candidates), dataset, [source])
    second = worker.evaluator.evaluate(batch.revisions[0], list(batch.candidates), dataset, [source])
    assert first == second


def test_sandbox_practice_runner_rejects_cross_tenant_handoff() -> None:
    handoff = source_handoff("a" * 64)
    values = handoff.model_dump(mode="json", exclude={"handoff_hash"})
    values["tenant_id"] = "other"
    values["actor_binding"] = handoff.actor_binding
    payload = seal_model(ActorLearningHandoff, "handoff_hash", **values).model_dump(mode="json")

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps({"status": "completed", "handoff": payload}).encode()

    runner = SandboxPracticeRunner(
        "http://practice-runner:8090",
        "practice-token",
        opener=lambda *_args, **_kwargs: Response(),
    )
    with pytest.raises(RuntimeError, match="crossed tenant or domain"):
        runner({"tenant_id": "tenant-a", "domain": "financial_disclosure"}, {"id": "task"})


def test_sandbox_practice_runner_treats_pending_as_no_result() -> None:
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b'{"status":"pending"}'

    runner = SandboxPracticeRunner(
        "http://practice-runner:8090",
        "practice-token",
        opener=lambda *_args, **_kwargs: Response(),
    )
    assert runner({"tenant_id": "tenant-a", "domain": "financial_disclosure"}, {"id": "task"}) is None


def test_evolution_worker_collects_only_assignment_bound_rollout_evidence() -> None:
    assignment = {
        "release_id": "release-a",
        "run_id": "run-a",
        "phase": "shadow",
        "arm": "shadow_candidate",
        "formal_artifact_id": "baseline",
        "formal_artifact_hash": "a" * 64,
        "shadow_artifact_id": "candidate",
        "shadow_artifact_hash": "b" * 64,
    }
    observation = {
        **{key: assignment[key] for key in ("release_id", "run_id", "phase", "arm")},
        "completed": True,
        "evidence_complete": True,
        "requires_review": False,
        "audited": True,
        "policy_violations": 0,
        "unauthorized_actions": 0,
        "missed_escalation": False,
        "major_correction": False,
        "latency_ms": 20,
        "artifact_id": "candidate",
        "artifact_hash": "b" * 64,
        "outcome_hash": "c" * 64,
    }

    class Runner:
        def evaluate_rollout(self, actual):
            assert actual == assignment
            return observation

    submitted = []

    def request(method, path, payload):
        if method == "GET":
            return {"items": [assignment], "next": None}
        submitted.append((path, payload))
        return {"observation_hash": "d" * 64}

    worker = EvolutionWorker(
        EvolutionConfig(
            "http://state", "token", "tenant-a", "financial_disclosure", "memory-a", "f" * 64
        ),
        request=request,
        practice_runner=Runner(),
    )
    assert worker.collect_rollout_observations([]) == 1
    assert submitted == [
        (
            "/api/v3.5/memory-rollout-observations?tenant_id=tenant-a&domain=financial_disclosure",
            observation,
        )
    ]
