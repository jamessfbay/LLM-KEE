from io import BytesIO
from urllib.error import HTTPError

from llm_kee.experience.hashing import canonical_hash, seal_model
from llm_kee.experience.models import (
    ActorBinding,
    ActorHandoffMode,
    ActorLearningHandoff,
    ImmutableMemorySnapshot,
    TaskVerdict,
)
from llm_kee.experience.worker import ExperienceWorker, WorkerConfig


def handoff(parent_hash: str) -> ActorLearningHandoff:
    return seal_model(
        ActorLearningHandoff,
        "handoff_hash",
        id="handoff-worker-1",
        tenant_id="tenant-a",
        domain="financial_disclosure",
        subject_id="issuer-a",
        decision_id="42",
        episode_id="episode-a",
        source_run_id="run-a",
        actor_binding=ActorBinding(
            provider="openai",
            model="model-a",
            prompt_hash="a" * 64,
            toolset_hash="b" * 64,
            protocol_schema_hash="c" * 64,
        ),
        handoff_mode=ActorHandoffMode.SAME_SESSION,
        verdict=TaskVerdict.PASS,
        action_refs=["internal-report"],
        effect_proof_refs=["effect-proof"],
        verifier_report_hash="d" * 64,
        outcome_hash="e" * 64,
        evidence_refs=["sec:official"],
        conclusions=["The verified internal report procedure was successful."],
        uncertainties=[],
        parent_memory_hash=parent_hash,
        promotion_eligible=True,
        created_at="2026-09-15T12:00:00Z",
    )


def test_worker_submits_one_atomic_retry_safe_batch() -> None:
    config = WorkerConfig(
        base_url="http://state-engine:4318",
        token="learner-token",
        tenant_id="tenant-a",
        domain="financial_disclosure",
        memory_binding="kee-financial_disclosure",
    )
    genesis = ImmutableMemorySnapshot.create(
        config.tenant_id,
        config.domain,
        config.memory_binding,
    )
    source = handoff(genesis.snapshot_hash)
    submissions: list[dict] = []

    def request(method: str, path: str, payload: dict | None) -> dict:
        if "/active?" in path:
            raise HTTPError(
                path,
                404,
                "Not Found",
                {},
                BytesIO(b'{"error":"Active memory artifact not found"}'),
            )
        if "/genesis?" in path:
            return {
                "artifact_hash": genesis.snapshot_hash,
                "canonical": {
                    "tenant_id": config.tenant_id,
                    "domain": config.domain,
                    "memory_binding": config.memory_binding,
                    "experiences": [],
                },
            }
        if "actor-learning-handoffs/pending" in path:
            return {"items": [source.model_dump(mode="json")] if not submissions else []}
        if method == "POST" and "/experience-batches?" in path:
            assert payload is not None
            submissions.append(payload)
            return {"batch": payload, "idempotent_replay": False}
        raise AssertionError((method, path))

    worker = ExperienceWorker(config, request=request)
    assert worker.run_once() == 1
    assert worker.run_once() == 0
    assert len(submissions) == 1
    batch = submissions[0]
    assert batch["handoff_id"] == source.id
    assert len(batch["candidates"]) == len(source.conclusions)
    assert len(batch["revisions"]) == len(source.conclusions)
    assert batch["batch_hash"] == canonical_hash(
        {key: value for key, value in batch.items() if key != "batch_hash"}
    )


def test_distillation_and_reconciliation_ids_are_deterministic() -> None:
    snapshot = ImmutableMemorySnapshot.create("tenant-a", "financial_disclosure", "memory-a")
    source = handoff(snapshot.snapshot_hash)
    worker = ExperienceWorker(
        WorkerConfig("http://state", "token", "tenant-a", "financial_disclosure", "memory-a"),
        request=lambda *_: {},
    )
    first = worker.learning.build_atomic_batch(source, snapshot)
    second = worker.learning.build_atomic_batch(source, snapshot)
    assert first == second


def test_worker_accepts_a_materialized_empty_genesis() -> None:
    config = WorkerConfig("http://state", "token", "tenant-a", "financial_disclosure", "memory-a")
    snapshot = ImmutableMemorySnapshot.create("tenant-a", "financial_disclosure", "memory-a")

    def request(method: str, path: str, payload: dict | None) -> dict:
        assert method == "GET" and "/active?" in path and payload is None
        return {
            "artifact_hash": snapshot.snapshot_hash,
            "artifact": {
                "tenant_id": "tenant-a",
                "domain": "financial_disclosure",
                "memory_binding": "memory-a",
                "artifact_uri": "nox://memory/genesis",
                "experiences": [],
            },
        }

    assert ExperienceWorker(config, request=request).memory_snapshot() == snapshot
