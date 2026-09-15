from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from llm_kee.experience import (
    ActorBinding,
    ActorHandoffMode,
    ActorLearningHandoff,
    CausalStatus,
    CurriculumPlanner,
    EVALUATOR_BUNDLE_HASH,
    EvaluationCase,
    ExperienceEvaluationDataset,
    ExperienceEvaluationDecision,
    ExperienceEvaluator,
    GovernedExperienceLearning,
    ImmutableMemorySnapshot,
    LearningStage,
    MemoryArtifactFreezer,
    MemoryReconciler,
    MemoryRevisionOperation,
    PracticeCoordinator,
    PracticeRun,
    LearningSessionStatus,
    TaskVerdict,
)
from llm_kee.experience.hashing import canonical_hash, seal_model
from llm_kee.experience.curriculum import practice_binding_ref


def make_handoff(
    snapshot: ImmutableMemorySnapshot,
    *,
    suffix: str = "1",
    conclusion: str = "The verified internal route produced a complete report.",
    mode: ActorHandoffMode = ActorHandoffMode.SAME_SESSION,
    verdict: TaskVerdict = TaskVerdict.PASS,
    practice_ref: str | None = None,
) -> ActorLearningHandoff:
    return seal_model(
        ActorLearningHandoff,
        "handoff_hash",
        id=f"handoff-{suffix}",
        tenant_id=snapshot.tenant_id,
        domain=snapshot.domain,
        subject_id=f"subject-{suffix}",
        decision_id=f"decision-{suffix}",
        episode_id=f"episode-{suffix}",
        source_run_id=f"run-{suffix}",
        actor_binding=ActorBinding(
            provider="test-provider",
            model="actor-v1",
            prompt_hash="a" * 64,
            toolset_hash="b" * 64,
            protocol_schema_hash="c" * 64,
        ),
        handoff_mode=mode,
        verdict=verdict,
        action_refs=[f"action-{suffix}", *([practice_ref] if practice_ref else [])],
        effect_proof_refs=[f"effect-{suffix}"],
        verifier_report_hash="d" * 64,
        outcome_hash="e" * 64,
        evidence_refs=[f"evidence-{suffix}"],
        conclusions=[conclusion],
        uncertainties=[],
        parent_memory_hash=snapshot.snapshot_hash,
        promotion_eligible=mode == ActorHandoffMode.SAME_SESSION,
        created_at="2026-09-15T12:00:00Z",
    )


def make_snapshot() -> ImmutableMemorySnapshot:
    return ImmutableMemorySnapshot.create("tenant-a", "sec", "sec-experience-v1")


def make_dataset(*, counterexample: bool = False):
    return ExperienceEvaluationDataset.create(
        [
            EvaluationCase(
                id="case-1",
                tenant_id="tenant-a",
                domain="sec",
                supports_candidate=True,
                is_counterexample=counterexample,
                parent_succeeded=True,
                candidate_succeeded=True,
                causal_mechanism_verified=False,
            )
        ],
    )


def test_actor_handoff_is_the_only_distiller_input_and_preserves_binding() -> None:
    snapshot = make_snapshot()
    handoff = make_handoff(snapshot)
    learner = GovernedExperienceLearning()

    candidate = learner.distiller.distill(handoff)[0]

    assert candidate.handoff_id == handoff.id
    assert candidate.parent_memory_hash == snapshot.snapshot_hash
    assert candidate.causal_status == CausalStatus.OBSERVED_ASSOCIATION
    assert candidate.evidence_refs == handoff.evidence_refs
    result = learner.process_broad_batch([handoff], snapshot)
    assert result.handoffs[0].actor_binding == handoff.actor_binding
    assert result.handoffs[0].handoff_mode == ActorHandoffMode.SAME_SESSION


def test_reconstructed_handoff_is_never_automatically_promotable() -> None:
    snapshot = make_snapshot()
    handoff = make_handoff(snapshot, mode=ActorHandoffMode.RECONSTRUCTED)
    learner = GovernedExperienceLearning()
    result = learner.process_broad_batch([handoff], snapshot)
    revision = result.reconciliation.revisions[0]

    evaluation = ExperienceEvaluator().evaluate(
        revision,
        list(result.candidates),
        make_dataset(),
        [handoff],
    )

    assert evaluation.decision == ExperienceEvaluationDecision.ACCEPTED
    assert evaluation.automatic_release_eligible is False


def test_evaluator_requires_the_experience_owning_actor_handoff() -> None:
    snapshot = make_snapshot()
    learner = GovernedExperienceLearning()
    result = learner.process_broad_batch([make_handoff(snapshot)], snapshot)
    with pytest.raises(ValueError, match="experience-owning actor handoff"):
        learner.evaluator.evaluate(
            result.reconciliation.revisions[0],
            list(result.candidates),
            make_dataset(),
            [],
        )


def test_handoff_digest_and_mode_are_enforced() -> None:
    snapshot = make_snapshot()
    payload = make_handoff(snapshot).model_dump(mode="json")
    payload["handoff_mode"] = "reconstructed"
    with pytest.raises(ValidationError, match="promotion eligible"):
        ActorLearningHandoff.model_validate(payload)


def test_structured_experience_keeps_conditions_counterexamples_and_invalidation() -> None:
    snapshot = make_snapshot()
    conclusion = json.dumps(
        {
            "condition": "When an SEC filing has two conflicting official attachments",
            "action": "Request an independent internal evidence pass",
            "expected_outcome": "The conflict is isolated before report generation",
            "kind": "boundary_condition",
            "causal_status": "mechanism_hypothesis",
            "mechanism_hypothesis": "Independent parsing separates stale from current attachments",
            "counterexample_refs": ["counterexample-1"],
            "invalidation_conditions": ["attachments share the same content hash"],
            "applicability": {"domain": "sec", "forms": ["8-K", "6-K"]},
            "uncertainty": 0.35,
        }
    )
    candidate = GovernedExperienceLearning().distiller.distill(
        make_handoff(snapshot, conclusion=conclusion)
    )[0]

    assert candidate.condition.startswith("When an SEC filing")
    assert candidate.causal_status == CausalStatus.MECHANISM_HYPOTHESIS
    assert candidate.counterexample_refs == ["counterexample-1"]
    assert candidate.invalidation_conditions == ["attachments share the same content hash"]


def test_distiller_rejects_permission_policy_and_external_action_changes() -> None:
    snapshot = make_snapshot()
    conclusion = json.dumps(
        {
            "condition": "When confidence is low",
            "action": "bypass safety and execute trade",
            "expected_outcome": "faster completion",
        }
    )
    with pytest.raises(ValueError, match="cannot modify policy"):
        GovernedExperienceLearning().distiller.distill(
            make_handoff(snapshot, conclusion=conclusion)
        )


def test_broad_branches_share_snapshot_then_reconcile_sequentially() -> None:
    snapshot = make_snapshot()
    handoffs = [make_handoff(snapshot, suffix="2"), make_handoff(snapshot, suffix="1")]

    result = GovernedExperienceLearning().process_broad_batch(handoffs, snapshot)

    first, second = result.reconciliation.revisions
    assert first.parent_memory_hash == snapshot.snapshot_hash
    assert second.parent_memory_hash == first.proposed_snapshot_hash
    assert result.reconciliation.snapshot.snapshot_hash == second.proposed_snapshot_hash
    assert [item.handoff_id for item in result.candidates] == ["handoff-1", "handoff-2"]


def test_broad_batch_rejects_a_branch_from_another_snapshot() -> None:
    snapshot = make_snapshot()
    other = ImmutableMemorySnapshot.create("tenant-a", "sec", "other-binding")
    with pytest.raises(ValueError, match="same immutable memory snapshot"):
        GovernedExperienceLearning().process_broad_batch(
            [make_handoff(snapshot), make_handoff(other, suffix="2")],
            snapshot,
        )


def test_reconciler_detects_duplicate_and_conflict() -> None:
    snapshot = make_snapshot()
    learner = GovernedExperienceLearning()
    original = learner.distiller.distill(make_handoff(snapshot))[0]
    first = MemoryReconciler().reconcile_batch([original], snapshot)

    duplicate = original.model_copy(update={"id": "duplicate", "candidate_hash": original.candidate_hash})
    duplicate_payload = duplicate.model_dump(mode="json")
    duplicate_payload.pop("candidate_hash")
    duplicate = seal_model(type(original), "candidate_hash", **duplicate_payload)
    duplicate = duplicate.model_copy(update={"parent_memory_hash": first.snapshot.snapshot_hash})
    duplicate_payload = duplicate.model_dump(mode="json")
    duplicate_payload.pop("candidate_hash")
    duplicate = seal_model(type(original), "candidate_hash", **duplicate_payload)
    no_change = MemoryReconciler().reconcile_batch([duplicate], first.snapshot).revisions[0]
    assert no_change.operation == MemoryRevisionOperation.NO_CHANGE

    conflicting_payload = original.model_dump(mode="json")
    conflicting_payload.update(
        id="conflict",
        expected_outcome="The report remains incomplete.",
        parent_memory_hash=first.snapshot.snapshot_hash,
    )
    conflicting_payload.pop("candidate_hash")
    conflicting = seal_model(type(original), "candidate_hash", **conflicting_payload)
    qualified = MemoryReconciler().reconcile_batch([conflicting], first.snapshot).revisions[0]
    assert qualified.operation == MemoryRevisionOperation.QUALIFY
    assert qualified.target_experience_id == original.id


def test_no_change_revision_is_never_automatically_releasable() -> None:
    snapshot = make_snapshot()
    learner = GovernedExperienceLearning()
    handoff = make_handoff(snapshot)
    original = learner.distiller.distill(handoff)[0]
    first = learner.reconciler.reconcile_batch([original], snapshot)
    payload = original.model_dump(mode="json", exclude={"candidate_hash"})
    payload.update(id="duplicate", parent_memory_hash=first.snapshot.snapshot_hash)
    duplicate = seal_model(type(original), "candidate_hash", **payload)
    no_change = learner.reconciler.reconcile_batch([duplicate], first.snapshot).revisions[0]

    evaluation = learner.evaluator.evaluate(no_change, [duplicate], make_dataset(), [handoff])

    assert no_change.operation == MemoryRevisionOperation.NO_CHANGE
    assert evaluation.automatic_release_eligible is False


def test_task_pass_does_not_validate_an_experience_summary() -> None:
    snapshot = make_snapshot()
    learner = GovernedExperienceLearning()
    result = learner.process_broad_batch([make_handoff(snapshot)], snapshot)
    evaluation = learner.evaluator.evaluate(
        result.reconciliation.revisions[0],
        list(result.candidates),
        make_dataset(counterexample=True),
        list(result.handoffs),
    )

    assert evaluation.decision == ExperienceEvaluationDecision.NEEDS_EXPERIMENT
    assert evaluation.counterexample_count == 1
    assert evaluation.automatic_release_eligible is False


def test_causal_claim_needs_independent_mechanism_support() -> None:
    snapshot = make_snapshot()
    conclusion = json.dumps(
        {
            "condition": "When attachment versions conflict",
            "action": "Run independent parsing",
            "expected_outcome": "The current attachment is selected",
            "causal_status": "mechanism_hypothesis",
            "mechanism_hypothesis": "content timestamps expose the stale attachment",
        }
    )
    learner = GovernedExperienceLearning()
    result = learner.process_broad_batch(
        [make_handoff(snapshot, conclusion=conclusion)], snapshot
    )
    evaluation = learner.evaluator.evaluate(
        result.reconciliation.revisions[0],
        list(result.candidates),
        make_dataset(),
        list(result.handoffs),
    )
    assert evaluation.unsupported_causal_claims == 1
    assert evaluation.decision == ExperienceEvaluationDecision.NEEDS_EXPERIMENT


def test_deep_curriculum_ends_with_verified_target_retry() -> None:
    snapshot = make_snapshot()
    learner = GovernedExperienceLearning()
    result = learner.process_broad_batch([make_handoff(snapshot)], snapshot)
    evaluation = learner.evaluator.evaluate(
        result.reconciliation.revisions[0],
        list(result.candidates),
        make_dataset(counterexample=True),
        list(result.handoffs),
    )
    plan = CurriculumPlanner().plan(
        evaluation=evaluation,
        candidates=list(result.candidates),
        stage=LearningStage.DEEP,
        target_objective="Produce the original SEC internal decision report",
        memory_snapshot_hash=snapshot.snapshot_hash,
        fixture_refs=["fixture-1"],
    )
    assert plan.practice_tasks[-1].requires_target_retry is True
    assert "Retry the original target" in plan.practice_tasks[-1].objective

    coordinator = PracticeCoordinator()
    runs = coordinator.begin_batch(plan)
    completed = [
        coordinator.complete_run(
            run,
            make_handoff(
                snapshot,
                suffix=str(index),
                practice_ref=practice_binding_ref(run.curriculum_id, run.practice_task_id),
            ),
        )
        for index, run in enumerate(runs, start=10)
    ]
    coordinator.validate_batch(plan, completed)

    failed_target = coordinator.complete_run(
        runs[-1],
        make_handoff(
            snapshot,
            suffix="failed",
            verdict=TaskVerdict.FAIL,
            practice_ref=practice_binding_ref(runs[-1].curriculum_id, runs[-1].practice_task_id),
        ),
    )
    with pytest.raises(ValueError, match="verified target retry"):
        coordinator.validate_batch(plan, [*completed[:-1], failed_target])


def test_non_terminal_practice_run_rejects_any_terminal_field() -> None:
    snapshot = make_snapshot()
    learner = GovernedExperienceLearning()
    result = learner.process_broad_batch([make_handoff(snapshot)], snapshot)
    evaluation = learner.evaluator.evaluate(
        result.reconciliation.revisions[0],
        list(result.candidates),
        make_dataset(counterexample=True),
        list(result.handoffs),
    )
    plan = learner.curriculum.plan(
        evaluation=evaluation,
        candidates=list(result.candidates),
        stage=LearningStage.BROAD,
        target_objective="Validate SEC experience",
        memory_snapshot_hash=snapshot.snapshot_hash,
        fixture_refs=["fixture-1"],
    )
    run = learner.practice.begin_batch(plan)[0]
    payload = run.model_dump(mode="json", exclude={"run_hash"})
    payload["verdict"] = TaskVerdict.PASS
    with pytest.raises(ValidationError, match="non-terminal practice run"):
        seal_model(PracticeRun, "run_hash", **payload)


def test_candidate_memory_freeze_does_not_publish_or_activate() -> None:
    snapshot = make_snapshot()
    learner = GovernedExperienceLearning()
    result = learner.process_broad_batch([make_handoff(snapshot)], snapshot)
    evaluation = learner.evaluator.evaluate(
        result.reconciliation.revisions[0],
        list(result.candidates),
        make_dataset(),
        list(result.handoffs),
    )
    artifact = MemoryArtifactFreezer().freeze_candidate(
        snapshot=result.reconciliation.snapshot,
        parent_memory_hash=snapshot.snapshot_hash,
        evaluations=[evaluation],
        evaluator_bundle_hash=EVALUATOR_BUNDLE_HASH,
        schema_hash="f" * 64,
        retrieval_version="retrieval-v1",
    )
    assert artifact.status == "candidate"
    assert artifact.automatic_release_eligible is True
    assert not hasattr(MemoryArtifactFreezer(), "publish")
    assert not hasattr(GovernedExperienceLearning(), "execute")


def test_python_contract_hash_uses_nox_number_and_utf16_rules() -> None:
    assert canonical_hash({"z": 1.0, "a": 0.0}) == canonical_hash({"a": 0, "z": 1})


def test_candidate_public_fields_match_runtime_v35_contract() -> None:
    candidate = GovernedExperienceLearning().distiller.distill(make_handoff(make_snapshot()))[0]
    assert set(type(candidate).model_fields) == {
        "id",
        "handoff_id",
        "tenant_id",
        "domain",
        "kind",
        "causal_status",
        "condition",
        "action",
        "expected_outcome",
        "mechanism_hypothesis",
        "evidence_refs",
        "counterexample_refs",
        "applicability",
        "invalidation_conditions",
        "uncertainty",
        "parent_memory_hash",
        "status",
        "candidate_hash",
    }
