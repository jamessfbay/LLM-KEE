from __future__ import annotations

from typing import Iterable

from llm_kee.experience.hashing import canonical_hash, seal_model
from llm_kee.experience.models import (
    ActorLearningHandoff,
    CurriculumBudget,
    CurriculumPlan,
    ExperienceCandidate,
    ExperienceEvaluation,
    LearningSessionStatus,
    LearningStage,
    PracticeRun,
    PracticeTask,
    TaskVerdict,
)


SAFE_PRACTICE_ACTIONS = ["replay", "sandbox_read", "generate_internal_report"]


def practice_binding_ref(curriculum_id: str, practice_task_id: str) -> str:
    return "nox:practice:" + canonical_hash(
        {"curriculum_id": curriculum_id, "practice_task_id": practice_task_id}
    )


class CurriculumPlanner:
    """Plan bounded replay/sandbox practice; it never executes the plan."""

    def plan(
        self,
        *,
        evaluation: ExperienceEvaluation,
        candidates: list[ExperienceCandidate],
        stage: LearningStage,
        target_objective: str,
        memory_snapshot_hash: str,
        fixture_refs: list[str],
        budget: CurriculumBudget | None = None,
    ) -> CurriculumPlan:
        if not target_objective.strip() or not fixture_refs:
            raise ValueError("curriculum requires a target objective and immutable fixtures")
        if any(
            item.tenant_id != evaluation.tenant_id or item.domain != evaluation.domain
            for item in candidates
        ):
            raise ValueError("curriculum candidates must match evaluation scope")
        hypotheses = self._hypotheses(evaluation, candidates)
        plan_identity = canonical_hash(
            {
                "evaluation_hash": evaluation.evaluation_hash,
                "stage": str(stage),
                "target_objective": target_objective,
                "memory_snapshot_hash": memory_snapshot_hash,
                "fixture_refs": list(dict.fromkeys(fixture_refs)),
            }
        )
        tasks = self._tasks(stage, target_objective, hypotheses, fixture_refs, plan_identity)
        selected_budget = budget or CurriculumBudget(
            max_practice_tasks=max(4, len(tasks)),
            max_reasoning_calls=max(2, len(tasks)),
            max_cost_microusd=100_000,
            max_wall_time_ms=900_000,
        )
        return seal_model(
            CurriculumPlan,
            "plan_hash",
            id=f"curriculum_{plan_identity[:24]}",
            evaluation_id=evaluation.id,
            revision_id=evaluation.revision_id,
            candidate_ids=[item.id for item in candidates],
            tenant_id=evaluation.tenant_id,
            domain=evaluation.domain,
            stage=stage,
            target_objective_hash=canonical_hash(target_objective),
            memory_snapshot_hash=memory_snapshot_hash,
            hypotheses=hypotheses,
            practice_tasks=tasks,
            budget=selected_budget,
            stop_conditions=[
                "budget_exhausted",
                "infrastructure_unresolved",
                "independent_verifier_rejects_all_hypotheses",
                "target_retry_verified" if stage == LearningStage.DEEP else "broad_coverage_complete",
            ],
            status=LearningSessionStatus.PLANNED,
            created_at=evaluation.evaluated_at,
        )

    @staticmethod
    def _hypotheses(
        evaluation: ExperienceEvaluation,
        candidates: list[ExperienceCandidate],
    ) -> list[str]:
        hypotheses = list(dict.fromkeys(evaluation.concerns))
        if not hypotheses:
            hypotheses = [
                f"The bounded experience {candidate.id} transfers under its declared applicability."
                for candidate in candidates
            ]
        return hypotheses

    @staticmethod
    def _tasks(
        stage: LearningStage,
        target: str,
        hypotheses: list[str],
        fixture_refs: list[str],
        plan_identity: str,
    ) -> list[PracticeTask]:
        tasks: list[PracticeTask] = []
        for index, hypothesis in enumerate(hypotheses, start=1):
            tasks.append(
                PracticeTask(
                    id=f"practice_task_{canonical_hash({'plan': plan_identity, 'sequence': index})[:24]}",
                    sequence=index,
                    objective=f"Discriminate hypothesis: {hypothesis}",
                    fixture_refs=list(dict.fromkeys(fixture_refs)),
                    allowed_internal_actions=list(SAFE_PRACTICE_ACTIONS),
                    success_criteria=[
                        "independent verifier produces a terminal verdict",
                        "effect proof and outcome hashes are present",
                    ],
                    discriminates_hypotheses=[hypothesis],
                    requires_target_retry=False,
                )
            )
        if stage == LearningStage.DEEP:
            tasks.append(
                PracticeTask(
                    id=f"practice_task_{canonical_hash({'plan': plan_identity, 'sequence': len(tasks) + 1})[:24]}",
                    sequence=len(tasks) + 1,
                    objective=f"Retry the original target after focused practice: {target}",
                    fixture_refs=list(dict.fromkeys(fixture_refs)),
                    allowed_internal_actions=list(SAFE_PRACTICE_ACTIONS),
                    success_criteria=[
                        "original target is retried from a clean environment",
                        "independent verifier confirms the final target result",
                    ],
                    discriminates_hypotheses=list(hypotheses),
                    requires_target_retry=True,
                )
            )
        return tasks


class PracticeCoordinator:
    """Create auditable practice run contracts without executing any action."""

    def begin_batch(self, plan: CurriculumPlan) -> list[PracticeRun]:
        started = plan.created_at
        return [
            seal_model(
                PracticeRun,
                "run_hash",
                id=f"practice_run_{canonical_hash({'plan_hash': plan.plan_hash, 'task_id': task.id})[:24]}",
                curriculum_id=plan.id,
                practice_task_id=task.id,
                tenant_id=plan.tenant_id,
                domain=plan.domain,
                memory_snapshot_hash=plan.memory_snapshot_hash,
                status=LearningSessionStatus.RUNNING,
                verdict=None,
                verifier_report_hash=None,
                outcome_hash=None,
                actor_handoff_id=None,
                started_at=started,
                ended_at=None,
            )
            for task in plan.practice_tasks
        ]

    def complete_run(
        self,
        run: PracticeRun,
        handoff: ActorLearningHandoff,
    ) -> PracticeRun:
        if (
            handoff.tenant_id != run.tenant_id
            or handoff.domain != run.domain
            or handoff.parent_memory_hash != run.memory_snapshot_hash
        ):
            raise ValueError("practice handoff does not match the run's immutable scope and memory")
        if practice_binding_ref(run.curriculum_id, run.practice_task_id) not in handoff.action_refs:
            raise ValueError("practice handoff is not bound to the curriculum task")
        return seal_model(
            PracticeRun,
            "run_hash",
            **{
                **run.model_dump(mode="json", exclude={"run_hash"}),
                "status": LearningSessionStatus.COMPLETED,
                "verdict": handoff.verdict,
                "verifier_report_hash": handoff.verifier_report_hash,
                "outcome_hash": handoff.outcome_hash,
                "actor_handoff_id": handoff.id,
                "ended_at": handoff.created_at,
            },
        )

    @staticmethod
    def validate_batch(plan: CurriculumPlan, runs: Iterable[PracticeRun]) -> None:
        completed = tuple(runs)
        by_task = {run.practice_task_id: run for run in completed}
        if set(by_task) != {task.id for task in plan.practice_tasks}:
            raise ValueError("practice batch does not cover the curriculum exactly once")
        if any(
            run.curriculum_id != plan.id
            or run.tenant_id != plan.tenant_id
            or run.domain != plan.domain
            or run.memory_snapshot_hash != plan.memory_snapshot_hash
            or run.status not in {LearningSessionStatus.COMPLETED, LearningSessionStatus.FAILED}
            for run in completed
        ):
            raise ValueError("practice batch violates scope, snapshot or terminal-state invariants")
        if plan.stage == LearningStage.DEEP:
            target = plan.practice_tasks[-1]
            target_run = by_task[target.id]
            if not target.requires_target_retry or target_run.verdict != TaskVerdict.PASS:
                raise ValueError("deep learning cannot complete without a verified target retry")
