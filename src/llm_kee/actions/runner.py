from typing import Any

from llm_kee.actions import ActionRegistry
from llm_kee.models import ActionArtifact, ActionRun, SkillPlan
from llm_kee.skills import SkillRetriever, TaskClassifier
from llm_kee.storage import KEEStore
from llm_kee.workflows import WorkflowExecutor, WorkflowPlanner


class ActionRunner:
    def __init__(
        self,
        store: KEEStore,
        registry: ActionRegistry,
        classifier: TaskClassifier,
        retriever: SkillRetriever,
        planner: WorkflowPlanner,
        executor: WorkflowExecutor,
    ) -> None:
        self.store = store
        self.registry = registry
        self.classifier = classifier
        self.retriever = retriever
        self.planner = planner
        self.executor = executor

    def run(self, action_type: str, input_payload: dict[str, Any]) -> ActionRun:
        action = self.registry.get_by_type(action_type)
        if not action or not action.enabled:
            run = ActionRun(action_type=action_type, input_payload=input_payload, status="failed", output={"error": "action not found or disabled"})
            return self.store.action_runs.upsert(run)

        task_type = self._task_type_for_action(action_type, input_payload)
        skill_plan = SkillPlan(
            task_type=task_type,
            skill_ids=[skill_id for skill_id in action.required_skills if self.store.skills.get(skill_id)],
            rationale=f"Action {action_type} requires its registered skills.",
            input_payload=input_payload,
        )
        if not skill_plan.skill_ids:
            task_type = self.classifier.classify(input_payload)
            skill_plan = self.retriever.retrieve(task_type, input_payload)
        self.store.skill_plans.upsert(skill_plan)

        workflow = self.planner.plan(skill_plan)
        workflow_run = self.executor.run(workflow)
        artifact = ActionArtifact(
            action_run_id="pending",
            artifact_type=action_type,
            title=action.name,
            content={
                "action_type": action_type,
                "task_type": task_type,
                "workflow_run_id": workflow_run.id,
                "summary": f"{action.name} completed with {len(workflow_run.steps)} workflow steps.",
                "workflow_output": workflow_run.output,
            },
            evidence_ids=input_payload.get("evidence_ids", []),
            confidence=0.75 if input_payload.get("evidence_ids") else 0.5,
        )
        run = ActionRun(
            action_type=action_type,
            input_payload=input_payload,
            workflow_run_id=workflow_run.id,
            status="completed",
            output={"artifact_title": artifact.title, "workflow_run_id": workflow_run.id},
        )
        run = self.store.action_runs.upsert(run)
        artifact.action_run_id = run.id
        artifact = self.store.action_artifacts.upsert(artifact)
        run.artifact_ids = [artifact.id]
        run.output["artifact_id"] = artifact.id
        return self.store.action_runs.upsert(run)

    def _task_type_for_action(self, action_type: str, input_payload: dict[str, Any]) -> str:
        if action_type == "rebuild_timeline":
            return "timeline_reconstruction"
        if action_type == "detect_missing_or_conflicting_information":
            return "missing_or_conflict_detection"
        return str(input_payload.get("task_type") or "intelligence_pack")
