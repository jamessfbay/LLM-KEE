from typing import Any

from llm_kee.models import SkillPlan, WorkflowDefinition, WorkflowRun, WorkflowStep
from llm_kee.storage import KEEStore


class WorkflowPlanner:
    def __init__(self, store: KEEStore) -> None:
        self.store = store

    def plan(self, skill_plan: SkillPlan) -> WorkflowDefinition:
        workflow = WorkflowDefinition(
            task_type=skill_plan.task_type,
            skill_sequence=skill_plan.skill_ids,
            input_payload=skill_plan.input_payload,
            constraints={"source": "skill_plan", "skill_plan_id": skill_plan.id},
        )
        return self.store.workflow_definitions.upsert(workflow)


class WorkflowExecutor:
    def __init__(self, store: KEEStore) -> None:
        self.store = store

    def run(self, workflow: WorkflowDefinition) -> WorkflowRun:
        steps: list[WorkflowStep] = []
        accumulated: dict[str, Any] = {"input": workflow.input_payload, "skill_outputs": []}
        for index, skill_id in enumerate(workflow.skill_sequence, start=1):
            step = WorkflowStep(
                order=index,
                skill_id=skill_id,
                name=skill_id.replace("_", " ").title(),
                inputs={"task_type": workflow.task_type, "payload": workflow.input_payload},
                status="completed",
                output={
                    "skill_id": skill_id,
                    "summary": f"Executed {skill_id} for {workflow.task_type}.",
                    "evidence_ids": workflow.input_payload.get("evidence_ids", []),
                },
            )
            steps.append(step)
            accumulated["skill_outputs"].append(step.output)
        run = WorkflowRun(
            workflow_id=workflow.id,
            task_type=workflow.task_type,
            steps=steps,
            status="completed",
            output=accumulated,
        )
        return self.store.workflow_runs.upsert(run)
