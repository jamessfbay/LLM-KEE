from collections.abc import Callable

from llm_kee.actions.runner import ActionRunner
from llm_kee.agents.proposal_generator import ProposalGenerator
from llm_kee.models import MAPEExecution, MAPEPlan, UpdateProposal, ValidationResult, WorkflowDefinition
from llm_kee.storage import KEEStore


EvaluationRunner = Callable[[UpdateProposal], tuple[list, object]]


class MAPEExecutor:
    def __init__(
        self,
        store: KEEStore,
        action_runner: ActionRunner,
        proposal_generator: ProposalGenerator,
        evaluation_runner: EvaluationRunner,
        artifact_validator: Callable[[str], ValidationResult | None],
    ) -> None:
        self.store = store
        self.action_runner = action_runner
        self.proposal_generator = proposal_generator
        self.evaluation_runner = evaluation_runner
        self.artifact_validator = artifact_validator

    def execute(self, plan: MAPEPlan) -> MAPEExecution:
        execution = MAPEExecution(plan_id=plan.id, status="completed")
        for step in plan.steps:
            if step.get("type") == "workflow":
                workflow = WorkflowDefinition(
                    task_type=(step.get("input_payload") or {}).get("task_type", "workflow"),
                    skill_sequence=step.get("skill_ids") or [],
                    input_payload=step.get("input_payload") or {},
                    constraints={"source": "mape_plan", "mape_plan_id": plan.id},
                )
                workflow = self.store.workflow_definitions.upsert(workflow)
                run = self.action_runner.executor.run(workflow)
                execution.action_run_ids.append(run.id)
                continue
            if step.get("type") != "action":
                execution.errors.append(f"Unsupported MAPE step type: {step.get('type')}")
                continue
            try:
                run = self.action_runner.run(step["action_type"], step.get("input_payload") or {})
                execution.action_run_ids.append(run.id)
                for artifact_id in run.artifact_ids:
                    execution.artifact_ids.append(artifact_id)
                    self.artifact_validator(artifact_id)
                    artifact = self.store.action_artifacts.get(artifact_id)
                    if artifact:
                        proposal = self.proposal_generator.from_action_artifact(artifact)
                        proposal = self.store.proposals.upsert(proposal)
                        execution.proposal_ids.append(proposal.id)
                        results, _ = self.evaluation_runner(proposal)
                        execution.evaluation_ids.extend(result.id for result in results)
                        decisions = [
                            decision
                            for decision in self.store.decisions.list()
                            if decision.proposal_id == proposal.id
                        ]
                        execution.decision_ids.extend(decision.id for decision in decisions)
            except Exception as exc:  # pragma: no cover - defensive guard for workflow integrations
                execution.status = "completed_with_errors"
                execution.errors.append(str(exc))
        return self.store.mape_executions.upsert(execution)
