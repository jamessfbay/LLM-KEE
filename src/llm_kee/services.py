from typing import Any

from llm_kee.actions import ActionRegistry, default_actions
from llm_kee.actions.runner import ActionRunner
from llm_kee.agents.feedback_interpreter import FeedbackInterpreter
from llm_kee.agents.proposal_generator import ProposalGenerator
from llm_kee.apply import SafeApplyService
from llm_kee.config import Settings
from llm_kee.evaluation import (
    BehaviorSignalEvaluator,
    ConflictChecker,
    DeterministicLLMJudge,
    EvaluationAggregator,
    EvidenceChecker,
    RuleEngine,
)
from llm_kee.evolution import EvolutionService
from llm_kee.gates import LearningGate
from llm_kee.integrations import KGClient, LLMKGClient
from llm_kee.mape import MAPELoop
from llm_kee.models import (
    ActionDefinition,
    ActionRun,
    AggregatedEvaluation,
    ChangeSet,
    EvaluationResult,
    EvolutionEvent,
    LearnedPattern,
    LearningDecisionType,
    LearningSignal,
    ProposalStatus,
    ReasoningTrace,
    ReviewStatus,
    SchemaSuggestion,
    SkillDefinition,
    SkillPlan,
    UpdateProposal,
    ValidationResult,
    WorkflowDefinition,
    WorkflowRun,
    UserFeedback,
)
from llm_kee.models.base import now_utc
from llm_kee.skills import SkillRegistry, SkillRetriever, TaskClassifier, default_skills
from llm_kee.storage import KEEStore
from llm_kee.workflows import WorkflowExecutor, WorkflowPlanner


class KEEEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.store = KEEStore(settings.workspace)
        self.feedback_interpreter = FeedbackInterpreter()
        self.proposal_generator = ProposalGenerator()
        self.skill_registry = SkillRegistry(self.store)
        self.action_registry = ActionRegistry(self.store)
        self.task_classifier = TaskClassifier()
        self.skill_retriever = SkillRetriever(self.skill_registry)
        self.workflow_planner = WorkflowPlanner(self.store)
        self.workflow_executor = WorkflowExecutor(self.store)
        self.action_runner = ActionRunner(
            self.store,
            self.action_registry,
            self.task_classifier,
            self.skill_retriever,
            self.workflow_planner,
            self.workflow_executor,
        )
        self.evolution = EvolutionService(self.store)
        self.mape = MAPELoop(self.store)
        self._ensure_defaults()
        self.evaluators = self._build_evaluators()
        self.aggregator = EvaluationAggregator()
        self.gate = LearningGate()
        self.safe_apply = SafeApplyService(self._build_kg_client())

    def _ensure_defaults(self) -> None:
        if self.settings.skills.register_defaults:
            existing_skill_ids = {skill.id for skill in self.store.skills.list()}
            for skill in default_skills():
                if skill.id not in existing_skill_ids:
                    self.store.skills.upsert(skill)
        if self.settings.actions.register_defaults:
            existing_action_ids = {action.id for action in self.store.action_definitions.list()}
            for action in default_actions():
                if action.id not in existing_action_ids:
                    self.store.action_definitions.upsert(action)

    def _build_kg_client(self) -> KGClient:
        if self.settings.kg.enable_direct_adapter and self.settings.kg.workspace:
            return LLMKGClient(self.settings.kg.workspace, self.settings.kg.project_path)
        return KGClient()

    def _build_evaluators(self) -> list:
        config = self.settings.evaluation
        evaluators = []
        if config.enable_rule_engine:
            evaluators.append(RuleEngine())
        if config.enable_evidence_checker:
            evaluators.append(EvidenceChecker())
        if config.enable_conflict_checker:
            evaluators.append(ConflictChecker())
        if config.enable_behavior_signal:
            evaluators.append(BehaviorSignalEvaluator())
        evaluators.extend(
            DeterministicLLMJudge(judge)
            for judge in config.judges
            if judge.enabled
        )
        return evaluators

    def accept_feedback(self, feedback: UserFeedback) -> tuple[UserFeedback, UpdateProposal]:
        feedback.status = "interpreted"
        feedback.updated_at = now_utc()
        self.store.feedback.upsert(feedback)

        signal = self.feedback_interpreter.interpret(feedback)
        self.store.signals.upsert(signal)

        proposal = self.proposal_generator.from_signal(signal)
        self.store.proposals.upsert(proposal)
        feedback.status = "proposed"
        feedback.updated_at = now_utc()
        self.store.feedback.upsert(feedback)
        return feedback, proposal

    def register_skill(self, skill: SkillDefinition) -> SkillDefinition:
        return self.skill_registry.register(skill)

    def register_action(self, action: ActionDefinition) -> ActionDefinition:
        return self.action_registry.register(action)

    def classify_task(self, task: dict[str, Any]) -> str:
        return self.task_classifier.classify(task)

    def plan_workflow(self, task: dict[str, Any]) -> WorkflowDefinition:
        task_type = self.classify_task(task)
        skill_plan = self.skill_retriever.retrieve(task_type, task)
        self.store.skill_plans.upsert(skill_plan)
        return self.workflow_planner.plan(skill_plan)

    def run_workflow(self, workflow: WorkflowDefinition) -> WorkflowRun:
        return self.workflow_executor.run(workflow)

    def run_action(self, action_type: str, input_payload: dict[str, Any]) -> ActionRun:
        return self.action_runner.run(action_type, input_payload)

    def validate_artifact(self, artifact_id: str) -> ValidationResult | None:
        artifact = self.store.action_artifacts.get(artifact_id)
        if not artifact:
            return None
        has_evidence = bool(artifact.evidence_ids)
        validation = ValidationResult(
            target_type="action_artifact",
            target_id=artifact.id,
            valid=has_evidence,
            issues=[] if has_evidence else ["Artifact has no evidence IDs."],
        )
        return self.store.validation_results.upsert(validation)

    def create_evolution_event(self, change_set: ChangeSet) -> EvolutionEvent:
        return self.evolution.create_event(change_set)

    def run_mape_cycle(self, signal_batch: list[LearningSignal]) -> object:
        return self.mape.run(signal_batch)

    def save_trace(self, trace: ReasoningTrace) -> ReasoningTrace:
        self.store.traces.upsert(trace)
        if trace.reusable:
            pattern = LearnedPattern(
                pattern_type="reasoning",
                name=f"Reusable trace: {trace.question[:60]}",
                description=trace.final_answer,
                examples=[step.description for step in trace.reasoning_steps],
                source_trace_ids=[trace.id],
            )
            self.store.patterns.upsert(pattern)
        return trace

    def mark_trace_reusable(self, trace_id: str) -> ReasoningTrace | None:
        trace = self.store.traces.get(trace_id)
        if not trace:
            return None
        trace.reusable = True
        trace.updated_at = now_utc()
        return self.save_trace(trace)

    def run_evaluations(self, proposal: UpdateProposal) -> tuple[list[EvaluationResult], AggregatedEvaluation]:
        results = [evaluator.evaluate(proposal) for evaluator in self.evaluators]
        for result in results:
            self.store.evaluations.upsert(result)
        aggregate = self.aggregator.aggregate(proposal.id, results)
        self.store.aggregates.upsert(aggregate)

        decision = self.gate.decide(aggregate)
        self.store.decisions.upsert(decision)
        proposal.status = self._status_from_decision(decision.decision)
        proposal.updated_at = now_utc()
        self.store.proposals.upsert(proposal)
        return results, aggregate

    def approve_proposal(self, proposal: UpdateProposal) -> UpdateProposal:
        proposal.status = ProposalStatus.APPROVED
        proposal.updated_at = now_utc()
        return self.store.proposals.upsert(proposal)

    def reject_proposal(self, proposal: UpdateProposal) -> UpdateProposal:
        proposal.status = ProposalStatus.REJECTED
        proposal.updated_at = now_utc()
        return self.store.proposals.upsert(proposal)

    def request_evidence(self, proposal: UpdateProposal) -> UpdateProposal:
        proposal.status = ProposalStatus.NEED_MORE_EVIDENCE
        proposal.updated_at = now_utc()
        return self.store.proposals.upsert(proposal)

    def approve_pattern(self, pattern: LearnedPattern) -> LearnedPattern:
        pattern.status = ReviewStatus.APPROVED
        pattern.updated_at = now_utc()
        return self.store.patterns.upsert(pattern)

    def retire_pattern(self, pattern: LearnedPattern) -> LearnedPattern:
        pattern.status = ReviewStatus.RETIRED
        pattern.updated_at = now_utc()
        return self.store.patterns.upsert(pattern)

    def approve_schema_suggestion(self, suggestion: SchemaSuggestion) -> SchemaSuggestion:
        suggestion.status = ReviewStatus.APPROVED
        suggestion.updated_at = now_utc()
        return self.store.schema_suggestions.upsert(suggestion)

    def reject_schema_suggestion(self, suggestion: SchemaSuggestion) -> SchemaSuggestion:
        suggestion.status = ReviewStatus.REJECTED
        suggestion.updated_at = now_utc()
        return self.store.schema_suggestions.upsert(suggestion)

    def _status_from_decision(self, decision: str) -> ProposalStatus:
        if decision == LearningDecisionType.AUTO_APPLY:
            return ProposalStatus.APPROVED
        if decision == LearningDecisionType.NEED_MORE_EVIDENCE:
            return ProposalStatus.NEED_MORE_EVIDENCE
        if decision == LearningDecisionType.CONFLICT_REVIEW:
            return ProposalStatus.CONFLICT_REVIEW
        if decision == LearningDecisionType.REJECT:
            return ProposalStatus.REJECTED
        return ProposalStatus.PENDING_REVIEW
