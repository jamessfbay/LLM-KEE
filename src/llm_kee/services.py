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
    EvaluationAggregator,
    EvidenceChecker,
    LLMJudge,
    RuleEngine,
)
from llm_kee.evolution import EvolutionService
from llm_kee.gates import LearningGate
from llm_kee.intent import IntentDetector, default_intent_patterns
from llm_kee.failures import FailureDetector
from llm_kee.improvements import ImprovementService
from llm_kee.integrations import KGClient, LLMKGClient
from llm_kee.loops import AgentImprovementLoop, KnowledgeEvolutionLoop, SkillSelectionLoop
from llm_kee.mape import MAPEAnalyzer, MAPEExecutor, MAPELoop, MAPEPlanner
from llm_kee.memory import MemoryDreamingService
from llm_kee.monitoring import MonitorService
from llm_kee.models import (
    ActionDefinition,
    ActionRun,
    AgentImprovementCycle,
    AggregatedEvaluation,
    ChangeSet,
    ConversationIntent,
    EvaluationResult,
    EvolutionEvent,
    FailureRecord,
    ImprovementProposal,
    KnowledgeEvolutionCycle,
    LearnedPattern,
    LearningDecisionType,
    LearningSignal,
    MAPEAnalysis,
    MAPEExecution,
    MAPEPlan,
    ProposalStatus,
    ReasoningTrace,
    ReviewStatus,
    SchemaSuggestion,
    SkillDefinition,
    SkillSelectionCycle,
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
        self.monitor = MonitorService(self.store)
        self.intent_detector = IntentDetector(self.store)
        self.failure_detector = FailureDetector(self.store)
        self.improvements = ImprovementService(self.store)
        self.mape_analyzer = MAPEAnalyzer()
        self.mape_planner = MAPEPlanner()
        self._ensure_defaults()
        self.evaluators = self._build_evaluators()
        self.aggregator = EvaluationAggregator()
        self.gate = LearningGate()
        self.safe_apply = SafeApplyService(self._build_kg_client())
        self.mape_executor = MAPEExecutor(
            self.store,
            self.action_runner,
            self.proposal_generator,
            self.run_evaluations,
            self.validate_artifact,
        )
        self.mape = MAPELoop(
            self.store,
            analyzer=self.mape_analyzer,
            planner=self.mape_planner,
            executor=self.mape_executor,
        )
        self.knowledge_loop = KnowledgeEvolutionLoop(self.store, self.run_mape_cycle)
        self.skill_loop = SkillSelectionLoop(
            self.store,
            self.intent_detector,
            self.plan_workflow,
            self.run_workflow,
        )
        self.agent_loop = AgentImprovementLoop(
            self.store,
            self.intent_detector,
            self.failure_detector,
            self.improvements,
        )
        self.memory = MemoryDreamingService(self.store, self.settings.workspace)

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
        if self.settings.intent.register_defaults:
            existing_pattern_ids = {pattern.id for pattern in self.store.intent_patterns.list()}
            for pattern in default_intent_patterns():
                if pattern.id not in existing_pattern_ids:
                    self.store.intent_patterns.upsert(pattern)

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
            LLMJudge(judge)
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

    def detect_intent(self, payload: dict[str, Any]) -> ConversationIntent:
        return self.intent_detector.detect_conversation(payload)

    def record_failure(self, payload: dict[str, Any]) -> FailureRecord:
        return self.failure_detector.record(payload)

    def propose_improvement(self, failure_id: str) -> ImprovementProposal:
        return self.improvements.propose_for_failure(failure_id)

    def approve_improvement(self, proposal_id: str) -> ImprovementProposal:
        return self.improvements.review(proposal_id, approve=True)

    def reject_improvement(self, proposal_id: str) -> ImprovementProposal:
        return self.improvements.review(proposal_id, approve=False)

    def monitor_path(self, path: Any) -> list[LearningSignal]:
        return self.monitor.scan(path)

    def analyze_signals(self, signals: list[LearningSignal]) -> MAPEAnalysis:
        analysis = self.mape_analyzer.analyze(signals)
        return self.store.mape_analyses.upsert(analysis)

    def plan_mape_actions(self, analysis: MAPEAnalysis) -> MAPEPlan:
        signals = [
            signal
            for signal in self.store.signals.list()
            if signal.id in set(analysis.signal_ids)
        ]
        plan = self.mape_planner.plan(analysis, signals)
        return self.store.mape_plans.upsert(plan)

    def execute_mape_plan(self, plan: MAPEPlan) -> MAPEExecution:
        return self.mape_executor.execute(plan)

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
        signals = [self.store.signals.upsert(signal) for signal in signal_batch]
        return self.mape.run(signals)

    def run_knowledge_evolution(self, signals: list[LearningSignal]) -> KnowledgeEvolutionCycle:
        persisted = [self.store.signals.upsert(signal) for signal in signals]
        return self.knowledge_loop.run(persisted)

    def run_skill_selection(self, task: dict[str, Any]) -> SkillSelectionCycle:
        return self.skill_loop.run(task)

    def run_agent_improvement(self, payload: dict[str, Any]) -> AgentImprovementCycle:
        return self.agent_loop.run(payload)

    def list_memory(self) -> dict[str, Any]:
        return self.memory.list_memory()

    def read_memory(self, scope: str, subject_id: str | None = None) -> dict[str, Any]:
        return self.memory.read_memory(scope, subject_id)

    def search_memory(self, query: str) -> dict[str, Any]:
        return self.memory.search(query)

    def draft_memory(self, payload: dict[str, Any]) -> object:
        return self.memory.draft(payload)

    def apply_memory_draft(self, draft_id: str) -> object:
        return self.memory.apply_draft(draft_id)

    def reject_memory_draft(self, draft_id: str) -> object:
        return self.memory.reject_draft(draft_id)

    def run_dream(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.memory.run_dream(payload)

    def dream_scheduler_status(self) -> dict[str, Any]:
        return self.memory.dream_scheduler_status()

    def run_dream_scheduler_once(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.memory.run_dream_scheduler_once(payload)

    def start_dream_scheduler(self, interval_minutes: int, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.memory.start_dream_scheduler(interval_minutes, payload)

    def dream_diary(self, dream_run_id: str) -> object:
        return self.memory.dream_diary(dream_run_id)

    def review_dream_proposal(self, proposal_id: str, approve: bool, notes: str | None = None) -> object:
        return self.memory.review_dream_proposal(proposal_id, approve=approve, notes=notes)

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
