from pathlib import Path

from llm_kee.models import (
    ActionArtifact,
    ActionDefinition,
    ActionRun,
    AggregatedEvaluation,
    ChangeSet,
    ClaimNode,
    EntityNode,
    EvaluationRecord,
    EvaluationResult,
    EvidenceNode,
    EvolutionEvent,
    KnowledgeVersion,
    LearnedPattern,
    LearningDecision,
    LearningSignal,
    MAPECycle,
    MAPEAnalysis,
    MAPEExecution,
    MAPEObservation,
    MAPEPlan,
    MonitorEvent,
    MonitorSnapshot,
    RelationEdge,
    ReasoningTrace,
    SchemaSuggestion,
    SkillDefinition,
    SkillPlan,
    SourceRecord,
    UpdateProposal,
    UserFeedback,
    ValidationResult,
    WorkflowDefinition,
    WorkflowRun,
)
from llm_kee.storage.json_store import JsonRepository


class KEEStore:
    def __init__(self, workspace: Path) -> None:
        root = workspace / ".llm_kee"
        self.feedback = JsonRepository(root / "feedback.json", UserFeedback)
        self.traces = JsonRepository(root / "traces.json", ReasoningTrace)
        self.signals = JsonRepository(root / "signals.json", LearningSignal)
        self.proposals = JsonRepository(root / "proposals.json", UpdateProposal)
        self.evaluations = JsonRepository(root / "evaluations.json", EvaluationResult)
        self.aggregates = JsonRepository(root / "aggregates.json", AggregatedEvaluation)
        self.decisions = JsonRepository(root / "decisions.json", LearningDecision)
        self.patterns = JsonRepository(root / "patterns.json", LearnedPattern)
        self.schema_suggestions = JsonRepository(root / "schema_suggestions.json", SchemaSuggestion)
        self.sources = JsonRepository(root / "sources.json", SourceRecord)
        self.evidence_nodes = JsonRepository(root / "evidence_nodes.json", EvidenceNode)
        self.claim_nodes = JsonRepository(root / "claim_nodes.json", ClaimNode)
        self.entity_nodes = JsonRepository(root / "entity_nodes.json", EntityNode)
        self.relation_edges = JsonRepository(root / "relation_edges.json", RelationEdge)
        self.skills = JsonRepository(root / "skills.json", SkillDefinition)
        self.skill_plans = JsonRepository(root / "skill_plans.json", SkillPlan)
        self.workflow_definitions = JsonRepository(root / "workflow_definitions.json", WorkflowDefinition)
        self.workflow_runs = JsonRepository(root / "workflow_runs.json", WorkflowRun)
        self.action_definitions = JsonRepository(root / "action_definitions.json", ActionDefinition)
        self.action_runs = JsonRepository(root / "action_runs.json", ActionRun)
        self.action_artifacts = JsonRepository(root / "action_artifacts.json", ActionArtifact)
        self.evaluation_records = JsonRepository(root / "evaluation_records.json", EvaluationRecord)
        self.validation_results = JsonRepository(root / "validation_results.json", ValidationResult)
        self.change_sets = JsonRepository(root / "change_sets.json", ChangeSet)
        self.knowledge_versions = JsonRepository(root / "knowledge_versions.json", KnowledgeVersion)
        self.evolution_events = JsonRepository(root / "evolution_events.json", EvolutionEvent)
        self.mape_observations = JsonRepository(root / "mape_observations.json", MAPEObservation)
        self.monitor_snapshots = JsonRepository(root / "monitor_snapshots.json", MonitorSnapshot)
        self.monitor_events = JsonRepository(root / "monitor_events.json", MonitorEvent)
        self.mape_analyses = JsonRepository(root / "mape_analyses.json", MAPEAnalysis)
        self.mape_plans = JsonRepository(root / "mape_plans.json", MAPEPlan)
        self.mape_executions = JsonRepository(root / "mape_executions.json", MAPEExecution)
        self.mape_cycles = JsonRepository(root / "mape_cycles.json", MAPECycle)
