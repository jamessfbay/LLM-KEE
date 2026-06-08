from typing import Any

from pydantic import Field

from llm_kee.models.base import StoredModel, new_id


class SourceRecord(StoredModel):
    id: str = Field(default_factory=lambda: new_id("src"))
    title: str
    uri: str | None = None
    source_type: str = "unknown"
    observed_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceNode(StoredModel):
    id: str = Field(default_factory=lambda: new_id("ev"))
    source_id: str
    quote: str
    citation: str | None = None
    claim_text: str | None = None
    source_type: str = "unknown"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    verification_status: str = "unverified"
    observed_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClaimNode(StoredModel):
    id: str = Field(default_factory=lambda: new_id("claim"))
    subject: str
    predicate: str
    object: str
    qualifier: dict[str, Any] = Field(default_factory=dict)
    status: str = "active"
    evidence_ids: list[str] = Field(default_factory=list)
    conflict_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class EntityNode(StoredModel):
    id: str = Field(default_factory=lambda: new_id("ent"))
    entity_type: str
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)


class RelationEdge(StoredModel):
    id: str = Field(default_factory=lambda: new_id("rel"))
    subject_id: str
    predicate: str
    object_id: str
    claim_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    valid_from: str | None = None
    valid_to: str | None = None


class SkillDefinition(StoredModel):
    id: str = Field(default_factory=lambda: new_id("skill"))
    name: str
    description: str
    use_cases: list[str] = Field(default_factory=list)
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
    evaluation_metrics: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    enabled: bool = True


class SkillPlan(StoredModel):
    id: str = Field(default_factory=lambda: new_id("skill_plan"))
    task_type: str
    skill_ids: list[str] = Field(default_factory=list)
    rationale: str
    input_payload: dict[str, Any] = Field(default_factory=dict)


class WorkflowStep(StoredModel):
    id: str = Field(default_factory=lambda: new_id("step"))
    order: int
    skill_id: str
    name: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    status: str = "pending"
    output: dict[str, Any] | None = None
    error: str | None = None


class WorkflowDefinition(StoredModel):
    id: str = Field(default_factory=lambda: new_id("workflow"))
    task_type: str
    skill_sequence: list[str] = Field(default_factory=list)
    constraints: dict[str, Any] = Field(default_factory=dict)
    input_payload: dict[str, Any] = Field(default_factory=dict)


class WorkflowRun(StoredModel):
    id: str = Field(default_factory=lambda: new_id("run"))
    workflow_id: str
    task_type: str
    steps: list[WorkflowStep] = Field(default_factory=list)
    status: str = "pending"
    output: dict[str, Any] = Field(default_factory=dict)


class ActionDefinition(StoredModel):
    id: str = Field(default_factory=lambda: new_id("action"))
    action_type: str
    name: str
    description: str
    required_skills: list[str] = Field(default_factory=list)
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


class ActionArtifact(StoredModel):
    id: str = Field(default_factory=lambda: new_id("artifact"))
    action_run_id: str
    artifact_type: str
    title: str
    content: dict[str, Any] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ActionRun(StoredModel):
    id: str = Field(default_factory=lambda: new_id("action_run"))
    action_type: str
    input_payload: dict[str, Any] = Field(default_factory=dict)
    workflow_run_id: str | None = None
    artifact_ids: list[str] = Field(default_factory=list)
    status: str = "pending"
    output: dict[str, Any] = Field(default_factory=dict)


class MetricResult(StoredModel):
    id: str = Field(default_factory=lambda: new_id("metric"))
    name: str
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    details: dict[str, Any] = Field(default_factory=dict)


class ValidationResult(StoredModel):
    id: str = Field(default_factory=lambda: new_id("validation"))
    target_type: str
    target_id: str
    valid: bool
    metrics: list[MetricResult] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)


class EvaluationRecord(StoredModel):
    id: str = Field(default_factory=lambda: new_id("eval_record"))
    target_type: str
    target_id: str
    evaluator_name: str
    metrics: list[MetricResult] = Field(default_factory=list)
    decision: str = "review"
    notes: str | None = None


class ChangeSet(StoredModel):
    id: str = Field(default_factory=lambda: new_id("changeset"))
    target_type: str
    target_id: str
    operation: str
    before: dict[str, Any] | None = None
    after: dict[str, Any] = Field(default_factory=dict)
    reason: str
    evidence_ids: list[str] = Field(default_factory=list)
    proposal_id: str | None = None


class KnowledgeVersion(StoredModel):
    id: str = Field(default_factory=lambda: new_id("version"))
    target_type: str
    target_id: str
    version: int
    valid_from: str | None = None
    valid_to: str | None = None
    status: str = "active"
    snapshot: dict[str, Any] = Field(default_factory=dict)
    reason: str
    supersedes_version_id: str | None = None


class EvolutionEvent(StoredModel):
    id: str = Field(default_factory=lambda: new_id("evo"))
    event_type: str
    target_type: str
    target_id: str
    change_set_id: str
    from_version_id: str | None = None
    to_version_id: str | None = None
    reason: str
    evidence_ids: list[str] = Field(default_factory=list)


class MAPEObservation(StoredModel):
    id: str = Field(default_factory=lambda: new_id("obs"))
    signal_ids: list[str] = Field(default_factory=list)
    summary: str
    payload: dict[str, Any] = Field(default_factory=dict)


class MonitorEvent(StoredModel):
    id: str = Field(default_factory=lambda: new_id("mon_event"))
    root_path: str
    path: str
    event_type: str
    old_hash: str | None = None
    new_hash: str | None = None
    old_mtime: float | None = None
    new_mtime: float | None = None
    signal_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MonitorSnapshot(StoredModel):
    id: str = Field(default_factory=lambda: new_id("mon_snapshot"))
    root_path: str
    files: dict[str, dict[str, Any]] = Field(default_factory=dict)
    event_ids: list[str] = Field(default_factory=list)


class MAPEAnalysis(StoredModel):
    id: str = Field(default_factory=lambda: new_id("mape_analysis"))
    signal_ids: list[str] = Field(default_factory=list)
    signal_count: int = 0
    high_priority_count: int = 0
    signal_types: list[str] = Field(default_factory=list)
    impact_level: str = "low"
    recommended_actions: list[str] = Field(default_factory=list)
    requires_review: bool = False
    rationale: str = ""


class MAPEPlan(StoredModel):
    id: str = Field(default_factory=lambda: new_id("mape_plan"))
    analysis_id: str
    steps: list[dict[str, Any]] = Field(default_factory=list)
    requires_review: bool = False


class MAPEExecution(StoredModel):
    id: str = Field(default_factory=lambda: new_id("mape_exec"))
    plan_id: str
    status: str = "completed"
    action_run_ids: list[str] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)
    proposal_ids: list[str] = Field(default_factory=list)
    evaluation_ids: list[str] = Field(default_factory=list)
    decision_ids: list[str] = Field(default_factory=list)
    evolution_event_ids: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class MAPECycle(StoredModel):
    id: str = Field(default_factory=lambda: new_id("mape"))
    observation_ids: list[str] = Field(default_factory=list)
    analysis: dict[str, Any] = Field(default_factory=dict)
    plan: dict[str, Any] = Field(default_factory=dict)
    execution: dict[str, Any] = Field(default_factory=dict)
    learned: dict[str, Any] = Field(default_factory=dict)
    action_run_ids: list[str] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)
    proposal_ids: list[str] = Field(default_factory=list)
    evaluation_ids: list[str] = Field(default_factory=list)
    decision_ids: list[str] = Field(default_factory=list)
    evolution_event_ids: list[str] = Field(default_factory=list)
    status: str = "completed"


class IntentPattern(StoredModel):
    id: str = Field(default_factory=lambda: new_id("intent_pattern"))
    intent_type: str
    description: str
    trigger_terms: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    default_skill_ids: list[str] = Field(default_factory=list)
    enabled: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskIntent(StoredModel):
    id: str = Field(default_factory=lambda: new_id("task_intent"))
    intent_type: str
    task_type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    matched_pattern_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    success_criteria: list[str] = Field(default_factory=list)


class ConversationIntent(StoredModel):
    id: str = Field(default_factory=lambda: new_id("conv_intent"))
    intent_type: str
    utterance: str | None = None
    task_type: str = "general_knowledge_task"
    payload: dict[str, Any] = Field(default_factory=dict)
    matched_pattern_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    success_criteria: list[str] = Field(default_factory=list)


class FailureRecord(StoredModel):
    id: str = Field(default_factory=lambda: new_id("failure"))
    failure_type: str
    summary: str
    severity: str = "medium"
    intent_id: str | None = None
    target_type: str | None = None
    target_id: str | None = None
    artifact_id: str | None = None
    proposal_id: str | None = None
    workflow_run_id: str | None = None
    action_run_id: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)
    status: str = "open"


class FailureCluster(StoredModel):
    id: str = Field(default_factory=lambda: new_id("failure_cluster"))
    failure_type: str
    failure_ids: list[str] = Field(default_factory=list)
    summary: str
    suggested_improvement_type: str | None = None
    status: str = "open"


class ImprovementAction(StoredModel):
    id: str = Field(default_factory=lambda: new_id("improvement_action"))
    action_type: str
    target_type: str | None = None
    target_id: str | None = None
    description: str
    payload: dict[str, Any] = Field(default_factory=dict)


class ImprovementProposal(StoredModel):
    id: str = Field(default_factory=lambda: new_id("improvement"))
    improvement_type: str
    title: str
    description: str
    failure_ids: list[str] = Field(default_factory=list)
    intent_ids: list[str] = Field(default_factory=list)
    actions: list[ImprovementAction] = Field(default_factory=list)
    status: str = "pending_review"
    rationale: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ImprovementReview(StoredModel):
    id: str = Field(default_factory=lambda: new_id("improvement_review"))
    improvement_id: str
    decision: str
    reviewer: str = "human"
    notes: str | None = None


class KnowledgeEvolutionCycle(StoredModel):
    id: str = Field(default_factory=lambda: new_id("knowledge_loop"))
    signal_ids: list[str] = Field(default_factory=list)
    mape_cycle_id: str | None = None
    proposal_ids: list[str] = Field(default_factory=list)
    evaluation_ids: list[str] = Field(default_factory=list)
    decision_ids: list[str] = Field(default_factory=list)
    evolution_event_ids: list[str] = Field(default_factory=list)
    status: str = "completed"


class SkillSelectionCycle(StoredModel):
    id: str = Field(default_factory=lambda: new_id("skill_loop"))
    task_intent_id: str
    skill_plan_id: str | None = None
    workflow_id: str | None = None
    workflow_run_id: str | None = None
    evaluation_record_ids: list[str] = Field(default_factory=list)
    improvement_proposal_ids: list[str] = Field(default_factory=list)
    status: str = "completed"


class AgentImprovementCycle(StoredModel):
    id: str = Field(default_factory=lambda: new_id("agent_loop"))
    conversation_intent_id: str | None = None
    failure_ids: list[str] = Field(default_factory=list)
    improvement_proposal_ids: list[str] = Field(default_factory=list)
    review_ids: list[str] = Field(default_factory=list)
    status: str = "pending_review"
