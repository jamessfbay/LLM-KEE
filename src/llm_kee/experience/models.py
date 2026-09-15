from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from llm_kee.experience.hashing import canonical_hash, model_digest


HASH_PATTERN = r"^[A-Fa-f0-9]{64}$"
IMAGE_DIGEST_PATTERN = r"^(?:sha256:)?[A-Fa-f0-9]{64}$"


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def now_rfc3339() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)


class ActorHandoffMode(StrEnum):
    SAME_SESSION = "same_session"
    RECONSTRUCTED = "reconstructed"


class TaskVerdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"


class ExperienceKind(StrEnum):
    PROCEDURE = "procedure"
    CONSTRAINT = "constraint"
    FAILURE_LESSON = "failure_lesson"
    ROUTING_HINT = "routing_hint"
    BOUNDARY_CONDITION = "boundary_condition"


class CausalStatus(StrEnum):
    OBSERVED_ASSOCIATION = "observed_association"
    MECHANISM_HYPOTHESIS = "mechanism_hypothesis"
    INTERVENTION_SUPPORTED = "intervention_supported"


class ExperienceCandidateStatus(StrEnum):
    PROPOSED = "proposed"
    RECONCILED = "reconciled"
    REJECTED = "rejected"


class MemoryRevisionOperation(StrEnum):
    ADD = "add"
    QUALIFY = "qualify"
    SUPERSEDE = "supersede"
    RETRACT = "retract"
    NO_CHANGE = "no_change"


class ExperienceEvaluationDecision(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    NEEDS_EXPERIMENT = "needs_experiment"


class LearningStage(StrEnum):
    BROAD = "broad"
    DEEP = "deep"


class LearningSessionStatus(StrEnum):
    PLANNED = "planned"
    RUNNING = "running"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    FAILED = "failed"


class MemoryArtifactStatus(StrEnum):
    CANDIDATE = "candidate"
    REPLAY = "replay"
    SHADOW = "shadow"
    CANARY10 = "canary10"
    CANARY50 = "canary50"
    ACTIVE = "active"
    ROLLED_BACK = "rolled_back"
    REJECTED = "rejected"


class ActorBinding(ContractModel):
    provider: str = Field(min_length=1, max_length=2048)
    model: str = Field(min_length=1, max_length=2048)
    prompt_hash: str = Field(pattern=HASH_PATTERN)
    toolset_hash: str = Field(pattern=HASH_PATTERN)
    protocol_schema_hash: str = Field(pattern=HASH_PATTERN)


class ActorLearningHandoff(ContractModel):
    id: str = Field(min_length=1, max_length=2048)
    tenant_id: str = Field(min_length=1, max_length=2048)
    domain: str = Field(min_length=1, max_length=2048)
    subject_id: str = Field(min_length=1, max_length=2048)
    decision_id: str = Field(min_length=1, max_length=2048)
    episode_id: str = Field(min_length=1, max_length=2048)
    source_run_id: str = Field(min_length=1, max_length=2048)
    actor_binding: ActorBinding
    handoff_mode: ActorHandoffMode
    verdict: TaskVerdict
    action_refs: list[str] = Field(min_length=1)
    effect_proof_refs: list[str] = Field(min_length=1)
    verifier_report_hash: str = Field(pattern=HASH_PATTERN)
    outcome_hash: str = Field(pattern=HASH_PATTERN)
    evidence_refs: list[str] = Field(min_length=1)
    conclusions: list[str] = Field(min_length=1)
    uncertainties: list[str] = Field(default_factory=list)
    parent_memory_hash: str = Field(pattern=HASH_PATTERN)
    promotion_eligible: bool
    created_at: str
    handoff_hash: str = Field(pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_integrity(self) -> "ActorLearningHandoff":
        expected_eligible = self.handoff_mode == ActorHandoffMode.SAME_SESSION
        if self.promotion_eligible != expected_eligible:
            raise ValueError("only a same-session actor handoff can be promotion eligible")
        for name in ("action_refs", "effect_proof_refs", "evidence_refs", "conclusions"):
            values = getattr(self, name)
            if any(not item.strip() for item in values) or len(values) != len(set(values)):
                raise ValueError(f"{name} must contain unique non-empty values")
        datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        if self.handoff_hash != model_digest(self, "handoff_hash"):
            raise ValueError("actor handoff digest mismatch")
        return self


class ExperienceCandidate(ContractModel):
    id: str = Field(min_length=1, max_length=2048)
    handoff_id: str = Field(min_length=1, max_length=2048)
    tenant_id: str = Field(min_length=1, max_length=2048)
    domain: str = Field(min_length=1, max_length=2048)
    kind: ExperienceKind
    causal_status: CausalStatus
    condition: str = Field(min_length=1, max_length=2048)
    action: str = Field(min_length=1, max_length=2048)
    expected_outcome: str = Field(min_length=1, max_length=2048)
    mechanism_hypothesis: str | None = Field(default=None, min_length=1, max_length=2048)
    evidence_refs: list[str] = Field(min_length=1)
    counterexample_refs: list[str] = Field(default_factory=list)
    applicability: dict[str, Any]
    invalidation_conditions: list[str] = Field(default_factory=list)
    uncertainty: float = Field(ge=0.0, le=1.0)
    parent_memory_hash: str = Field(pattern=HASH_PATTERN)
    status: ExperienceCandidateStatus = ExperienceCandidateStatus.PROPOSED
    candidate_hash: str = Field(pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_integrity(self) -> "ExperienceCandidate":
        needs_mechanism = self.causal_status != CausalStatus.OBSERVED_ASSOCIATION
        if needs_mechanism != bool(self.mechanism_hypothesis and self.mechanism_hypothesis.strip()):
            raise ValueError("causal claims beyond association require a mechanism hypothesis")
        if self.candidate_hash != model_digest(self, "candidate_hash"):
            raise ValueError("experience candidate digest mismatch")
        if len(self.evidence_refs) != len(set(self.evidence_refs)):
            raise ValueError("experience evidence references must be unique")
        return self


class MemoryRevisionProposal(ContractModel):
    id: str = Field(min_length=1, max_length=2048)
    tenant_id: str = Field(min_length=1, max_length=2048)
    domain: str = Field(min_length=1, max_length=2048)
    memory_binding: str = Field(min_length=1, max_length=2048)
    parent_memory_hash: str = Field(pattern=HASH_PATTERN)
    candidate_ids: list[str]
    operation: MemoryRevisionOperation
    target_experience_id: str | None = None
    conflicts_with: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1, max_length=2048)
    proposed_snapshot_hash: str = Field(pattern=HASH_PATTERN)
    revision_hash: str = Field(pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_integrity(self) -> "MemoryRevisionProposal":
        target_required = self.operation in {
            MemoryRevisionOperation.QUALIFY,
            MemoryRevisionOperation.SUPERSEDE,
            MemoryRevisionOperation.RETRACT,
        }
        if target_required != bool(self.target_experience_id):
            raise ValueError("revision target does not match operation")
        if self.operation == MemoryRevisionOperation.NO_CHANGE and self.candidate_ids:
            raise ValueError("no-change revisions cannot carry candidates")
        if self.operation != MemoryRevisionOperation.NO_CHANGE and not self.candidate_ids:
            raise ValueError("memory-changing revisions require candidates")
        if len(self.candidate_ids) != len(set(self.candidate_ids)):
            raise ValueError("memory revision candidate IDs must be unique")
        if self.revision_hash != model_digest(self, "revision_hash"):
            raise ValueError("memory revision digest mismatch")
        return self


class ExperienceEvaluation(ContractModel):
    id: str = Field(min_length=1, max_length=2048)
    revision_id: str = Field(min_length=1, max_length=2048)
    tenant_id: str = Field(min_length=1, max_length=2048)
    domain: str = Field(min_length=1, max_length=2048)
    evaluator_bundle_hash: str = Field(pattern=HASH_PATTERN)
    dataset_hash: str = Field(pattern=HASH_PATTERN)
    direct_support_count: int = Field(ge=0)
    counterexample_count: int = Field(ge=0)
    regression_case_count: int = Field(ge=1)
    candidate_task_success_rate: float = Field(ge=0.0, le=1.0)
    parent_task_success_rate: float = Field(ge=0.0, le=1.0)
    negative_transfer_rate: float = Field(ge=0.0, le=1.0)
    unsupported_causal_claims: int = Field(ge=0)
    cross_tenant_violations: int = Field(ge=0)
    decision: ExperienceEvaluationDecision
    concerns: list[str] = Field(default_factory=list)
    automatic_release_eligible: bool
    evaluated_at: str
    evaluation_hash: str = Field(pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_integrity(self) -> "ExperienceEvaluation":
        safely_accepted = (
            self.decision == ExperienceEvaluationDecision.ACCEPTED
            and self.unsupported_causal_claims == 0
            and self.cross_tenant_violations == 0
            and self.negative_transfer_rate <= 0.01
        )
        if self.automatic_release_eligible and not safely_accepted:
            raise ValueError("unsafe evaluation cannot be release eligible")
        datetime.fromisoformat(self.evaluated_at.replace("Z", "+00:00"))
        if self.evaluation_hash != model_digest(self, "evaluation_hash"):
            raise ValueError("experience evaluation digest mismatch")
        return self


class PracticeTask(ContractModel):
    id: str = Field(min_length=1, max_length=2048)
    sequence: int = Field(ge=1)
    objective: str = Field(min_length=1, max_length=2048)
    fixture_refs: list[str] = Field(min_length=1)
    allowed_internal_actions: list[str] = Field(min_length=1)
    success_criteria: list[str] = Field(min_length=1)
    discriminates_hypotheses: list[str] = Field(min_length=1)
    requires_target_retry: bool


class CurriculumBudget(ContractModel):
    max_practice_tasks: int = Field(ge=1)
    max_reasoning_calls: int = Field(ge=1)
    max_cost_microusd: int = Field(ge=0)
    max_wall_time_ms: int = Field(gt=0)


class CurriculumPlan(ContractModel):
    id: str = Field(min_length=1, max_length=2048)
    evaluation_id: str = Field(min_length=1, max_length=2048)
    revision_id: str = Field(min_length=1, max_length=2048)
    candidate_ids: list[str] = Field(min_length=1)
    tenant_id: str = Field(min_length=1, max_length=2048)
    domain: str = Field(min_length=1, max_length=2048)
    stage: LearningStage
    target_objective_hash: str = Field(pattern=HASH_PATTERN)
    memory_snapshot_hash: str = Field(pattern=HASH_PATTERN)
    hypotheses: list[str] = Field(min_length=1)
    practice_tasks: list[PracticeTask] = Field(min_length=1)
    budget: CurriculumBudget
    stop_conditions: list[str] = Field(min_length=1)
    status: LearningSessionStatus = LearningSessionStatus.PLANNED
    created_at: str
    plan_hash: str = Field(pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_integrity(self) -> "CurriculumPlan":
        if [task.sequence for task in self.practice_tasks] != list(range(1, len(self.practice_tasks) + 1)):
            raise ValueError("practice task sequences must be contiguous")
        if len(self.practice_tasks) > self.budget.max_practice_tasks:
            raise ValueError("practice task count exceeds budget")
        if self.stage == LearningStage.DEEP and not self.practice_tasks[-1].requires_target_retry:
            raise ValueError("deep curriculum must end with a target retry")
        if len(self.hypotheses) != len(set(self.hypotheses)):
            raise ValueError("curriculum hypotheses must be unique")
        if len(self.candidate_ids) != len(set(self.candidate_ids)):
            raise ValueError("curriculum candidate IDs must be unique")
        datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        if self.plan_hash != model_digest(self, "plan_hash"):
            raise ValueError("curriculum plan digest mismatch")
        return self


class PracticeRun(ContractModel):
    id: str = Field(min_length=1, max_length=2048)
    curriculum_id: str = Field(min_length=1, max_length=2048)
    practice_task_id: str = Field(min_length=1, max_length=2048)
    tenant_id: str = Field(min_length=1, max_length=2048)
    domain: str = Field(min_length=1, max_length=2048)
    memory_snapshot_hash: str = Field(pattern=HASH_PATTERN)
    status: LearningSessionStatus
    verdict: TaskVerdict | None = None
    verifier_report_hash: str | None = Field(default=None, pattern=HASH_PATTERN)
    outcome_hash: str | None = Field(default=None, pattern=HASH_PATTERN)
    actor_handoff_id: str | None = None
    started_at: str
    ended_at: str | None = None
    run_hash: str = Field(pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_integrity(self) -> "PracticeRun":
        terminal = self.status in {LearningSessionStatus.COMPLETED, LearningSessionStatus.FAILED}
        fields = (
            self.verdict,
            self.verifier_report_hash,
            self.outcome_hash,
            self.actor_handoff_id,
            self.ended_at,
        )
        if terminal and not all(fields):
            raise ValueError("practice terminal fields do not match status")
        if not terminal and any(field is not None for field in fields):
            raise ValueError("non-terminal practice run cannot carry terminal fields")
        if self.run_hash != model_digest(self, "run_hash"):
            raise ValueError("practice run digest mismatch")
        return self


class FrozenMemoryArtifact(ContractModel):
    id: str = Field(min_length=1, max_length=2048)
    tenant_id: str = Field(min_length=1, max_length=2048)
    domain: str = Field(min_length=1, max_length=2048)
    memory_binding: str = Field(min_length=1, max_length=2048)
    parent_artifact_id: str | None = None
    parent_memory_hash: str = Field(pattern=HASH_PATTERN)
    artifact_uri: str = Field(min_length=1, max_length=2048)
    artifact_hash: str = Field(pattern=HASH_PATTERN)
    schema_hash: str = Field(pattern=HASH_PATTERN)
    experience_ids: list[str] = Field(min_length=1)
    experiences: list[ExperienceCandidate] = Field(min_length=1)
    distiller_prompt_hash: str = Field(pattern=HASH_PATTERN)
    reconciler_prompt_hash: str = Field(pattern=HASH_PATTERN)
    evaluator_bundle_hash: str = Field(pattern=HASH_PATTERN)
    retrieval_version: str = Field(min_length=1, max_length=2048)
    status: MemoryArtifactStatus = MemoryArtifactStatus.CANDIDATE
    automatic_release_eligible: bool
    created_at: str

    @model_validator(mode="after")
    def validate_integrity(self) -> "FrozenMemoryArtifact":
        if len(self.experience_ids) != len(set(self.experience_ids)):
            raise ValueError("frozen memory experience IDs must be unique")
        if self.experience_ids != [item.id for item in self.experiences]:
            raise ValueError("frozen memory IDs must exactly match its immutable payload")
        if any(item.status != ExperienceCandidateStatus.RECONCILED for item in self.experiences):
            raise ValueError("frozen memory may contain only reconciled experience")
        if any(item.tenant_id != self.tenant_id or item.domain != self.domain for item in self.experiences):
            raise ValueError("frozen memory cannot contain cross-scope experience")
        expected_hash = canonical_hash(
            {
                "tenant_id": self.tenant_id,
                "domain": self.domain,
                "memory_binding": self.memory_binding,
                "experiences": [item.model_dump(mode="json") for item in self.experiences],
            }
        )
        if self.artifact_hash != expected_hash:
            raise ValueError("frozen memory artifact does not match its immutable payload")
        if self.automatic_release_eligible and self.status == MemoryArtifactStatus.REJECTED:
            raise ValueError("rejected memory cannot be release eligible")
        datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        return self


class ExperienceLearningBatch(ContractModel):
    id: str = Field(min_length=1, max_length=2048)
    handoff_id: str = Field(min_length=1, max_length=2048)
    tenant_id: str = Field(min_length=1, max_length=2048)
    domain: str = Field(min_length=1, max_length=2048)
    memory_binding: str = Field(min_length=1, max_length=2048)
    candidates: list[ExperienceCandidate] = Field(min_length=1)
    revisions: list[MemoryRevisionProposal] = Field(min_length=1)
    final_snapshot_hash: str = Field(pattern=HASH_PATTERN)
    batch_hash: str = Field(pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_integrity(self) -> "ExperienceLearningBatch":
        if len(self.candidates) != len(self.revisions):
            raise ValueError("each distilled experience must have one reconciliation result")
        if self.batch_hash != model_digest(self, "batch_hash"):
            raise ValueError("experience learning batch digest mismatch")
        return self


class MemoryReleaseTarget(ContractModel):
    """Read-only deployment target contract; KEE never activates it."""

    tenant_id: str = Field(min_length=1, max_length=2048)
    domain: str = Field(min_length=1, max_length=2048)
    memory_binding: str = Field(min_length=1, max_length=2048)
    baseline_artifact_id: str = Field(min_length=1, max_length=2048)
    baseline_artifact_hash: str = Field(pattern=HASH_PATTERN)
    candidate_artifact_id: str = Field(min_length=1, max_length=2048)
    candidate_artifact_hash: str = Field(pattern=HASH_PATTERN)
    protocol_schema_hash: str = Field(pattern=HASH_PATTERN)
    service_image_digests: dict[str, str] = Field(min_length=1)
    distiller_prompt_hash: str = Field(pattern=HASH_PATTERN)
    reconciler_prompt_hash: str = Field(pattern=HASH_PATTERN)
    evaluator_bundle_hash: str = Field(pattern=HASH_PATTERN)
    retrieval_version: str = Field(min_length=1, max_length=2048)
    policy_version: str = Field(min_length=1, max_length=2048)

    @model_validator(mode="after")
    def validate_integrity(self) -> "MemoryReleaseTarget":
        if self.baseline_artifact_id == self.candidate_artifact_id:
            raise ValueError("memory release candidate must differ from baseline")
        if self.baseline_artifact_hash == self.candidate_artifact_hash:
            raise ValueError("memory release candidate hash must differ from baseline")
        import re

        if any(
            not name.strip() or not re.fullmatch(IMAGE_DIGEST_PATTERN, digest)
            for name, digest in self.service_image_digests.items()
        ):
            raise ValueError("memory release service image digest is invalid")
        return self


class ImmutableMemorySnapshot(ContractModel):
    """KEE-internal immutable view; the hash is the v3.5 parent memory hash."""

    tenant_id: str
    domain: str
    memory_binding: str
    experiences: tuple[ExperienceCandidate, ...] = ()
    snapshot_hash: str = Field(pattern=HASH_PATTERN)

    @classmethod
    def create(
        cls,
        tenant_id: str,
        domain: str,
        memory_binding: str,
        experiences: tuple[ExperienceCandidate, ...] = (),
    ) -> "ImmutableMemorySnapshot":
        payload = {
            "domain": domain,
            "experiences": [item.model_dump(mode="json") for item in experiences],
            "memory_binding": memory_binding,
            "tenant_id": tenant_id,
        }
        return cls(
            tenant_id=tenant_id,
            domain=domain,
            memory_binding=memory_binding,
            experiences=experiences,
            snapshot_hash=canonical_hash(payload),
        )

    @model_validator(mode="after")
    def validate_integrity(self) -> "ImmutableMemorySnapshot":
        payload = {
            "domain": self.domain,
            "experiences": [item.model_dump(mode="json") for item in self.experiences],
            "memory_binding": self.memory_binding,
            "tenant_id": self.tenant_id,
        }
        if self.snapshot_hash != canonical_hash(payload):
            raise ValueError("immutable memory snapshot digest mismatch")
        if any(item.tenant_id != self.tenant_id or item.domain != self.domain for item in self.experiences):
            raise ValueError("memory snapshot cannot contain cross-scope experiences")
        return self


class EvaluationCase(ContractModel):
    id: str
    tenant_id: str
    domain: str
    supports_candidate: bool
    is_counterexample: bool = False
    parent_succeeded: bool
    candidate_succeeded: bool
    negative_transfer: bool = False
    causal_mechanism_verified: bool = False


class ExperienceEvaluationDataset(ContractModel):
    cases: tuple[EvaluationCase, ...] = Field(min_length=1)
    dataset_hash: str = Field(pattern=HASH_PATTERN)

    @classmethod
    def create(
        cls,
        cases: list[EvaluationCase],
    ) -> "ExperienceEvaluationDataset":
        payload = [case.model_dump(mode="json") for case in cases]
        return cls(
            cases=tuple(cases),
            dataset_hash=canonical_hash(payload),
        )

    @model_validator(mode="after")
    def validate_integrity(self) -> "ExperienceEvaluationDataset":
        if self.dataset_hash != canonical_hash([case.model_dump(mode="json") for case in self.cases]):
            raise ValueError("evaluation dataset digest mismatch")
        return self
