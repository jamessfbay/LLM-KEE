from typing import Any, Literal

from pydantic import Field

from llm_kee.models.base import StoredModel, new_id
from llm_kee.models.enums import ProposalStatus, ProposalType, TargetType


class UpdateProposalCreate(StoredModel):
    contract_version: Literal["improvement-proposal/2.0"] = "improvement-proposal/2.0"
    id: str = Field(default_factory=lambda: new_id("prop"))
    proposal_type: ProposalType
    target_type: TargetType
    target_id: str | None = None
    title: str
    rationale: str
    evidence_ids: list[str] = Field(default_factory=list)
    source_signal_ids: list[str] = Field(default_factory=list)
    proposed_change: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    status: ProposalStatus = ProposalStatus.DRAFT
    hypothesis: str | None = None
    base_version: str | None = None
    evidence_snapshot_hash: str | None = None
    success_metrics: list[dict[str, Any]] = Field(default_factory=list)
    guardrails: list[dict[str, Any]] = Field(default_factory=list)
    evaluation_plan: dict[str, Any] = Field(default_factory=dict)
    rollback_plan: dict[str, Any] = Field(default_factory=dict)


class UpdateProposal(UpdateProposalCreate):
    pass
