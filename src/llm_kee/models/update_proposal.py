from typing import Any

from pydantic import Field

from llm_kee.models.base import StoredModel, new_id
from llm_kee.models.enums import ProposalStatus, ProposalType, TargetType


class UpdateProposalCreate(StoredModel):
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


class UpdateProposal(UpdateProposalCreate):
    pass
