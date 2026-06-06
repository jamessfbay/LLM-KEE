from typing import Any

from pydantic import BaseModel, Field

from llm_kee.models import UpdateProposal
from llm_kee.models.base import new_id


class UpdatePlan(BaseModel):
    id: str = Field(default_factory=lambda: new_id("plan"))
    proposal_id: str
    operation: str
    target_type: str
    target_id: str | None = None
    status: str | None = None
    payload: dict[str, Any]


class UpdatePlanner:
    def build(self, proposal: UpdateProposal) -> UpdatePlan:
        return UpdatePlan(
            proposal_id=proposal.id,
            operation=proposal.proposal_type,
            target_type=proposal.target_type,
            target_id=proposal.target_id,
            status=proposal.status,
            payload={
                "rationale": proposal.rationale,
                "evidence_ids": proposal.evidence_ids,
                "change": proposal.proposed_change,
                "approved": str(proposal.status) == "approved",
            },
        )
