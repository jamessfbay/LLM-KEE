from llm_kee.integrations import KGClient
from llm_kee.models import ProposalStatus, UpdateProposal
from llm_kee.planning import UpdatePlanner


class SafeApplyService:
    def __init__(self, kg_client: KGClient, planner: UpdatePlanner | None = None) -> None:
        self.kg_client = kg_client
        self.planner = planner or UpdatePlanner()

    def apply(self, proposal: UpdateProposal) -> dict:
        if proposal.status != ProposalStatus.APPROVED:
            return {
                "status": "rejected",
                "message": f"Proposal must be approved before apply; current status is {proposal.status}.",
                "proposal_id": proposal.id,
            }
        plan = self.planner.build(proposal)
        result = self.kg_client.apply_update_plan(plan)
        if result.get("status") in {"applied", "dry_run"}:
            proposal.status = ProposalStatus.APPLIED
        return result
