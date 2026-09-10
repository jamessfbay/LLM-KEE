from llm_kee.integrations import KGClient
from llm_kee.models import ProposalStatus, UpdateProposal
from llm_kee.planning import UpdatePlanner


class SafeApplyService:
    def __init__(self, kg_client: KGClient, planner: UpdatePlanner | None = None, *, allow_direct_apply: bool = True) -> None:
        self.kg_client = kg_client
        self.planner = planner or UpdatePlanner()
        self.allow_direct_apply = allow_direct_apply

    def apply(self, proposal: UpdateProposal) -> dict:
        if not self.allow_direct_apply:
            return {
                "status": "rejected",
                "message": "Direct apply is disabled; NOX owns authorization, execution, and activation.",
                "proposal_id": proposal.id,
            }
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
