import sys
from pathlib import Path

from llm_kee.config import KGIntegrationConfig, Settings
from llm_kee.models import ProposalStatus, ProposalType, TargetType, UpdateProposal
from llm_kee.services import KEEEngine


def test_direct_llm_kg_adapter_applies_approved_plan(tmp_path: Path) -> None:
    kg_project = Path(__file__).resolve().parents[2] / "LLM-KG"
    if not kg_project.exists():
        kg_project = Path(__file__).resolve().parents[1].parent / "LLM-KG"
    sys.path.insert(0, str(kg_project / "src"))

    from llm_kg.models import Claim, Evidence
    from llm_kg.storage import JsonlStore

    kg_workspace = tmp_path / "kg"
    kg_workspace.mkdir()
    store = JsonlStore(kg_workspace)
    store.upsert("evidence.jsonl", [Evidence(id="ev_1", source_id="doc_1", quote="A requires C.", confidence=0.9)])
    store.upsert("claims.jsonl", [Claim(id="claim_1", text="A requires B.", evidence_ids=["ev_1"], confidence=0.7)])

    engine = KEEEngine(
        Settings(
            workspace=tmp_path / "kee",
            kg=KGIntegrationConfig(enable_direct_adapter=True, workspace=kg_workspace, project_path=kg_project),
        )
    )
    proposal = UpdateProposal(
        proposal_type=ProposalType.UPDATE_CLAIM,
        target_type=TargetType.CLAIM,
        target_id="claim_1",
        title="Update claim",
        rationale="Evidence says C.",
        evidence_ids=["ev_1"],
        proposed_change={"text": "A requires C.", "evidence_ids": ["ev_1"]},
        status=ProposalStatus.APPROVED,
    )

    result = engine.safe_apply.apply(proposal)

    updated = JsonlStore(kg_workspace).load("claims.jsonl", Claim)[0]
    assert result["status"] == "applied"
    assert updated.text == "A requires C."
    assert proposal.status == ProposalStatus.APPLIED


def test_direct_llm_kg_adapter_rejects_unapproved_plan(tmp_path: Path) -> None:
    engine = KEEEngine(
        Settings(
            workspace=tmp_path / "kee",
            kg=KGIntegrationConfig(enable_direct_adapter=True, workspace=tmp_path / "kg", project_path=Path(__file__).resolve().parents[2] / "LLM-KG"),
        )
    )
    proposal = UpdateProposal(
        proposal_type=ProposalType.UPDATE_CLAIM,
        target_type=TargetType.CLAIM,
        target_id="claim_1",
        title="Update claim",
        rationale="No approval.",
        proposed_change={"text": "A requires C."},
        status=ProposalStatus.NEED_MORE_EVIDENCE,
    )

    result = engine.safe_apply.apply(proposal)

    assert result["status"] == "rejected"
    assert proposal.status == ProposalStatus.NEED_MORE_EVIDENCE
