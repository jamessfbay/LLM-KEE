import sys
from pathlib import Path
from typing import Any

from llm_kee.planning import UpdatePlan


class KGClient:
    def get_object(self, object_type: str, object_id: str) -> dict[str, Any]:
        return {
            "id": object_id,
            "type": object_type,
            "status": "unconfigured",
            "message": "Use a direct LLM-KG Python integration to resolve objects.",
        }

    def apply_update_plan(self, plan: UpdatePlan) -> dict[str, Any]:
        return {
            "status": "dry_run",
            "message": "No direct LLM-KG adapter is configured; update plan was not applied.",
            "plan": plan.model_dump(mode="json"),
        }

    def repair(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"status": "dry_run", "payload": payload}


class LLMKGClient(KGClient):
    def __init__(self, workspace: Path, project_path: Path | None = None) -> None:
        self.workspace = workspace
        if project_path:
            src_path = project_path / "src"
            if src_path.exists() and str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))

    def get_object(self, object_type: str, object_id: str) -> dict[str, Any]:
        from llm_kg import trace_object, verify_object

        if object_type == "query":
            from llm_kg import trace_query

            return trace_query(object_id, workspace=self.workspace).model_dump(mode="json")
        verification = verify_object(object_type, object_id, workspace=self.workspace)
        trace = trace_object(object_type, object_id, workspace=self.workspace) if object_type in {"claim", "relation", "entity", "evidence"} else None
        return {
            "verification": verification.model_dump(mode="json"),
            "trace": trace.model_dump(mode="json") if trace else None,
        }

    def apply_update_plan(self, plan: UpdatePlan) -> dict[str, Any]:
        from llm_kg import apply_update_plan

        return apply_update_plan(plan.model_dump(mode="json"), workspace=self.workspace).model_dump(mode="json")

    def repair(self, payload: dict[str, Any]) -> dict[str, Any]:
        from llm_kg import create_proposal

        target_type = str(payload.get("target_type") or "claim")
        target_id = str(payload.get("target_id") or "")
        if not target_id:
            return {"status": "rejected", "message": "repair payload requires target_id"}
        proposal = create_proposal(target_type, target_id, payload.get("change") or payload, workspace=self.workspace)
        return {"status": "proposal_created", "proposal": proposal.model_dump(mode="json")}
