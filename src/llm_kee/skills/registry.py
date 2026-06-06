from typing import Any

from llm_kee.models import SkillDefinition, SkillPlan
from llm_kee.storage import KEEStore


class SkillRegistry:
    def __init__(self, store: KEEStore) -> None:
        self.store = store

    def register(self, skill: SkillDefinition) -> SkillDefinition:
        return self.store.skills.upsert(skill)

    def list(self) -> list[SkillDefinition]:
        return self.store.skills.list()

    def get(self, skill_id: str) -> SkillDefinition | None:
        return self.store.skills.get(skill_id)


class TaskClassifier:
    def classify(self, task: dict[str, Any]) -> str:
        if task.get("task_type"):
            return str(task["task_type"])
        text = " ".join(str(value).lower() for value in task.values())
        if "timeline" in text or "history" in text:
            return "timeline_reconstruction"
        if "missing" in text or "conflict" in text or "contradict" in text:
            return "missing_or_conflict_detection"
        if "schema" in text or "ontology" in text:
            return "ontology_design"
        return "intelligence_pack"


class SkillRetriever:
    TASK_SKILLS = {
        "timeline_reconstruction": ["rag_evidence_retrieval", "event_sourcing_timeline", "evidence_evaluation"],
        "missing_or_conflict_detection": ["rag_evidence_retrieval", "evidence_evaluation", "ontology_engineering"],
        "ontology_design": ["ontology_engineering", "evidence_evaluation"],
        "intelligence_pack": ["rag_evidence_retrieval", "evidence_evaluation", "ontology_engineering"],
    }

    def __init__(self, registry: SkillRegistry) -> None:
        self.registry = registry

    def retrieve(self, task_type: str, input_payload: dict[str, Any]) -> SkillPlan:
        available = {skill.id for skill in self.registry.list() if skill.enabled}
        selected = [
            skill_id
            for skill_id in self.TASK_SKILLS.get(task_type, self.TASK_SKILLS["intelligence_pack"])
            if skill_id in available
        ]
        return SkillPlan(
            task_type=task_type,
            skill_ids=selected,
            rationale=f"Selected {len(selected)} skills for {task_type}.",
            input_payload=input_payload,
        )
