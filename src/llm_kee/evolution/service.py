from typing import Any

from llm_kee.models import ChangeSet, EvolutionEvent, KnowledgeVersion
from llm_kee.storage import KEEStore


class EvolutionService:
    def __init__(self, store: KEEStore) -> None:
        self.store = store

    def create_event(self, change_set: ChangeSet) -> EvolutionEvent:
        change_set = self.store.change_sets.upsert(change_set)
        previous = self._latest_version(change_set.target_type, change_set.target_id)
        if previous:
            previous.status = "superseded"
            previous.valid_to = str(change_set.created_at)
            self.store.knowledge_versions.upsert(previous)

        version = KnowledgeVersion(
            target_type=change_set.target_type,
            target_id=change_set.target_id,
            version=(previous.version + 1) if previous else 1,
            valid_from=str(change_set.created_at),
            status="active",
            snapshot=change_set.after,
            reason=change_set.reason,
            supersedes_version_id=previous.id if previous else None,
        )
        version = self.store.knowledge_versions.upsert(version)
        event = EvolutionEvent(
            event_type=change_set.operation,
            target_type=change_set.target_type,
            target_id=change_set.target_id,
            change_set_id=change_set.id,
            from_version_id=previous.id if previous else None,
            to_version_id=version.id,
            reason=change_set.reason,
            evidence_ids=change_set.evidence_ids,
        )
        return self.store.evolution_events.upsert(event)

    def history(self, target_id: str) -> list[EvolutionEvent]:
        return [
            event
            for event in self.store.evolution_events.list()
            if event.target_id == target_id
        ]

    def _latest_version(self, target_type: str, target_id: str) -> KnowledgeVersion | None:
        versions = [
            version
            for version in self.store.knowledge_versions.list()
            if version.target_type == target_type and version.target_id == target_id
        ]
        if not versions:
            return None
        return sorted(versions, key=lambda item: item.version)[-1]


def change_set_from_payload(payload: dict[str, Any]) -> ChangeSet:
    return ChangeSet.model_validate(payload)
