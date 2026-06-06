from llm_kee.models import ActionDefinition
from llm_kee.storage import KEEStore


class ActionRegistry:
    def __init__(self, store: KEEStore) -> None:
        self.store = store

    def register(self, action: ActionDefinition) -> ActionDefinition:
        return self.store.action_definitions.upsert(action)

    def list(self) -> list[ActionDefinition]:
        return self.store.action_definitions.list()

    def get_by_type(self, action_type: str) -> ActionDefinition | None:
        return next((action for action in self.list() if action.action_type == action_type), None)
