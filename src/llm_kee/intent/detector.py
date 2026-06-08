from typing import Any

from llm_kee.models import ConversationIntent, IntentPattern, TaskIntent
from llm_kee.storage import KEEStore


class IntentDetector:
    def __init__(self, store: KEEStore) -> None:
        self.store = store

    def detect_conversation(self, payload: dict[str, Any]) -> ConversationIntent:
        text = self._payload_text(payload)
        patterns = self._matching_patterns(text)
        best = patterns[0] if patterns else None
        intent = ConversationIntent(
            intent_type=best.intent_type if best else "general_knowledge_task",
            utterance=payload.get("utterance") or payload.get("question") or payload.get("message"),
            task_type=payload.get("task_type") or (best.intent_type if best else "general_knowledge_task"),
            payload=payload,
            matched_pattern_ids=[pattern.id for pattern in patterns],
            confidence=0.85 if best else 0.35,
            success_criteria=best.success_criteria if best else ["answer should be grounded and auditable"],
        )
        return self.store.conversation_intents.upsert(intent)

    def detect_task(self, payload: dict[str, Any]) -> TaskIntent:
        text = self._payload_text(payload)
        patterns = self._matching_patterns(text)
        best = patterns[0] if patterns else None
        intent = TaskIntent(
            intent_type=best.intent_type if best else "general_knowledge_task",
            task_type=payload.get("task_type") or (best.intent_type if best else "general_knowledge_task"),
            payload=payload,
            matched_pattern_ids=[pattern.id for pattern in patterns],
            confidence=0.85 if best else 0.35,
            success_criteria=best.success_criteria if best else ["workflow should produce grounded output"],
        )
        return self.store.task_intents.upsert(intent)

    def _matching_patterns(self, text: str) -> list[IntentPattern]:
        scored: list[tuple[int, IntentPattern]] = []
        for pattern in self.store.intent_patterns.list():
            if not pattern.enabled:
                continue
            score = sum(self._term_weight(term) for term in pattern.trigger_terms if term.lower() in text)
            if score:
                scored.append((score, pattern))
        return [pattern for _, pattern in sorted(scored, key=lambda item: item[0], reverse=True)]

    def _term_weight(self, term: str) -> int:
        if term.lower() in {"permit", "timeline", "risk", "status"}:
            return 3
        return 1

    def _payload_text(self, payload: dict[str, Any]) -> str:
        values = [
            str(payload.get(key, ""))
            for key in ("utterance", "question", "message", "task_type", "target_type", "target_id", "summary")
        ]
        values.extend(str(value) for value in payload.get("tags", []) if value)
        return " ".join(values).lower()
