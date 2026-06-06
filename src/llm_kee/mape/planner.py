from llm_kee.models import LearningSignal, MAPEAnalysis, MAPEPlan


class MAPEPlanner:
    def plan(self, analysis: MAPEAnalysis, signals: list[LearningSignal]) -> MAPEPlan:
        steps = []
        for action_type in analysis.recommended_actions:
            related = [
                signal.model_dump(mode="json")
                for signal in signals
                if self._matches(action_type, signal)
            ]
            if not related:
                related = [signal.model_dump(mode="json") for signal in signals]
            steps.append(
                {
                    "type": "workflow" if action_type == "ontology_engineering" else "action",
                    "action_type": action_type,
                    "skill_ids": ["ontology_engineering"] if action_type == "ontology_engineering" else [],
                    "input_payload": {
                        "task_type": self._task_type(action_type),
                        "target_id": self._target_id(related),
                        "signals": related,
                        "evidence_ids": self._evidence_ids(related),
                    },
                }
            )
        return MAPEPlan(
            analysis_id=analysis.id,
            steps=steps,
            requires_review=analysis.requires_review,
        )

    def _matches(self, action_type: str, signal: LearningSignal) -> bool:
        text = f"{signal.signal_type} {signal.summary} {signal.payload}".lower()
        if action_type == "detect_missing_or_conflicting_information":
            return any(word in text for word in ["conflict", "missing", "low_confidence"])
        if action_type == "rebuild_timeline":
            return any(word in text for word in ["timeline", "status"])
        if action_type == "ontology_engineering":
            return any(word in text for word in ["schema", "ontology"])
        return True

    def _task_type(self, action_type: str) -> str:
        if action_type == "rebuild_timeline":
            return "timeline_reconstruction"
        if action_type == "detect_missing_or_conflicting_information":
            return "missing_or_conflict_detection"
        if action_type == "ontology_engineering":
            return "ontology_design"
        return "intelligence_pack"

    def _target_id(self, signal_records: list[dict]) -> str | None:
        for signal in signal_records:
            payload = signal.get("payload") or {}
            if payload.get("target_id"):
                return payload["target_id"]
            if signal.get("source_id"):
                return signal["source_id"]
        return None

    def _evidence_ids(self, signal_records: list[dict]) -> list[str]:
        evidence_ids: list[str] = []
        for signal in signal_records:
            payload = signal.get("payload") or {}
            for evidence_id in payload.get("evidence_ids") or []:
                if evidence_id not in evidence_ids:
                    evidence_ids.append(evidence_id)
        return evidence_ids
