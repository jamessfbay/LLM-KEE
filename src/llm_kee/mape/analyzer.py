from llm_kee.models import LearningSignal, MAPEAnalysis, SignalType


class MAPEAnalyzer:
    def analyze(self, signals: list[LearningSignal]) -> MAPEAnalysis:
        signal_types = sorted({str(signal.signal_type) for signal in signals})
        high_priority_count = sum(1 for signal in signals if signal.priority >= 8)
        recommended_actions = sorted({self._action_for_signal(signal) for signal in signals})
        impact_level = "high" if high_priority_count else "medium" if signals else "low"
        return MAPEAnalysis(
            signal_ids=[signal.id for signal in signals],
            signal_count=len(signals),
            high_priority_count=high_priority_count,
            signal_types=signal_types,
            impact_level=impact_level,
            recommended_actions=recommended_actions,
            requires_review=high_priority_count > 0,
            rationale=f"Analyzed {len(signals)} signals with impact level {impact_level}.",
        )

    def _action_for_signal(self, signal: LearningSignal) -> str:
        signal_type = signal.signal_type
        text = f"{signal.summary} {signal.payload}".lower()
        if signal_type == SignalType.GRAPH_CONFLICT:
            return "detect_missing_or_conflicting_information"
        if signal_type == SignalType.LOW_CONFIDENCE_CLAIM:
            return "detect_missing_or_conflicting_information"
        if signal_type == SignalType.SCHEMA_GAP:
            return "ontology_engineering"
        if "timeline" in text or "status" in text:
            return "rebuild_timeline"
        return "generate_intelligence_pack"
