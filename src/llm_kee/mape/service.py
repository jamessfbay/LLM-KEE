from llm_kee.models import LearningSignal, MAPECycle, MAPEObservation
from llm_kee.storage import KEEStore


class MAPELoop:
    def __init__(self, store: KEEStore) -> None:
        self.store = store

    def run(self, signals: list[LearningSignal]) -> MAPECycle:
        observation = MAPEObservation(
            signal_ids=[signal.id for signal in signals],
            summary=f"Observed {len(signals)} learning signals.",
            payload={"signals": [signal.model_dump(mode="json") for signal in signals]},
        )
        observation = self.store.mape_observations.upsert(observation)
        analysis = {
            "signal_count": len(signals),
            "high_priority_count": sum(1 for signal in signals if signal.priority >= 8),
            "signal_types": sorted({signal.signal_type for signal in signals}),
        }
        plan = {
            "recommended_action": "generate_update_proposals" if signals else "no_op",
            "requires_review": analysis["high_priority_count"] > 0,
        }
        execution = {
            "status": "planned",
            "created_proposals": [],
        }
        learned = {
            "notes": "MAPE skeleton recorded observation, analysis, plan, execution, and learn phases.",
        }
        cycle = MAPECycle(
            observation_ids=[observation.id],
            analysis=analysis,
            plan=plan,
            execution=execution,
            learned=learned,
            status="completed",
        )
        return self.store.mape_cycles.upsert(cycle)
