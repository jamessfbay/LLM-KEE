from collections.abc import Callable

from llm_kee.models import KnowledgeEvolutionCycle, LearningSignal, MAPECycle
from llm_kee.storage import KEEStore


class KnowledgeEvolutionLoop:
    def __init__(self, store: KEEStore, mape_runner: Callable[[list[LearningSignal]], MAPECycle]) -> None:
        self.store = store
        self.mape_runner = mape_runner

    def run(self, signals: list[LearningSignal]) -> KnowledgeEvolutionCycle:
        mape_cycle = self.mape_runner(signals)
        cycle = KnowledgeEvolutionCycle(
            signal_ids=[signal.id for signal in signals],
            mape_cycle_id=mape_cycle.id,
            proposal_ids=mape_cycle.proposal_ids,
            evaluation_ids=mape_cycle.evaluation_ids,
            decision_ids=mape_cycle.decision_ids,
            evolution_event_ids=mape_cycle.evolution_event_ids,
            status=mape_cycle.status,
        )
        return self.store.knowledge_evolution_cycles.upsert(cycle)
