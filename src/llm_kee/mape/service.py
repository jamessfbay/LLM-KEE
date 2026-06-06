from llm_kee.mape.analyzer import MAPEAnalyzer
from llm_kee.mape.executor import MAPEExecutor
from llm_kee.mape.planner import MAPEPlanner
from llm_kee.models import LearningSignal, MAPECycle, MAPEObservation
from llm_kee.storage import KEEStore


class MAPELoop:
    def __init__(
        self,
        store: KEEStore,
        analyzer: MAPEAnalyzer | None = None,
        planner: MAPEPlanner | None = None,
        executor: MAPEExecutor | None = None,
    ) -> None:
        self.store = store
        self.analyzer = analyzer or MAPEAnalyzer()
        self.planner = planner or MAPEPlanner()
        self.executor = executor

    def run(self, signals: list[LearningSignal]) -> MAPECycle:
        observation = MAPEObservation(
            signal_ids=[signal.id for signal in signals],
            summary=f"Observed {len(signals)} learning signals.",
            payload={"signals": [signal.model_dump(mode="json") for signal in signals]},
        )
        observation = self.store.mape_observations.upsert(observation)
        analysis = self.store.mape_analyses.upsert(self.analyzer.analyze(signals))
        plan = self.store.mape_plans.upsert(self.planner.plan(analysis, signals))
        execution = self.executor.execute(plan) if self.executor else None
        learned = {
            "notes": "MAPE cycle analyzed signals, planned actions, executed configured steps, and recorded resulting IDs.",
        }
        cycle = MAPECycle(
            observation_ids=[observation.id],
            analysis=analysis.model_dump(mode="json"),
            plan=plan.model_dump(mode="json"),
            execution=execution.model_dump(mode="json") if execution else {"status": "planned"},
            learned=learned,
            action_run_ids=execution.action_run_ids if execution else [],
            artifact_ids=execution.artifact_ids if execution else [],
            proposal_ids=execution.proposal_ids if execution else [],
            evaluation_ids=execution.evaluation_ids if execution else [],
            decision_ids=execution.decision_ids if execution else [],
            evolution_event_ids=execution.evolution_event_ids if execution else [],
            status="completed",
        )
        return self.store.mape_cycles.upsert(cycle)
