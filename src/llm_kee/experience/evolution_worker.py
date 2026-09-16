"""Automatic, scope-isolated KEE experience lifecycle orchestration.

The evaluator consumes the append-only NOX experience stream. It can submit
evaluations, curricula, terminal practice receipts, and candidate artifacts;
it cannot create actor handoffs, execute actions, create releases, or activate
memory. A separate sandbox runner may supply independently verified practice
handoffs through the injected ``practice_runner`` interface.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from llm_kee.experience.artifact import MemoryArtifactFreezer
from llm_kee.experience.curriculum import CurriculumPlanner, PracticeCoordinator
from llm_kee.experience.evaluator import EVALUATOR_BUNDLE_HASH, ExperienceEvaluator
from llm_kee.experience.hashing import canonical_hash, seal_model
from llm_kee.experience.models import (
    ActorLearningHandoff,
    EvaluationCase,
    ExperienceCandidate,
    ExperienceCandidateStatus,
    CurriculumPlan,
    ExperienceEvaluation,
    ExperienceEvaluationDataset,
    ExperienceEvaluationDecision,
    FrozenMemoryArtifact,
    ImmutableMemorySnapshot,
    LearningStage,
    MemoryRevisionOperation,
    MemoryRevisionProposal,
    PracticeRun,
    TaskVerdict,
)


JsonRequest = Callable[[str, str, dict[str, Any] | None], dict[str, Any]]
PracticeRunner = Callable[[dict[str, Any], dict[str, Any]], ActorLearningHandoff | None]


@dataclass(frozen=True)
class EvolutionConfig:
    base_url: str
    token: str
    tenant_id: str
    domain: str
    memory_binding: str
    schema_hash: str
    retrieval_version: str = "kee-conditional-memory-v1"
    interval_seconds: int = 60
    practice_runner_url: str | None = None
    practice_runner_token: str | None = None
    practice_timeout_seconds: int = 120

    @classmethod
    def from_env(cls) -> "EvolutionConfig":
        tenant = os.environ.get("NOX_KEE_TENANT_ID") or os.environ.get("NOX_DEFAULT_TENANT_ID", "")
        domain = os.environ.get("NOX_KEE_DOMAIN", "")
        token = os.environ.get("NOX_EXPERIENCE_EVALUATOR_TOKEN", "")
        schema_hash = os.environ.get("NOX_KEE_SCHEMA_HASH", "")
        if not tenant or not domain or not token or len(schema_hash) != 64:
            raise ValueError(
                "NOX_KEE_TENANT_ID, NOX_KEE_DOMAIN, NOX_EXPERIENCE_EVALUATOR_TOKEN "
                "and a 64-character NOX_KEE_SCHEMA_HASH are required"
            )
        practice_url = os.environ.get("NOX_KEE_PRACTICE_RUNNER_URL", "").strip()
        practice_token = os.environ.get("NOX_KEE_PRACTICE_RUNNER_TOKEN", "").strip()
        if bool(practice_url) != bool(practice_token):
            raise ValueError(
                "NOX_KEE_PRACTICE_RUNNER_URL and NOX_KEE_PRACTICE_RUNNER_TOKEN must be configured together"
            )
        return cls(
            base_url=os.environ.get("NOX_V3_INTERNAL_URL", "http://state-engine:4318").rstrip("/"),
            token=token,
            tenant_id=tenant,
            domain=domain,
            memory_binding=os.environ.get("NOX_KEE_MEMORY_BINDING", f"kee-{domain}"),
            schema_hash=schema_hash,
            retrieval_version=os.environ.get(
                "NOX_KEE_RETRIEVAL_VERSION", "kee-conditional-memory-v1"
            ),
            interval_seconds=max(
                5, int(os.environ.get("NOX_KEE_EVOLUTION_INTERVAL_SECONDS", "60"))
            ),
            practice_runner_url=practice_url.rstrip("/") or None,
            practice_runner_token=practice_token or None,
            practice_timeout_seconds=max(
                5, min(900, int(os.environ.get("NOX_KEE_PRACTICE_TIMEOUT_SECONDS", "120")))
            ),
        )


class EvolutionClient:
    def __init__(self, config: EvolutionConfig) -> None:
        self.config = config

    def request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload, separators=(",", ":")).encode()
        request = Request(
            f"{self.config.base_url}{path}",
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self.config.token}",
                "Content-Type": "application/json",
                "X-NOX-Actor": "kee:independent-evolution-worker-v1",
            },
        )
        with urlopen(request, timeout=20) as response:
            value = json.loads(response.read())
        if not isinstance(value, dict):
            raise RuntimeError("NOX evolution endpoint returned a non-object")
        return value


class SandboxPracticeRunner:
    """Use a separately credentialed sandbox/replay service for practice."""

    def __init__(
        self,
        url: str,
        token: str,
        timeout_seconds: int = 120,
        opener: Callable[..., Any] = urlopen,
    ) -> None:
        if not url.startswith(("http://", "https://")) or not token.strip():
            raise ValueError("A valid practice runner URL and token are required")
        self.url = url.rstrip("/")
        self.token = token
        self.timeout_seconds = timeout_seconds
        self.opener = opener

    def __call__(
        self, plan: dict[str, Any], task: dict[str, Any]
    ) -> ActorLearningHandoff | None:
        payload = json.dumps(
            {
                "protocol_version": "3.5",
                "mode": "historical_replay_or_sandbox",
                "plan": plan,
                "task": task,
            },
            separators=(",", ":"),
        ).encode()
        request = Request(
            f"{self.url}/v1/practice",
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "X-NOX-Actor": "kee:sandbox-practice-runner-v1",
            },
        )
        try:
            with self.opener(request, timeout=self.timeout_seconds) as response:
                value = json.loads(response.read())
        except HTTPError as error:
            if error.code in (404, 409, 425):
                return None
            raise
        if not isinstance(value, dict):
            raise RuntimeError("Practice runner returned a non-object")
        if value.get("status") in ("pending", "not_ready"):
            return None
        raw = value.get("handoff")
        if not isinstance(raw, dict):
            raise RuntimeError("Practice runner returned no sealed actor handoff")
        handoff = ActorLearningHandoff.model_validate(raw)
        if handoff.tenant_id != plan.get("tenant_id") or handoff.domain != plan.get("domain"):
            raise RuntimeError("Practice runner handoff crossed tenant or domain scope")
        return handoff

    def evaluate_rollout(self, assignment: dict[str, Any]) -> dict[str, Any] | None:
        """Replay both frozen memory arms and return an independently measured observation."""
        request = Request(
            f"{self.url}/v1/memory-rollout",
            data=json.dumps(
                {
                    "protocol_version": "3.5",
                    "mode": "paired_historical_replay_or_shadow",
                    "assignment": assignment,
                },
                separators=(",", ":"),
            ).encode(),
            method="POST",
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "X-NOX-Actor": "kee:sandbox-practice-runner-v1",
            },
        )
        try:
            with self.opener(request, timeout=self.timeout_seconds) as response:
                value = json.loads(response.read())
        except HTTPError as error:
            if error.code in (404, 409, 425):
                return None
            raise
        if not isinstance(value, dict) or value.get("status") in ("pending", "not_ready"):
            return None
        observation = value.get("observation")
        if not isinstance(observation, dict):
            raise RuntimeError("Practice runner returned no memory rollout observation")
        for key in ("release_id", "run_id", "phase", "arm"):
            if observation.get(key) != assignment.get(key):
                raise RuntimeError("Memory rollout observation does not match its assignment")
        expected_id = (
            assignment.get("shadow_artifact_id")
            if assignment.get("arm") == "shadow_candidate"
            else assignment.get("formal_artifact_id")
        )
        expected_hash = (
            assignment.get("shadow_artifact_hash")
            if assignment.get("arm") == "shadow_candidate"
            else assignment.get("formal_artifact_hash")
        )
        if observation.get("artifact_id") != expected_id or observation.get("artifact_hash") != expected_hash:
            raise RuntimeError("Memory rollout observation changed the frozen artifact")
        return observation


class EvolutionWorker:
    def __init__(
        self,
        config: EvolutionConfig,
        request: JsonRequest | None = None,
        practice_runner: PracticeRunner | None = None,
    ) -> None:
        self.config = config
        self.request = request or EvolutionClient(config).request
        self.practice_runner = practice_runner
        self.evaluator = ExperienceEvaluator()
        self.curriculum = CurriculumPlanner()
        self.practice = PracticeCoordinator()
        self.freezer = MemoryArtifactFreezer()

    @property
    def scope(self) -> str:
        return urlencode({"tenant_id": self.config.tenant_id, "domain": self.config.domain})

    def events(self) -> list[dict[str, Any]]:
        after = 0
        events: list[dict[str, Any]] = []
        while True:
            page = self.request(
                "GET", f"/api/v3.5/experience-events?{self.scope}&after={after}&limit=200", None
            )
            items = page.get("items", [])
            events.extend(items)
            next_cursor = page.get("next")
            if next_cursor is None:
                return events
            after = int(next_cursor)

    def heartbeat(self, status: str, detail: dict[str, Any]) -> None:
        self.request(
            "POST",
            f"/api/v3.5/kee-evaluator-heartbeat?{self.scope}",
            {"status": status, "detail": detail},
        )

    def collect_rollout_observations(self, events: list[dict[str, Any]]) -> int:
        evaluator = getattr(self.practice_runner, "evaluate_rollout", None)
        if not callable(evaluator):
            return 0
        observed = {
            item["subject_id"]
            for item in events
            if item.get("event_type") == "memory.rollout_observed"
        }
        changed = 0
        after = 0
        while True:
            page = self.request(
                "GET",
                f"/api/v3.5/memory-rollout-assignments?{self.scope}&after={after}&limit=200",
                None,
            )
            for assignment in page.get("items", []):
                if assignment.get("run_id") in observed:
                    continue
                observation = evaluator(assignment)
                if observation is None:
                    continue
                self.request(
                    "POST",
                    f"/api/v3.5/memory-rollout-observations?{self.scope}",
                    observation,
                )
                observed.add(assignment["run_id"])
                changed += 1
            next_cursor = page.get("next")
            if next_cursor is None:
                return changed
            after = int(next_cursor)

    def active_memory(self) -> tuple[ImmutableMemorySnapshot, str, str]:
        binding = self.config.memory_binding
        try:
            response = self.request(
                "GET", f"/api/v3.5/memory-artifacts/{binding}/active?{self.scope}", None
            )
            raw = response["artifact"]
            if raw.get("artifact_uri") == "nox://memory/genesis":
                snapshot = ImmutableMemorySnapshot.create(
                    self.config.tenant_id, self.config.domain, binding, ()
                )
            else:
                artifact = FrozenMemoryArtifact.model_validate(raw)
                snapshot = ImmutableMemorySnapshot.create(
                    artifact.tenant_id,
                    artifact.domain,
                    artifact.memory_binding,
                    tuple(artifact.experiences),
                )
            return snapshot, raw["id"], response["artifact_hash"]
        except HTTPError as error:
            body = error.read().decode(errors="replace")
            if error.code != 404 or "Active memory artifact not found" not in body:
                raise
        genesis = self.request(
            "GET", f"/api/v3.5/memory-artifacts/{binding}/genesis?{self.scope}", None
        )
        canonical = genesis["canonical"]
        snapshot = ImmutableMemorySnapshot.create(
            canonical["tenant_id"], canonical["domain"], canonical["memory_binding"], ()
        )
        if snapshot.snapshot_hash != genesis["artifact_hash"]:
            raise RuntimeError("NOX deterministic genesis hash mismatch")
        return snapshot, genesis["artifact_id"], genesis["artifact_hash"]

    @staticmethod
    def _indexes(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for kind, key in (
            ("experience.handoff_ready", "handoffs"),
            ("experience.distilled", "candidates"),
            ("experience.reconciled", "revisions"),
            ("experience.evaluated", "evaluations"),
            ("curriculum.planned", "curricula"),
            ("practice.completed", "practice"),
            ("memory.candidate_created", "artifacts"),
        ):
            result[key] = {
                item["subject_id"]: item["data"] for item in events if item["event_type"] == kind
            }
        return result

    @staticmethod
    def _bootstrap_dataset(revision: MemoryRevisionProposal, handoff: ActorLearningHandoff) -> ExperienceEvaluationDataset:
        # The author's own task result is deliberately not counted as direct
        # support. It only proves that an independent experiment is needed.
        passed = handoff.verdict == TaskVerdict.PASS
        return ExperienceEvaluationDataset.create([
            EvaluationCase(
                id=f"bootstrap_{revision.id}", tenant_id=revision.tenant_id, domain=revision.domain,
                supports_candidate=False, parent_succeeded=passed, candidate_succeeded=passed,
                negative_transfer=False, causal_mechanism_verified=False,
            )
        ])

    def _practice_dataset(
        self, revision: MemoryRevisionProposal, curricula: list[dict[str, Any]], runs: list[dict[str, Any]]
    ) -> ExperienceEvaluationDataset | None:
        relevant = [plan for plan in curricula if plan.get("revision_id") == revision.id]
        for plan in relevant:
            task_ids = {task["id"] for task in plan["practice_tasks"]}
            completed = [run for run in runs if run.get("curriculum_id") == plan["id"]]
            if {run["practice_task_id"] for run in completed} != task_ids:
                continue
            cases = []
            for run in completed:
                passed = run["verdict"] == TaskVerdict.PASS
                cases.append(EvaluationCase(
                    id=run["id"], tenant_id=run["tenant_id"], domain=run["domain"],
                    supports_candidate=passed, is_counterexample=not passed,
                    parent_succeeded=True, candidate_succeeded=passed,
                    negative_transfer=not passed, causal_mechanism_verified=False,
                ))
            return ExperienceEvaluationDataset.create(cases)
        return None

    @staticmethod
    def _reconciled(candidate: ExperienceCandidate) -> ExperienceCandidate:
        payload = candidate.model_dump(mode="json", exclude={"candidate_hash"})
        payload["status"] = ExperienceCandidateStatus.RECONCILED
        return seal_model(ExperienceCandidate, "candidate_hash", **payload)

    def run_once(self) -> int:
        events = self.events()
        index = self._indexes(events)
        handoffs = {key: ActorLearningHandoff.model_validate(value) for key, value in index["handoffs"].items()}
        candidates = {key: ExperienceCandidate.model_validate(value) for key, value in index["candidates"].items()}
        revisions = [MemoryRevisionProposal.model_validate(value) for value in index["revisions"].values()]
        evaluations = [ExperienceEvaluation.model_validate(value) for value in index["evaluations"].values()]
        curricula = list(index["curricula"].values())
        practice_runs = list(index["practice"].values())
        artifact_revisions = {
            evaluation.revision_id
            for artifact in index["artifacts"].values()
            for evaluation in evaluations
            if evaluation.evaluator_bundle_hash == artifact.get("evaluator_bundle_hash")
            and set(artifact.get("experience_ids", [])) >= set(
                next((revision.candidate_ids for revision in revisions if revision.id == evaluation.revision_id), [])
            )
        }
        changed = self.collect_rollout_observations(events)
        for revision in revisions:
            if revision.operation == MemoryRevisionOperation.NO_CHANGE or revision.id in artifact_revisions:
                continue
            relevant_candidates = [candidates[item] for item in revision.candidate_ids if item in candidates]
            if len(relevant_candidates) != len(revision.candidate_ids):
                continue
            source_handoffs = list({candidate.handoff_id: handoffs[candidate.handoff_id] for candidate in relevant_candidates}.values())
            existing = [evaluation for evaluation in evaluations if evaluation.revision_id == revision.id]
            practice_dataset = self._practice_dataset(revision, curricula, practice_runs)
            dataset = practice_dataset or self._bootstrap_dataset(revision, source_handoffs[0])
            evaluation = next((item for item in existing if item.dataset_hash == dataset.dataset_hash), None)
            if evaluation is None:
                evaluation = self.evaluator.evaluate(revision, relevant_candidates, dataset, source_handoffs)
                self.request("POST", f"/api/v3.5/experience-evaluations?{self.scope}", evaluation.model_dump(mode="json"))
                evaluations.append(evaluation)
                changed += 1
            if evaluation.decision == ExperienceEvaluationDecision.NEEDS_EXPERIMENT:
                plan_payload = next((plan for plan in curricula if plan.get("evaluation_id") == evaluation.id), None)
                if plan_payload is None:
                    fixtures = list(dict.fromkeys(ref for candidate in relevant_candidates for ref in candidate.evidence_refs))
                    plan = self.curriculum.plan(
                        evaluation=evaluation, candidates=relevant_candidates, stage=LearningStage.BROAD,
                        target_objective=f"Validate bounded experience revision {revision.id}",
                        memory_snapshot_hash=revision.parent_memory_hash, fixture_refs=fixtures,
                    )
                    self.request("POST", f"/api/v3.5/curriculum-plans?{self.scope}", plan.model_dump(mode="json"))
                    plan_payload = plan.model_dump(mode="json")
                    curricula.append(plan_payload)
                    changed += 1
                if self.practice_runner is not None:
                    plan = CurriculumPlan.model_validate(plan_payload)
                    existing_tasks = {
                        item["practice_task_id"] for item in practice_runs
                        if item.get("curriculum_id") == plan.id
                    }
                    pending = {
                        item.practice_task_id: item for item in self.practice.begin_batch(plan)
                        if item.practice_task_id not in existing_tasks
                    }
                    for task in plan.practice_tasks:
                        if task.id not in pending:
                            continue
                        handoff = self.practice_runner(
                            plan.model_dump(mode="json"), task.model_dump(mode="json")
                        )
                        if handoff is None:
                            continue
                        completed = self.practice.complete_run(pending[task.id], handoff)
                        self.request(
                            "POST", f"/api/v3.5/practice-runs?{self.scope}",
                            completed.model_dump(mode="json"),
                        )
                        practice_runs.append(completed.model_dump(mode="json"))
                        changed += 1
                continue
            if evaluation.decision != ExperienceEvaluationDecision.ACCEPTED or revision.id in artifact_revisions:
                continue
            snapshot, parent_id, parent_hash = self.active_memory()
            if snapshot.snapshot_hash != revision.parent_memory_hash or parent_hash != revision.parent_memory_hash:
                continue
            updated = ImmutableMemorySnapshot.create(
                snapshot.tenant_id, snapshot.domain, snapshot.memory_binding,
                snapshot.experiences + tuple(self._reconciled(item) for item in relevant_candidates),
            )
            if updated.snapshot_hash != revision.proposed_snapshot_hash:
                continue
            accepted = [item for item in evaluations if item.revision_id == revision.id and item.decision == ExperienceEvaluationDecision.ACCEPTED]
            artifact = self.freezer.freeze_candidate(
                snapshot=updated, parent_memory_hash=parent_hash, evaluations=accepted,
                evaluator_bundle_hash=EVALUATOR_BUNDLE_HASH, schema_hash=self.config.schema_hash,
                retrieval_version=self.config.retrieval_version, parent_artifact_id=parent_id,
            )
            self.request("POST", f"/api/v3.5/memory-artifacts?{self.scope}", artifact.model_dump(mode="json"))
            artifact_revisions.add(revision.id)
            changed += 1
        return changed


def main() -> None:
    parser = argparse.ArgumentParser(description="NOX v3.5 automatic experience evolution worker")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    config = EvolutionConfig.from_env()
    practice_runner = (
        SandboxPracticeRunner(
            config.practice_runner_url,
            config.practice_runner_token or "",
            config.practice_timeout_seconds,
        )
        if config.practice_runner_url
        else None
    )
    worker = EvolutionWorker(config, practice_runner=practice_runner)
    while True:
        try:
            changed = worker.run_once()
            worker.heartbeat("healthy", {"changed": changed})
            print(json.dumps({"event": "kee.evolution_cycle", "changed": changed}), flush=True)
        except Exception as error:
            try:
                worker.heartbeat("degraded", {"error": str(error)[:2048]})
            except Exception:
                pass
            print(json.dumps({"event": "kee.evolution_cycle_failed", "error": str(error)}), file=sys.stderr, flush=True)
            if args.once:
                raise
        if args.once:
            return
        time.sleep(config.interval_seconds)


if __name__ == "__main__":
    main()
