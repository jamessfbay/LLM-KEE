"""Scope-isolated NOX v3.5 governed experience learner.

The worker has proposal-only learner credentials. It reads verified actor
handoffs, distills and reconciles them against the active immutable memory,
then submits one atomic batch. It cannot evaluate, publish, execute, or change
permissions.
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
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from llm_kee.experience.models import (
    ActorLearningHandoff,
    FrozenMemoryArtifact,
    ImmutableMemorySnapshot,
)
from llm_kee.experience.service import GovernedExperienceLearning


JsonRequest = Callable[[str, str, dict[str, Any] | None], dict[str, Any]]


@dataclass(frozen=True)
class WorkerConfig:
    base_url: str
    token: str
    tenant_id: str
    domain: str
    memory_binding: str
    interval_seconds: int = 60
    batch_size: int = 25

    @classmethod
    def from_env(cls) -> "WorkerConfig":
        tenant = os.environ.get("NOX_KEE_TENANT_ID") or os.environ.get("NOX_DEFAULT_TENANT_ID", "")
        domain = os.environ.get("NOX_KEE_DOMAIN", "")
        token = os.environ.get("NOX_LEARNER_TOKEN", "")
        if not tenant or not domain or not token:
            raise ValueError("NOX_KEE_TENANT_ID, NOX_KEE_DOMAIN and NOX_LEARNER_TOKEN are required")
        return cls(
            base_url=os.environ.get("NOX_V3_INTERNAL_URL", "http://state-engine:4318").rstrip("/"),
            token=token,
            tenant_id=tenant,
            domain=domain,
            memory_binding=os.environ.get("NOX_KEE_MEMORY_BINDING", f"kee-{domain}"),
            interval_seconds=max(5, int(os.environ.get("NOX_KEE_LEARNING_INTERVAL_SECONDS", "60"))),
            batch_size=min(200, max(1, int(os.environ.get("NOX_KEE_BATCH_SIZE", "25")))),
        )


class NoxExperienceClient:
    def __init__(self, config: WorkerConfig) -> None:
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
                "X-NOX-Actor": "kee:experience-worker-v1",
            },
        )
        with urlopen(request, timeout=20) as response:
            value = json.loads(response.read())
        if not isinstance(value, dict):
            raise RuntimeError("NOX experience endpoint returned a non-object")
        return value


class ExperienceWorker:
    def __init__(self, config: WorkerConfig, request: JsonRequest | None = None) -> None:
        self.config = config
        self.request = request or NoxExperienceClient(config).request
        self.learning = GovernedExperienceLearning()

    @property
    def scope_query(self) -> str:
        return urlencode({"tenant_id": self.config.tenant_id, "domain": self.config.domain})

    def memory_snapshot(self) -> ImmutableMemorySnapshot:
        binding = quote(self.config.memory_binding, safe="")
        try:
            response = self.request(
                "GET",
                f"/api/v3.5/memory-artifacts/{binding}/active?{self.scope_query}",
                None,
            )
        except HTTPError as error:
            body = error.read().decode(errors="replace")
            if error.code != 404 or "Active memory artifact not found" not in body:
                raise
            genesis = self.request(
                "GET",
                f"/api/v3.5/memory-artifacts/{binding}/genesis?{self.scope_query}",
                None,
            )
            canonical = genesis["canonical"]
            snapshot = ImmutableMemorySnapshot.create(
                canonical["tenant_id"],
                canonical["domain"],
                canonical["memory_binding"],
                (),
            )
            if snapshot.snapshot_hash != genesis["artifact_hash"]:
                raise RuntimeError("NOX deterministic genesis hash mismatch")
            return snapshot

        raw_artifact = response["artifact"]
        if raw_artifact.get("artifact_uri") == "nox://memory/genesis":
            snapshot = ImmutableMemorySnapshot.create(
                raw_artifact["tenant_id"],
                raw_artifact["domain"],
                raw_artifact["memory_binding"],
                (),
            )
            if snapshot.snapshot_hash != response["artifact_hash"]:
                raise RuntimeError("materialized NOX genesis hash mismatch")
            return snapshot
        artifact = FrozenMemoryArtifact.model_validate(raw_artifact)
        if artifact.status != "active":
            raise RuntimeError("NOX returned a non-active memory artifact")
        return ImmutableMemorySnapshot.create(
            artifact.tenant_id,
            artifact.domain,
            artifact.memory_binding,
            tuple(artifact.experiences),
        )

    def heartbeat(self, status: str, detail: dict[str, Any]) -> None:
        self.request(
            "POST",
            f"/api/v3.5/kee-learner-heartbeat?{self.scope_query}",
            {"status": status, "detail": detail},
        )

    def run_once(self) -> int:
        response = self.request(
            "GET",
            "/api/v3.5/actor-learning-handoffs/pending?"
            + urlencode(
                {
                    "tenant_id": self.config.tenant_id,
                    "domain": self.config.domain,
                    "limit": self.config.batch_size,
                }
            ),
            None,
        )
        handoffs = [ActorLearningHandoff.model_validate(item) for item in response.get("items", [])]
        completed = 0
        for handoff in handoffs:
            snapshot = self.memory_snapshot()
            if handoff.parent_memory_hash != snapshot.snapshot_hash:
                raise RuntimeError(
                    f"handoff {handoff.id} is stale relative to active memory; a new actor handoff is required"
                )
            batch = self.learning.build_atomic_batch(handoff, snapshot)
            self.request(
                "POST",
                f"/api/v3.5/experience-batches?{self.scope_query}",
                batch.model_dump(mode="json"),
            )
            completed += 1
        return completed


def main() -> None:
    parser = argparse.ArgumentParser(description="NOX v3.5 governed experience learner")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    config = WorkerConfig.from_env()
    worker = ExperienceWorker(config)
    while True:
        try:
            completed = worker.run_once()
            worker.heartbeat("healthy", {"completed": completed})
            print(json.dumps({"event": "kee.experience_cycle", "completed": completed}), flush=True)
        except Exception as error:  # fail closed, retain the pending canonical handoff
            try:
                worker.heartbeat("degraded", {"error": str(error)[:2048]})
            except Exception:
                pass
            print(
                json.dumps({"event": "kee.experience_cycle_failed", "error": str(error)}),
                file=sys.stderr,
                flush=True,
            )
            if args.once:
                raise
        if args.once:
            return
        time.sleep(config.interval_seconds)


if __name__ == "__main__":
    main()
