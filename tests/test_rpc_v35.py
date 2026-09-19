from datetime import UTC, datetime
from pathlib import Path

import pytest

from llm_kee.rpc import ENGINE_ID, PROTOCOL_SCHEMA_HASH, EngineRpcRequest, RpcError, invoke


def rpc_request(**command_overrides):
    command = {
        "protocol_version": "3.5", "command_id": "cmd-1", "run_id": "run-1",
        "adapter_id": "kee", "capability": "learner", "operation": "propose",
        "input": {"feedback": "qualify the rule"}, "input_hash": "d" * 64,
        "idempotency_key": "proposal-once", "fencing": {}, "lease_token": 3,
        "deadline": int(datetime.now(UTC).timestamp() * 1000) + 60_000,
    }
    command.update(command_overrides)
    return EngineRpcRequest(protocol_version="3.5", protocol_schema_hash=PROTOCOL_SCHEMA_HASH, engine=ENGINE_ID, command=command)


def test_learner_rpc_creates_proposal_only_and_replays(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LLM_KEE_REQUIRE_LLM", "false")
    first = invoke(rpc_request(), tmp_path)
    second = invoke(rpc_request(), tmp_path)
    assert first.reply.output["status"] == "pending_evaluation"
    assert first.reply.output["activation_allowed"] is False
    assert first.replayed is False
    assert second.replayed is True


def test_learner_rpc_rejects_executor_authority(tmp_path: Path):
    request = rpc_request(capability="executor", operation="execute")
    with pytest.raises(RpcError, match="only accepts"):
        invoke(request, tmp_path)
