import json
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_kee.cli import _execute_runtime_command
from llm_kee.runtime_protocol import (
    OperationReceiptStore,
    RuntimeCommand,
    RuntimeEmitter,
)


def test_runtime_fixture_and_ndjson_event(tmp_path: Path):
    payload = json.loads((Path(__file__).parent / "fixtures" / "runtime_command_v1.json").read_text())
    command = RuntimeCommand.model_validate(payload)
    stream = StringIO()
    event = RuntimeEmitter(command, stream).emit("step.started", "running", payload={"phase": "test"})
    assert event.protocol_version == "1.0"
    emitted = json.loads(stream.getvalue())
    assert emitted["run_id"] == "run_contract"
    assert "error" not in emitted


def test_operation_receipt_is_idempotent_and_rejects_hash_conflict(tmp_path: Path):
    payload = json.loads((Path(__file__).parent / "fixtures" / "runtime_command_v1.json").read_text())
    command = RuntimeCommand.model_validate(payload)
    store = OperationReceiptStore(tmp_path, ".llm_kee")
    receipt = store.begin(command)
    store.complete(receipt, "succeeded", {"artifact_path": "/tmp/result.json"})
    assert store.begin(command).output["artifact_path"] == "/tmp/result.json"

    conflicting = command.model_copy(update={"input_hash": "different"})
    with pytest.raises(ValueError, match="different input hash"):
        store.begin(conflicting)


def test_action_run_supports_runtime_event_execution(tmp_path: Path):
    payload_path = tmp_path / "action.json"
    payload_path.write_text(json.dumps({"projectId": "project-1"}), encoding="utf-8")
    args = SimpleNamespace(
        command="action",
        action_command="run",
        action_type="generate_intelligence_pack",
        json_file=str(payload_path),
    )
    run = SimpleNamespace(
        id="action-run-1",
        status="completed",
        artifact_ids=["artifact-1"],
        workflow_run_id="workflow-1",
    )
    engine = SimpleNamespace(run_action=lambda action_type, payload: run)
    emitted = []
    emitter = SimpleNamespace(emit=lambda *values, **kwargs: emitted.append((values, kwargs)))

    output, status = _execute_runtime_command(args, engine, None, emitter)

    assert status == "succeeded"
    assert output["action_run_ids"] == ["action-run-1"]
    assert output["artifact_ids"] == ["artifact-1"]
    assert emitted[0][0] == ("step.progress", "running")
