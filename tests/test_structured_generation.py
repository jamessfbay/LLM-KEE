import json
import sys
from types import SimpleNamespace

import pytest

from llm_kee.actions.structured_generation import StructuredGenerationService
from llm_kee.config import GenerationLLMConfig, Settings
from llm_kee.services import KEEEngine


def _payload():
    return {
        "system_instruction": "Return a caller-defined research artifact.",
        "task_instruction": "Summarize the supplied record.",
        "input": {"record": {"id": "record_1", "summary": "A verified fact."}},
        "output_json_schema": {
            "type": "object",
            "required": ["record_id", "summary"],
            "additionalProperties": False,
            "properties": {
                "record_id": {"const": "record_1"},
                "summary": {"type": "string", "minLength": 1},
            },
        },
        "demo_output": {"record_id": "record_1", "summary": "A verified fact."},
        "evidence_ids": ["evidence_1"],
    }


def test_structured_generation_validates_caller_owned_demo_output(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = StructuredGenerationService().execute("structured_generation", _payload())

    assert result["record_id"] == "record_1"
    assert result["generation"]["provider"] == "caller_demo_output"
    assert len(result["generation"]["schema_hash"]) == 64
    assert result["generation"]["usage"] is None


def test_structured_generation_rejects_output_outside_caller_schema(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    payload = _payload()
    payload["demo_output"] = {"record_id": "invented", "summary": "A verified fact."}

    with pytest.raises(ValueError, match="schema validation"):
        StructuredGenerationService().execute("structured_generation", payload)


def test_production_mode_requires_configured_llm(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    service = StructuredGenerationService(model="gpt-test", require_llm=True)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        service.execute("structured_generation", _payload())


def test_openai_generation_uses_model_default_temperature(monkeypatch):
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(_payload()["demo_output"])))],
            )

    class FakeOpenAI:
        def __init__(self):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))

    result = StructuredGenerationService(model="gpt-5-mini", require_llm=True).execute(
        "structured_generation", _payload()
    )

    assert result["record_id"] == "record_1"
    assert "temperature" not in captured
    assert "exactly match a source_url" in captured["messages"][0]["content"]


def test_engine_persists_generic_structured_artifact(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    engine = KEEEngine(
        Settings(
            workspace=tmp_path,
            generation_llm=GenerationLLMConfig(provider="openai", model="", require_llm=False),
        )
    )

    run = engine.run_action("structured_generation", _payload())

    assert run.status == "completed"
    artifact = engine.store.action_artifacts.get(run.artifact_ids[0])
    assert artifact.artifact_type == "structured_generation"
    assert artifact.content["record_id"] == "record_1"
    assert artifact.content["generation"]["prompt_version"] == "structured-generation-v1"
    assert artifact.evidence_ids == ["evidence_1"]
