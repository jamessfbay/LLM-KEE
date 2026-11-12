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
    assert captured["max_completion_tokens"] == 16384
    assert "exactly match a source_url" in captured["messages"][0]["content"]


def test_openai_generation_retries_invalid_schema_output_with_feedback(monkeypatch):
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            content = {"summary": "Missing its required identifier."} if len(calls) == 1 else _payload()["demo_output"]
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(content)))],
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
    assert len(calls) == 2
    correction = json.loads(calls[1]["messages"][1]["content"])
    assert correction["validation_error"] == "'record_id' is a required property"
    assert correction["previous_invalid_output"] == {"summary": "Missing its required identifier."}


def test_request_can_route_structured_generation_to_configured_gemini_model(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(
                {
                    "candidates": [
                        {"content": {"parts": [{"text": json.dumps(_payload()["demo_output"])}]}}
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 20,
                        "candidatesTokenCount": 8,
                        "totalTokenCount": 28,
                    },
                }
            ).encode()

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data.decode())
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test-model")
    monkeypatch.setattr("llm_kee.actions.structured_generation.urlopen", fake_urlopen)
    payload = _payload()
    payload["generation_provider"] = "gemini"

    result = StructuredGenerationService(model="openai-default", require_llm=True).execute(
        "structured_generation", payload
    )

    assert result["generation"]["provider"] == "gemini"
    assert result["generation"]["model"] == "gemini-test-model"
    assert result["generation"]["usage"] == {
        "input_tokens": 20,
        "output_tokens": 8,
        "total_tokens": 28,
    }
    assert captured["url"].endswith("/models/gemini-test-model:generateContent")
    assert captured["headers"]["X-goog-api-key"] == "gemini-test-key"
    assert captured["body"]["generationConfig"]["responseMimeType"] == "application/json"
    assert "maxLength" not in json.dumps(captured["body"]["generationConfig"]["responseJsonSchema"])


def test_explicit_gemini_route_requires_gemini_configuration(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    payload = _payload()
    payload["generation_provider"] = "gemini"

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY and GEMINI_MODEL"):
        StructuredGenerationService(require_llm=False).execute("structured_generation", payload)


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
    assert artifact.content["generation"]["prompt_version"] == "structured-generation-v2"
    assert artifact.evidence_ids == ["evidence_1"]
