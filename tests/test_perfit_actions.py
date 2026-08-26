import pytest

from llm_kee.actions.perfit import PerfitActionService
from llm_kee.config import PerfitLLMConfig, Settings
from llm_kee.services import KEEEngine


def _opportunity_payload():
    return {
        "opportunity": {
            "id": "opp_1",
            "problem": "revenue teams cannot reliably qualify emerging demand",
        },
        "hypotheses": [{"id": "h1"}, {"id": "h2"}],
        "evidence_ids": ["ev_1", "ev_2"],
    }


def test_generate_validation_questions_returns_exactly_five_grounded_questions(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = PerfitActionService().execute("generate_validation_questions", _opportunity_payload())

    assert result["opportunity_id"] == "opp_1"
    assert len(result["questions"]) == 5
    assert len({question["text"] for question in result["questions"]}) == 5
    assert all(question["evidence_ids"] == ["ev_1", "ev_2"] for question in result["questions"])
    assert result["generation"]["provider"] == "deterministic_demo"


def test_generate_outreach_emails_maps_one_email_to_each_question(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    service = PerfitActionService()
    questions = service.execute("generate_validation_questions", _opportunity_payload())["questions"]
    result = service.execute(
        "generate_outreach_emails",
        {**_opportunity_payload(), "questions": questions},
    )

    assert len(result["emails"]) == 5
    assert {email["question_id"] for email in result["emails"]} == {
        question["id"] for question in questions
    }


def test_production_mode_requires_configured_llm(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    service = PerfitActionService(model="gpt-test", require_llm=True)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        service.execute("generate_validation_questions", _opportunity_payload())


def test_validation_rejects_invented_hypothesis_and_evidence_ids():
    service = PerfitActionService()
    content = service._mock("generate_validation_questions", _opportunity_payload())
    content["questions"][0]["hypothesis_id"] = "invented_hypothesis"

    with pytest.raises(ValueError, match="hypothesis id"):
        service._validate("generate_validation_questions", content, _opportunity_payload())

    content = service._mock("generate_validation_questions", _opportunity_payload())
    content["questions"][0]["evidence_ids"] = ["invented_evidence"]
    with pytest.raises(ValueError, match="evidence ids"):
        service._validate("generate_validation_questions", content, _opportunity_payload())


def test_engine_persists_perfit_artifact_and_records_generation_metadata(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    engine = KEEEngine(
        Settings(
            workspace=tmp_path,
            perfit_llm=PerfitLLMConfig(provider="openai", model="", require_llm=False),
        )
    )

    run = engine.run_action("generate_validation_questions", _opportunity_payload())

    assert run.status == "completed"
    artifact = engine.store.action_artifacts.get(run.artifact_ids[0])
    assert len(artifact.content["questions"]) == 5
    assert artifact.content["generation"]["prompt_version"] == "perfit-actions-v1"
    assert artifact.evidence_ids == ["ev_1", "ev_2"]


def test_engine_records_failed_action_when_llm_is_required_but_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    engine = KEEEngine(
        Settings(
            workspace=tmp_path,
            perfit_llm=PerfitLLMConfig(provider="openai", model="gpt-test", require_llm=True),
        )
    )

    run = engine.run_action("generate_validation_questions", _opportunity_payload())

    assert run.status == "failed"
    assert "OPENAI_API_KEY" in run.output["error"]
