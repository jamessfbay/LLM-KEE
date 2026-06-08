from pathlib import Path
import os
import tomllib

from pydantic import BaseModel, Field


class JudgeConfig(BaseModel):
    name: str
    provider: str = "mock"
    model: str = "mock-judge"
    enabled: bool = True
    weight: float = Field(default=1.0, ge=0.0)
    api_key_env: str | None = None


class EvaluationConfig(BaseModel):
    enable_rule_engine: bool = True
    enable_evidence_checker: bool = True
    enable_conflict_checker: bool = True
    enable_behavior_signal: bool = True
    min_llm_judge_agreement: float = Field(default=0.67, ge=0.0, le=1.0)
    judges: list[JudgeConfig] = Field(
        default_factory=lambda: [
            JudgeConfig(name="openai_mock", provider="mock", model="gpt-style-judge"),
            JudgeConfig(name="anthropic_mock", provider="mock", model="claude-style-judge"),
            JudgeConfig(name="gemini_mock", provider="mock", model="gemini-style-judge"),
        ]
    )


class KGIntegrationConfig(BaseModel):
    enable_direct_adapter: bool = False
    workspace: Path | None = None
    project_path: Path | None = None


class SkillsConfig(BaseModel):
    register_defaults: bool = True


class ActionsConfig(BaseModel):
    register_defaults: bool = True


class WorkflowsConfig(BaseModel):
    record_step_outputs: bool = True


class MAPEConfig(BaseModel):
    enabled: bool = True


class EvolutionConfig(BaseModel):
    enabled: bool = True
    require_evidence_for_versions: bool = False


class IntentConfig(BaseModel):
    register_defaults: bool = True
    detector_provider: str = "deterministic"


class FailuresConfig(BaseModel):
    enabled: bool = True


class ImprovementsConfig(BaseModel):
    enabled: bool = True
    require_review: bool = True


class LoopsConfig(BaseModel):
    knowledge_enabled: bool = True
    skill_enabled: bool = True
    agent_enabled: bool = True


class Settings(BaseModel):
    workspace: Path = Path(".")
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    kg: KGIntegrationConfig = Field(default_factory=KGIntegrationConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    actions: ActionsConfig = Field(default_factory=ActionsConfig)
    workflows: WorkflowsConfig = Field(default_factory=WorkflowsConfig)
    mape: MAPEConfig = Field(default_factory=MAPEConfig)
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
    intent: IntentConfig = Field(default_factory=IntentConfig)
    failures: FailuresConfig = Field(default_factory=FailuresConfig)
    improvements: ImprovementsConfig = Field(default_factory=ImprovementsConfig)
    loops: LoopsConfig = Field(default_factory=LoopsConfig)


def load_settings() -> Settings:
    workspace = Path(os.getenv("LLM_KEE_WORKSPACE", ".")).expanduser().resolve()
    config_path = Path(os.getenv("LLM_KEE_CONFIG", workspace / "llm_kee.toml")).expanduser()
    data = _read_toml(config_path)
    settings_data = _normalize_settings_data(data)
    settings = Settings.model_validate(settings_data)
    settings.workspace = Path(
        os.getenv("LLM_KEE_WORKSPACE", str(settings.workspace))
    ).expanduser().resolve()
    if os.getenv("LLM_KEE_KG_ENABLE_DIRECT_ADAPTER") is not None:
        settings.kg.enable_direct_adapter = _bool(os.getenv("LLM_KEE_KG_ENABLE_DIRECT_ADAPTER"))
    if os.getenv("LLM_KG_WORKSPACE"):
        settings.kg.workspace = Path(os.environ["LLM_KG_WORKSPACE"]).expanduser().resolve()
    if os.getenv("LLM_KG_PROJECT_PATH"):
        settings.kg.project_path = Path(os.environ["LLM_KG_PROJECT_PATH"]).expanduser().resolve()
    return settings


def _read_toml(path: Path) -> dict:
    if not path.exists():
        return {}
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _normalize_settings_data(data: dict) -> dict:
    normalized = dict(data)
    storage = normalized.pop("storage", {})
    if "workspace" in storage and "workspace" not in normalized:
        normalized["workspace"] = storage["workspace"]
    return normalized


def _bool(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}
