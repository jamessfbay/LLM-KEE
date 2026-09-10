from pydantic import Field
from typing import Literal

from llm_kee.models.base import StoredModel, new_id
from llm_kee.models.enums import EvaluationDecision, EvaluatorType


class EvaluationResult(StoredModel):
    contract_version: Literal["evaluation-observation/2.0"] = "evaluation-observation/2.0"
    id: str = Field(default_factory=lambda: new_id("eval"))
    proposal_id: str
    evaluator_type: EvaluatorType
    evaluator_name: str | None = None
    weight: float = Field(default=1.0, ge=0.0)
    score: float = Field(ge=0.0, le=1.0)
    decision: EvaluationDecision
    concerns: list[str] = Field(default_factory=list)
    recommended_changes: list[str] = Field(default_factory=list)
    criterion: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    scorer_version: str = "legacy-screening/1"
    score_semantics: Literal["ordinal_screening_not_probability"] = "ordinal_screening_not_probability"


class AggregatedEvaluation(StoredModel):
    id: str = Field(default_factory=lambda: new_id("agg"))
    proposal_id: str
    final_score: float = Field(ge=0.0, le=1.0)
    agreement_level: float = Field(ge=0.0, le=1.0)
    recommendation: EvaluationDecision
    evaluator_count: int
    concerns: list[str] = Field(default_factory=list)
