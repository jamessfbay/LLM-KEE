from pydantic import Field

from llm_kee.models.base import StoredModel, new_id


class ReasoningStep(StoredModel):
    id: str = Field(default_factory=lambda: new_id("step"))
    order: int
    description: str
    claim_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class ReasoningTraceCreate(StoredModel):
    id: str = Field(default_factory=lambda: new_id("trace"))
    question: str
    final_answer: str
    reasoning_steps: list[ReasoningStep] = Field(default_factory=list)
    used_claim_ids: list[str] = Field(default_factory=list)
    used_relation_ids: list[str] = Field(default_factory=list)
    used_evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    user_reaction: str | None = None
    reusable: bool = False


class ReasoningTrace(ReasoningTraceCreate):
    pass
