from pydantic import Field

from llm_kee.models.base import StoredModel, new_id
from llm_kee.models.enums import LearningDecisionType


class LearningDecision(StoredModel):
    id: str = Field(default_factory=lambda: new_id("dec"))
    proposal_id: str
    decision: LearningDecisionType
    reason: str
    final_score: float = Field(ge=0.0, le=1.0)
    required_actions: list[str] = Field(default_factory=list)
