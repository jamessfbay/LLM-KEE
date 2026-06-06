from pydantic import Field

from llm_kee.models.base import StoredModel, new_id
from llm_kee.models.enums import ReviewStatus


class LearnedPattern(StoredModel):
    id: str = Field(default_factory=lambda: new_id("pat"))
    pattern_type: str
    name: str
    description: str
    examples: list[str] = Field(default_factory=list)
    source_trace_ids: list[str] = Field(default_factory=list)
    status: ReviewStatus = ReviewStatus.PENDING
