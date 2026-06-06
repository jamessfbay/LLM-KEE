from typing import Any

from pydantic import Field

from llm_kee.models.base import StoredModel, new_id
from llm_kee.models.enums import ReviewStatus


class SchemaSuggestion(StoredModel):
    id: str = Field(default_factory=lambda: new_id("schema"))
    suggestion_type: str
    element_name: str
    rationale: str
    examples: list[str] = Field(default_factory=list)
    proposed_schema: dict[str, Any] = Field(default_factory=dict)
    status: ReviewStatus = ReviewStatus.PENDING
