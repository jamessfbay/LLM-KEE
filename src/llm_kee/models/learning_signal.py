from typing import Any

from pydantic import Field

from llm_kee.models.base import StoredModel, new_id
from llm_kee.models.enums import SignalType


class LearningSignal(StoredModel):
    id: str = Field(default_factory=lambda: new_id("sig"))
    signal_type: SignalType
    source_id: str | None = None
    summary: str
    payload: dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(ge=0, le=10, default=5)
