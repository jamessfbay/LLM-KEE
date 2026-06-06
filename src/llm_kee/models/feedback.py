from typing import Any

from pydantic import Field

from llm_kee.models.base import StoredModel, new_id
from llm_kee.models.enums import FeedbackStatus, FeedbackType, TargetType


class UserFeedbackCreate(StoredModel):
    id: str = Field(default_factory=lambda: new_id("fb"))
    target_type: TargetType
    target_id: str | None = None
    feedback_type: FeedbackType
    old_value: dict[str, Any] | None = None
    new_value: dict[str, Any] | None = None
    comment: str | None = None
    source_user: str | None = None
    status: FeedbackStatus = FeedbackStatus.RECEIVED


class UserFeedback(UserFeedbackCreate):
    pass
