from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def now_utc() -> datetime:
    return datetime.now(UTC)


class KEEModel(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)


class StoredModel(KEEModel):
    id: str = Field(default_factory=lambda: new_id("obj"))
    created_at: datetime = Field(default_factory=now_utc)
    updated_at: datetime = Field(default_factory=now_utc)
