from __future__ import annotations

from decimal import Decimal
import hashlib
import json
import math
from typing import Any, TypeVar

from pydantic import BaseModel


ModelT = TypeVar("ModelT", bound=BaseModel)


def canonical_hash(value: Any) -> str:
    """Hash the same UTF-8 canonical JSON representation as nox-kernel."""

    def number(item: float | int) -> str:
        if isinstance(item, int):
            return str(item)
        if not math.isfinite(item):
            raise ValueError("non-finite value cannot be hashed")
        if item == 0:
            return "0"
        rendered = repr(item).lower()
        absolute = abs(item)
        if 1e-6 <= absolute < 1e21:
            if "e" in rendered:
                return format(Decimal(rendered), "f")
            return rendered[:-2] if rendered.endswith(".0") else rendered
        if "e" not in rendered:
            rendered = format(Decimal(rendered).normalize(), "e")
        mantissa, exponent = rendered.split("e")
        if mantissa.endswith(".0"):
            mantissa = mantissa[:-2]
        exponent_value = int(exponent)
        exponent_text = f"+{exponent_value}" if exponent_value >= 0 else str(exponent_value)
        return f"{mantissa}e{exponent_text}"

    def canonical(item: Any) -> str:
        if item is None:
            return "null"
        if item is True:
            return "true"
        if item is False:
            return "false"
        if isinstance(item, (int, float)):
            return number(item)
        if isinstance(item, str):
            return json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        if isinstance(item, (list, tuple)):
            return "[" + ",".join(canonical(value) for value in item) + "]"
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("canonical JSON object keys must be strings")
            keys = sorted(item, key=lambda key: key.encode("utf-16-be", "surrogatepass"))
            return "{" + ",".join(f"{canonical(key)}:{canonical(item[key])}" for key in keys) + "}"
        raise TypeError(f"unsupported canonical JSON value: {type(item).__name__}")

    return hashlib.sha256(canonical(value).encode()).hexdigest()


def model_digest(model: Any, digest_field: str) -> str:
    payload = model.model_dump(mode="json")
    payload.pop(digest_field, None)
    return canonical_hash(payload)


def seal_model(model_type: type[ModelT], digest_field: str, **values: Any) -> ModelT:
    """Construct and validate a self-hashed Runtime contract record."""
    draft = model_type.model_construct(**values, **{digest_field: ""})
    values[digest_field] = model_digest(draft, digest_field)
    return model_type.model_validate(values)
