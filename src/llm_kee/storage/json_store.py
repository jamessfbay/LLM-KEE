import json
from pathlib import Path
import tempfile
from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class JsonRepository(Generic[T]):
    def __init__(self, path: Path, model_type: type[T]) -> None:
        self.path = path
        self.model_type = model_type
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[T]:
        if not self.path.exists():
            return []
        raw = self.path.read_text(encoding="utf-8") or "[]"
        records = self._loads(raw)
        return [self.model_type.model_validate(record) for record in records]

    def get(self, item_id: str) -> T | None:
        return next((item for item in self.list() if item.id == item_id), None)

    def upsert(self, item: T) -> T:
        items = self.list()
        replaced = False
        for index, existing in enumerate(items):
            if existing.id == item.id:
                items[index] = item
                replaced = True
                break
        if not replaced:
            items.append(item)
        self._write(items)
        return item

    def _write(self, items: list[T]) -> None:
        payload = [item.model_dump(mode="json") for item in items]
        encoded = json.dumps(payload, indent=2, ensure_ascii=False)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=self.path.parent,
            delete=False,
        ) as handle:
            handle.write(encoded)
            temp_path = Path(handle.name)
        temp_path.replace(self.path)

    def _loads(self, raw: str) -> list:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            decoder = json.JSONDecoder()
            records, _ = decoder.raw_decode(raw)
            return records
