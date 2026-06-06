import hashlib
from pathlib import Path

from llm_kee.models import LearningSignal, MonitorEvent, MonitorSnapshot, SignalType
from llm_kee.storage import KEEStore


IGNORED_DIRS = {".git", ".venv", ".llm_kee", "__pycache__", ".pytest_cache"}
IGNORED_SUFFIXES = {".pyc", ".pyo"}


class MonitorService:
    def __init__(self, store: KEEStore) -> None:
        self.store = store

    def scan(self, path: Path | str) -> list[LearningSignal]:
        root = Path(path).expanduser().resolve()
        current_files = self._collect_files(root)
        previous = self._latest_snapshot(root)
        events = self._diff(root, previous.files if previous else {}, current_files)
        signals = [self._signal_from_event(event) for event in events]
        for event, signal in zip(events, signals, strict=True):
            signal = self.store.signals.upsert(signal)
            event.signal_id = signal.id
            event = self.store.monitor_events.upsert(event)
        snapshot = MonitorSnapshot(
            root_path=str(root),
            files=current_files,
            event_ids=[event.id for event in events],
        )
        self.store.monitor_snapshots.upsert(snapshot)
        return signals

    def diff(self, path: Path | str) -> list[MonitorEvent]:
        root = Path(path).expanduser().resolve()
        current_files = self._collect_files(root)
        previous = self._latest_snapshot(root)
        return self._diff(root, previous.files if previous else {}, current_files)

    def _latest_snapshot(self, root: Path) -> MonitorSnapshot | None:
        snapshots = [
            snapshot
            for snapshot in self.store.monitor_snapshots.list()
            if snapshot.root_path == str(root)
        ]
        if not snapshots:
            return None
        return sorted(snapshots, key=lambda item: item.created_at)[-1]

    def _collect_files(self, root: Path) -> dict[str, dict]:
        if root.is_file():
            return {root.name: self._file_record(root)}
        records: dict[str, dict] = {}
        for file_path in sorted(root.rglob("*")):
            if not file_path.is_file() or self._ignored(file_path, root):
                continue
            records[str(file_path.relative_to(root))] = self._file_record(file_path)
        return records

    def _file_record(self, file_path: Path) -> dict:
        stat = file_path.stat()
        return {
            "hash": self._hash(file_path),
            "mtime": stat.st_mtime,
            "size": stat.st_size,
            "suffix": file_path.suffix.lower(),
        }

    def _diff(
        self,
        root: Path,
        previous: dict[str, dict],
        current: dict[str, dict],
    ) -> list[MonitorEvent]:
        events: list[MonitorEvent] = []
        for path, record in current.items():
            old = previous.get(path)
            if old is None:
                events.append(
                    MonitorEvent(
                        root_path=str(root),
                        path=path,
                        event_type="created",
                        new_hash=record["hash"],
                        new_mtime=record["mtime"],
                        metadata={"suffix": record.get("suffix"), "size": record.get("size")},
                    )
                )
            elif old.get("hash") != record.get("hash"):
                events.append(
                    MonitorEvent(
                        root_path=str(root),
                        path=path,
                        event_type="modified",
                        old_hash=old.get("hash"),
                        new_hash=record.get("hash"),
                        old_mtime=old.get("mtime"),
                        new_mtime=record.get("mtime"),
                        metadata={"suffix": record.get("suffix"), "size": record.get("size")},
                    )
                )
        for path, old in previous.items():
            if path not in current:
                events.append(
                    MonitorEvent(
                        root_path=str(root),
                        path=path,
                        event_type="deleted",
                        old_hash=old.get("hash"),
                        old_mtime=old.get("mtime"),
                        metadata={"suffix": old.get("suffix"), "size": old.get("size")},
                    )
                )
        return events

    def _signal_from_event(self, event: MonitorEvent) -> LearningSignal:
        signal_type = self._signal_type(event)
        return LearningSignal(
            signal_type=signal_type,
            source_id=event.path,
            summary=f"Monitor detected {event.event_type} file: {event.path}",
            payload=event.model_dump(mode="json"),
            priority=8 if event.event_type in {"created", "modified"} else 5,
        )

    def _signal_type(self, event: MonitorEvent) -> SignalType:
        text = f"{event.path} {event.event_type}".lower()
        if "conflict" in text:
            return SignalType.GRAPH_CONFLICT
        if "schema" in text or "ontology" in text:
            return SignalType.SCHEMA_GAP
        if "low_confidence" in text or "uncertain" in text:
            return SignalType.LOW_CONFIDENCE_CLAIM
        if "orphan" in text:
            return SignalType.ORPHAN_NODE
        return SignalType.NEW_SOURCE

    def _ignored(self, file_path: Path, root: Path) -> bool:
        relative = file_path.relative_to(root)
        parts = set(relative.parts)
        if parts & IGNORED_DIRS:
            return True
        if file_path.name == ".env" or file_path.name.startswith(".env."):
            return True
        return file_path.suffix in IGNORED_SUFFIXES

    def _hash(self, file_path: Path) -> str:
        digest = hashlib.sha256()
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
