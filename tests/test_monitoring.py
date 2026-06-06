from pathlib import Path

from llm_kee.config import Settings
from llm_kee.models import SignalType
from llm_kee.services import KEEEngine


def test_monitor_first_scan_creates_new_source_signals(tmp_path: Path) -> None:
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    (source_dir / "doc.md").write_text("# Demo\n", encoding="utf-8")

    engine = KEEEngine(Settings(workspace=tmp_path / "kee"))
    signals = engine.monitor_path(source_dir)

    assert len(signals) == 1
    assert signals[0].signal_type == SignalType.NEW_SOURCE
    assert engine.store.monitor_snapshots.list()[0].root_path == str(source_dir.resolve())


def test_monitor_second_scan_without_changes_has_no_signals(tmp_path: Path) -> None:
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    (source_dir / "doc.md").write_text("# Demo\n", encoding="utf-8")

    engine = KEEEngine(Settings(workspace=tmp_path / "kee"))
    assert len(engine.monitor_path(source_dir)) == 1
    assert engine.monitor_path(source_dir) == []


def test_monitor_detects_modified_file(tmp_path: Path) -> None:
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    doc = source_dir / "doc.md"
    doc.write_text("# Demo\n", encoding="utf-8")

    engine = KEEEngine(Settings(workspace=tmp_path / "kee"))
    engine.monitor_path(source_dir)
    doc.write_text("# Demo\nChanged\n", encoding="utf-8")
    signals = engine.monitor_path(source_dir)

    assert len(signals) == 1
    event = engine.store.monitor_events.list()[-1]
    assert event.event_type == "modified"
    assert event.old_hash != event.new_hash


def test_monitor_ignores_env_and_local_runtime_dirs(tmp_path: Path) -> None:
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    (source_dir / ".env").write_text("SECRET=value\n", encoding="utf-8")
    (source_dir / ".env.local").write_text("SECRET=value\n", encoding="utf-8")
    (source_dir / ".llm_kee").mkdir()
    (source_dir / ".llm_kee" / "data.json").write_text("{}", encoding="utf-8")
    (source_dir / ".venv").mkdir()
    (source_dir / ".venv" / "pyvenv.cfg").write_text("", encoding="utf-8")
    (source_dir / "__pycache__").mkdir()
    (source_dir / "__pycache__" / "x.pyc").write_bytes(b"cache")

    engine = KEEEngine(Settings(workspace=tmp_path / "kee"))
    assert engine.monitor_path(source_dir) == []
