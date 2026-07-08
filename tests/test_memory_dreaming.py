import json

import pytest

from llm_kee.config import Settings
from llm_kee.services import KEEEngine


def engine(tmp_path):
    return KEEEngine(Settings(workspace=tmp_path))


def test_memory_draft_apply_and_search(tmp_path):
    kee = engine(tmp_path)
    draft = kee.draft_memory(
        {
            "scope": "project",
            "subject_id": "permit-demo",
            "proposed_content": "# Project Memory\n\nPermit review happened today.",
            "rationale": "Capture stable project memory.",
        }
    )

    applied = kee.apply_memory_draft(draft.id)
    assert applied.status == "applied"

    memory = kee.read_memory("project", "permit-demo")
    assert "Permit review happened" in memory["content"]
    assert "today" not in memory["content"].lower()
    assert kee.search_memory("Permit review")["matches"]
    assert kee.search_memory("project:other permit_history")["matches"]


def test_memory_draft_hash_conflict(tmp_path):
    kee = engine(tmp_path)
    draft = kee.draft_memory(
        {
            "scope": "org",
            "proposed_content": "# Organization Memory\n\nFirst proposal.",
            "rationale": "Initial draft.",
        }
    )
    path = tmp_path / ".llm_kee" / "memory" / draft.target_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("Changed outside draft.", encoding="utf-8")

    with pytest.raises(ValueError, match="hash conflict"):
        kee.apply_memory_draft(draft.id)


def test_dream_run_creates_diary_and_reviewable_proposal(tmp_path):
    kee = engine(tmp_path)
    audit_file = tmp_path / "audit.json"
    audit_file.write_text(
        json.dumps(
            [
                {
                    "id": "trace-a",
                    "decision_id": "decision-a",
                    "evidence": {"payload": {"missing_data": ["No verified raw-source evidence found for permit history."]}},
                    "kg": {"payload": {"used_claim_count": 0}},
                    "review": {"status": "approved"},
                },
                {
                    "id": "trace-b",
                    "decision_id": "decision-b",
                    "evidence": {"payload": {"missing_data": ["No verified raw-source evidence found for permit history."]}},
                    "kg": {"payload": {"used_claim_count": 0}},
                    "review": {"status": "required"},
                },
            ]
        ),
        encoding="utf-8",
    )

    result = kee.run_dream({"audit_file": str(audit_file), "token_budget": 1000})
    assert result["run"].status == "completed"
    assert result["diary"].narrative.startswith("# Dream Diary")
    assert result["proposals"]

    proposal = result["proposals"][0]
    reviewed = kee.review_dream_proposal(proposal.id, approve=True, notes="Approved in test.")
    assert reviewed.status == "approved"
    assert kee.read_memory("org")["content"]


def test_dream_scheduler_run_once_records_state(tmp_path):
    kee = KEEEngine(Settings(workspace=tmp_path))
    audit_file = tmp_path / "audit.json"
    audit_file.write_text(
        json.dumps(
            [
                {
                    "id": "trace-1",
                    "decision_id": "decision-1",
                    "evidence": {"payload": {"missing_data": ["Missing city council agenda"]}},
                    "kg": {"payload": {"used_claim_count": 0}},
                    "review": {"status": "required"},
                }
            ]
        ),
        encoding="utf-8",
    )

    result = kee.run_dream_scheduler_once({"audit_file": str(audit_file), "token_budget": 1000})
    status = kee.dream_scheduler_status()

    assert result["run"].id == status["last_dream_run_id"]
    assert status["status"] == "idle"
    assert status["last_run"]
    assert status["next_due"]
    assert status["locks"] == []


def test_memory_apply_respects_active_lock(tmp_path):
    kee = KEEEngine(Settings(workspace=tmp_path))
    draft = kee.draft_memory(
        {
            "scope": "org",
            "proposed_content": "# Organization Memory\n\nLocked update.\n",
            "rationale": "Exercise lock conflict.",
        }
    )
    locks = tmp_path / ".llm_kee" / "locks"
    locks.mkdir(parents=True, exist_ok=True)
    lock_name = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in f"memory-{draft.target_path}")[:120]
    (locks / f"{lock_name}.lock").write_text(
        json.dumps(
            {
                "owner": "test",
                "target": draft.target_path,
                "base_hash": draft.base_hash,
                "created_at": "2026-01-01T00:00:00+00:00",
                "expires_at": "2999-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Lock conflict"):
        kee.apply_memory_draft(draft.id)
