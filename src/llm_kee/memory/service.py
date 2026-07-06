from __future__ import annotations

from collections import Counter
from datetime import UTC, date, datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from llm_kee.models import (
    DreamDiary,
    DreamInsight,
    DreamProposal,
    DreamRun,
    MemoryDraft,
    MemoryFile,
)
from llm_kee.models.base import now_utc
from llm_kee.storage import KEEStore


class MemoryDreamingService:
    def __init__(self, store: KEEStore, workspace: Path) -> None:
        self.store = store
        self.workspace = workspace
        self.root = workspace / ".llm_kee" / "memory"
        self.root.mkdir(parents=True, exist_ok=True)
        for relative in ("org", "projects", "agents", "scratch", ".drafts", ".dreams"):
            (self.root / relative).mkdir(parents=True, exist_ok=True)

    def list_memory(self) -> dict[str, Any]:
        self.refresh_manifest()
        return {
            "files": self.store.memory_files.list(),
            "drafts": self.store.memory_drafts.list(),
            "dream_runs": self.store.dream_runs.list(),
            "dream_insights": self.store.dream_insights.list(),
            "dream_proposals": self.store.dream_proposals.list(),
            "dream_diaries": self.store.dream_diaries.list(),
            "memory_root": str(self.root),
        }

    def read_memory(self, scope: str, subject_id: str | None = None) -> dict[str, Any]:
        path = self._scope_path(scope, subject_id)
        content = path.read_text(encoding="utf-8") if path.exists() else ""
        memory = self._memory_file(path, scope, subject_id, content)
        return {"file": memory, "content": content}

    def search(self, query: str) -> dict[str, Any]:
        needle = query.strip().lower()
        tokens = [token for token in re.split(r"[^a-z0-9]+", needle) if len(token) > 3]
        matches: list[dict[str, Any]] = []
        for path in sorted(self.root.rglob("*.md")):
            if self._is_internal(path):
                continue
            content = path.read_text(encoding="utf-8")
            haystack = content.lower()
            path_text = path.as_posix().lower()
            if not needle or needle in haystack or needle in path_text or any(token in haystack or token in path_text for token in tokens):
                matches.append(
                    {
                        "path": self._relative(path),
                        "hash": self._hash(content),
                        "excerpt": self._excerpt(content, needle),
                    }
                )
        return {"query": query, "matches": matches}

    def draft(self, payload: dict[str, Any]) -> MemoryDraft:
        scope = str(payload.get("scope") or "org")
        subject_id = payload.get("subject_id")
        path = self._target_path(payload, scope, str(subject_id) if subject_id else None)
        current = path.read_text(encoding="utf-8") if path.exists() else ""
        proposed = self._normalize_relative_dates(str(payload.get("proposed_content") or ""))
        if not proposed:
            proposed = self._append_section(
                current,
                str(payload.get("title") or "Memory Update"),
                str(payload.get("summary") or payload.get("rationale") or "No summary provided."),
            )
        draft = MemoryDraft(
            target_path=self._relative(path),
            base_hash=self._hash(current),
            proposed_content=proposed,
            rationale=str(payload.get("rationale") or "Proposed memory update."),
            metadata={key: value for key, value in payload.items() if key not in {"proposed_content"}},
        )
        self._write_internal_json(".drafts", draft.id, draft.model_dump(mode="json"))
        return self.store.memory_drafts.upsert(draft)

    def apply_draft(self, draft_id: str) -> MemoryDraft:
        draft = self.store.memory_drafts.get(draft_id)
        if not draft:
            raise ValueError(f"Memory draft not found: {draft_id}")
        path = self._safe_path(draft.target_path)
        current = path.read_text(encoding="utf-8") if path.exists() else ""
        current_hash = self._hash(current)
        if current_hash != draft.base_hash:
            raise ValueError(
                f"Memory hash conflict for {draft.target_path}: expected {draft.base_hash}, got {current_hash}"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._normalize_relative_dates(draft.proposed_content), encoding="utf-8")
        draft.status = "applied"
        draft.applied_at = now_utc()
        draft.updated_at = now_utc()
        self.store.memory_drafts.upsert(draft)
        self.refresh_manifest()
        return draft

    def reject_draft(self, draft_id: str) -> MemoryDraft:
        draft = self.store.memory_drafts.get(draft_id)
        if not draft:
            raise ValueError(f"Memory draft not found: {draft_id}")
        draft.status = "rejected"
        draft.rejected_at = now_utc()
        draft.updated_at = now_utc()
        return self.store.memory_drafts.upsert(draft)

    def run_dream(self, payload: dict[str, Any]) -> dict[str, Any]:
        traces = self._load_source_traces(payload)
        run = DreamRun(
            source_trace_ids=[str(trace.get("id") or trace.get("decision_id") or "") for trace in traces],
            token_budget=int(payload.get("token_budget") or 0),
            metadata={
                "trace_count": len(traces),
                "source": payload.get("source", "manual"),
            },
        )
        run = self.store.dream_runs.upsert(run)
        insights = self._build_insights(run.id, traces)
        proposals: list[DreamProposal] = []
        for insight in insights:
            self.store.dream_insights.upsert(insight)
            run.insight_ids.append(insight.id)
            if insight.score >= 3.0:
                proposal = self._proposal_from_insight(run.id, insight)
                proposals.append(self.store.dream_proposals.upsert(proposal))
                run.proposal_ids.append(proposal.id)
        diary = self._diary(run, insights, proposals)
        self.store.dream_diaries.upsert(diary)
        run.diary_id = diary.id
        run.updated_at = now_utc()
        run = self.store.dream_runs.upsert(run)
        self._write_internal_json(".dreams", run.id, {"run": run.model_dump(mode="json"), "diary": diary.model_dump(mode="json")})
        return {"run": run, "insights": insights, "proposals": proposals, "diary": diary}

    def dream_diary(self, dream_run_id: str) -> DreamDiary:
        diary = next((item for item in self.store.dream_diaries.list() if item.dream_run_id == dream_run_id), None)
        if not diary:
            raise ValueError(f"Dream diary not found for run: {dream_run_id}")
        return diary

    def review_dream_proposal(self, proposal_id: str, approve: bool, notes: str | None = None) -> DreamProposal:
        proposal = self.store.dream_proposals.get(proposal_id)
        if not proposal:
            raise ValueError(f"Dream proposal not found: {proposal_id}")
        if approve:
            self.apply_draft(proposal.memory_draft_id)
            proposal.status = "approved"
        else:
            self.reject_draft(proposal.memory_draft_id)
            proposal.status = "rejected"
        proposal.review_notes = notes
        proposal.updated_at = now_utc()
        return self.store.dream_proposals.upsert(proposal)

    def refresh_manifest(self) -> list[MemoryFile]:
        files: list[MemoryFile] = []
        for path in sorted(self.root.rglob("*.md")):
            if self._is_internal(path):
                continue
            content = path.read_text(encoding="utf-8")
            scope, subject_id = self._scope_from_path(path)
            files.append(self.store.memory_files.upsert(self._memory_file(path, scope, subject_id, content)))
        manifest = [item.model_dump(mode="json") for item in files]
        (self.root / ".manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return files

    def _build_insights(self, run_id: str, traces: list[dict[str, Any]]) -> list[DreamInsight]:
        missing_counter: Counter[str] = Counter()
        policy_counter: Counter[str] = Counter()
        kg_empty = 0
        source_ids: set[str] = set()
        for trace in traces:
            source_ids.add(str(trace.get("decision_id") or trace.get("id") or "unknown"))
            missing = self._nested(trace, "evidence", "payload", "missing_data") or self._nested(trace, "event", "context", "missing_data")
            if isinstance(missing, list):
                missing_counter.update(str(item) for item in missing)
            if self._nested(trace, "kg", "payload", "used_claim_count") == 0:
                kg_empty += 1
            review = self._nested(trace, "review", "status") or self._nested(trace, "human_review", "status")
            if review:
                policy_counter.update([str(review)])
        failure_types = Counter(failure.failure_type for failure in self.store.failure_records.list())
        for failure_type, count in failure_types.items():
            missing_counter.update([f"Failure memory repeats: {failure_type}"] * count)

        insights: list[DreamInsight] = []
        if missing_counter:
            summary, count = missing_counter.most_common(1)[0]
            insights.append(
                self._insight(
                    run_id,
                    f"Repeated evidence gap: {summary}",
                    count,
                    len(source_ids),
                    {"kind": "missing_evidence", "top_missing_data": summary},
                )
            )
        if kg_empty:
            insights.append(
                self._insight(
                    run_id,
                    "KG traces frequently have no claim/evidence IDs.",
                    kg_empty,
                    len(source_ids),
                    {"kind": "empty_kg_trace"},
                )
            )
        if policy_counter:
            summary, count = policy_counter.most_common(1)[0]
            insights.append(
                self._insight(
                    run_id,
                    f"Human review pattern repeats: {summary}.",
                    count,
                    len(source_ids),
                    {"kind": "human_review_pattern"},
                )
            )
        if not insights:
            insights.append(
                self._insight(
                    run_id,
                    "No repeated cross-session pattern was strong enough to update memory.",
                    1,
                    len(source_ids),
                    {"kind": "no_update"},
                )
            )
        return insights

    def _proposal_from_insight(self, run_id: str, insight: DreamInsight) -> DreamProposal:
        target_path = self._scope_path("org", None)
        current = target_path.read_text(encoding="utf-8") if target_path.exists() else "# Organization Memory\n"
        section = self._append_section(
            current,
            "Dream Insight",
            f"{insight.summary}\n\nRecommended operating behavior: search official planning agendas, permit portals, and source-linked evidence before treating approval-risk decisions as complete.",
        )
        draft = self.draft(
            {
                "target_path": self._relative(target_path),
                "proposed_content": section,
                "rationale": f"Dreaming generated this update from insight {insight.id}.",
                "dream_run_id": run_id,
                "insight_id": insight.id,
            }
        )
        return DreamProposal(
            dream_run_id=run_id,
            memory_draft_id=draft.id,
            target_memory_path=draft.target_path,
            change_summary=insight.summary,
            diff=f"+ {insight.summary}",
            confidence=min(0.95, 0.5 + insight.score / 10),
        )

    def _diary(self, run: DreamRun, insights: list[DreamInsight], proposals: list[DreamProposal]) -> DreamDiary:
        lines = [
            f"# Dream Diary {run.id}",
            "",
            f"Reviewed {len(run.source_trace_ids)} decision traces and produced {len(insights)} insights.",
            "",
            "## Insights",
        ]
        lines.extend(f"- {insight.summary} (score {insight.score:.2f})" for insight in insights)
        lines.extend(["", "## Proposed Memory Updates"])
        if proposals:
            lines.extend(f"- {proposal.change_summary} -> {proposal.target_memory_path}" for proposal in proposals)
        else:
            lines.append("- No memory update proposed.")
        return DreamDiary(
            dream_run_id=run.id,
            title=f"Dream Diary {run.id}",
            narrative="\n".join(lines),
            insight_ids=[insight.id for insight in insights],
            proposal_ids=[proposal.id for proposal in proposals],
        )

    def _insight(self, run_id: str, summary: str, frequency: int, diversity: int, metadata: dict[str, Any]) -> DreamInsight:
        recency = 1.0
        score = float(frequency) + min(diversity, 3) + recency + 1.0
        return DreamInsight(
            dream_run_id=run_id,
            summary=summary,
            frequency=float(frequency),
            relevance=1.0,
            query_diversity=float(min(diversity, 3)),
            recency=recency,
            cross_day_repeat=1.0 if diversity > 1 else 0.0,
            concept_richness=1.0,
            score=score,
            metadata=metadata,
        )

    def _load_source_traces(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        if isinstance(payload.get("audit_traces"), list):
            return [item for item in payload["audit_traces"] if isinstance(item, dict)]
        audit_file = payload.get("audit_file")
        if audit_file:
            path = Path(str(audit_file)).expanduser()
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                return [item for item in raw if isinstance(item, dict)]
            if isinstance(raw, dict):
                return [raw]
        return []

    def _scope_path(self, scope: str, subject_id: str | None) -> Path:
        if scope == "project":
            safe_id = self._safe_name(subject_id or "unknown")
            return self._safe_path(f"projects/{safe_id}/MEMORY.md")
        if scope == "agent":
            safe_id = self._safe_name(subject_id or "default")
            return self._safe_path(f"agents/{safe_id}/MEMORY.md")
        if scope == "scratch":
            safe_id = self._safe_name(subject_id or "scratch")
            return self._safe_path(f"scratch/{safe_id}.md")
        return self._safe_path("org/MEMORY.md")

    def _target_path(self, payload: dict[str, Any], scope: str, subject_id: str | None) -> Path:
        if payload.get("target_path"):
            return self._safe_path(str(payload["target_path"]))
        return self._scope_path(scope, subject_id)

    def _safe_path(self, relative: str) -> Path:
        candidate = (self.root / relative).resolve()
        root = self.root.resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError("Memory path boundary violation.")
        return candidate

    def _relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.root.resolve()).as_posix()

    def _is_internal(self, path: Path) -> bool:
        relative = self._relative(path)
        return relative.startswith(".drafts/") or relative.startswith(".dreams/")

    def _scope_from_path(self, path: Path) -> tuple[str, str | None]:
        relative = self._relative(path)
        parts = relative.split("/")
        if parts[0] == "projects" and len(parts) > 1:
            return "project", parts[1]
        if parts[0] == "agents" and len(parts) > 1:
            return "agent", parts[1]
        if parts[0] == "scratch":
            return "scratch", path.stem
        return "org", None

    def _memory_file(self, path: Path, scope: str, subject_id: str | None, content: str) -> MemoryFile:
        return MemoryFile(path=self._relative(path), scope=scope, subject_id=subject_id, hash=self._hash(content))

    def _write_internal_json(self, folder: str, item_id: str, payload: dict[str, Any]) -> None:
        path = self._safe_path(f"{folder}/{self._safe_name(item_id)}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _append_section(self, current: str, title: str, body: str) -> str:
        prefix = current.rstrip() or "# Organization Memory"
        stamp = datetime.now(UTC).date().isoformat()
        return f"{prefix}\n\n## {title} ({stamp})\n{self._normalize_relative_dates(body).strip()}\n"

    def _normalize_relative_dates(self, content: str) -> str:
        today = date.today().isoformat()
        replacements = {
            "yesterday": f"the day before {today}",
            "today": today,
            "tomorrow": f"the day after {today}",
        }
        normalized = content
        for source, target in replacements.items():
            normalized = normalized.replace(source, target).replace(source.title(), target)
        return normalized

    def _hash(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _excerpt(self, content: str, needle: str) -> str:
        if not content:
            return ""
        lower = content.lower()
        index = lower.find(needle) if needle else 0
        if index < 0:
            index = 0
        return content[max(0, index - 120) : index + 240].strip()

    def _safe_name(self, value: str) -> str:
        return "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in value)[:120]

    def _nested(self, payload: dict[str, Any], *keys: str) -> Any:
        current: Any = payload
        for key in keys:
            if not isinstance(current, dict):
                return None
            current = current.get(key)
        return current
