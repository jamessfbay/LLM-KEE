import argparse
import json
from pathlib import Path
import sys
from typing import Any

from llm_kee.config import load_settings
from llm_kee.models import ChangeSet, LearningSignal, UserFeedback, WorkflowDefinition
from llm_kee.services import KEEEngine
from llm_kee.runtime_protocol import OperationReceiptStore, RuntimeEmitter, RuntimeError, load_runtime_command


def read_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_dream_payload(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    raw = read_json(path)
    if isinstance(raw, list):
        return {"audit_traces": raw}
    if isinstance(raw, dict):
        return raw
    return {"input": raw}


def print_json(payload: Any) -> None:
    print(json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False))


def to_jsonable(payload: Any) -> Any:
    if hasattr(payload, "model_dump"):
        return payload.model_dump(mode="json")
    if isinstance(payload, list):
        return [to_jsonable(item) for item in payload]
    if isinstance(payload, dict):
        return {key: to_jsonable(value) for key, value in payload.items()}
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-KEE local utility")
    parser.add_argument("--event-stream", action="store_true", help="Emit RuntimeEvent v1 NDJSON")
    parser.add_argument("--runtime-context", help="Path to a RuntimeCommand v1 JSON file")
    parser.add_argument("--idempotency-key", help="Override the RuntimeCommand idempotency key")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("stats", help="Show local store counts")

    skills_parser = subparsers.add_parser("skills", help="Skill registry commands")
    skills_subparsers = skills_parser.add_subparsers(dest="skills_command", required=True)
    skills_subparsers.add_parser("list", help="List registered skills")

    actions_parser = subparsers.add_parser("actions", help="Action registry commands")
    actions_subparsers = actions_parser.add_subparsers(dest="actions_command", required=True)
    actions_subparsers.add_parser("list", help="List registered actions")

    action_parser = subparsers.add_parser("action", help="Run an AI action")
    action_subparsers = action_parser.add_subparsers(dest="action_command", required=True)
    action_run_parser = action_subparsers.add_parser("run", help="Run an action with JSON input")
    action_run_parser.add_argument("action_type")
    action_run_parser.add_argument("json_file")
    action_list_runs_parser = action_subparsers.add_parser("list-runs", help="List executed action instances")
    action_list_runs_parser.add_argument("--type", dest="action_type", default=None)
    action_artifacts_parser = action_subparsers.add_parser("artifacts", help="List action artifacts")
    action_artifacts_parser.add_argument("--run-id", dest="run_id", default=None)
    action_show_artifact_parser = action_subparsers.add_parser("show-artifact", help="Show one action artifact")
    action_show_artifact_parser.add_argument("artifact_id")

    monitor_parser = subparsers.add_parser("monitor", help="Monitor files and create learning signals")
    monitor_subparsers = monitor_parser.add_subparsers(dest="monitor_command", required=True)
    monitor_scan_parser = monitor_subparsers.add_parser("scan", help="Scan a path and persist monitor signals")
    monitor_scan_parser.add_argument("path")
    monitor_diff_parser = monitor_subparsers.add_parser("diff", help="Preview monitor events without persisting")
    monitor_diff_parser.add_argument("path")
    monitor_signals_parser = monitor_subparsers.add_parser("signals", help="Scan a path and print learning signals")
    monitor_signals_parser.add_argument("path")

    workflow_parser = subparsers.add_parser("workflow", help="Plan or run workflows")
    workflow_subparsers = workflow_parser.add_subparsers(dest="workflow_command", required=True)
    workflow_plan_parser = workflow_subparsers.add_parser("plan", help="Plan a workflow from a task JSON file")
    workflow_plan_parser.add_argument("json_file")
    workflow_run_parser = workflow_subparsers.add_parser("run", help="Run a workflow definition JSON file")
    workflow_run_parser.add_argument("json_file")

    evolution_parser = subparsers.add_parser("evolution", help="Inspect or create evolution records")
    evolution_subparsers = evolution_parser.add_subparsers(dest="evolution_command", required=True)
    evolution_history_parser = evolution_subparsers.add_parser("history", help="Show target evolution history")
    evolution_history_parser.add_argument("target_id")
    evolution_create_parser = evolution_subparsers.add_parser("create", help="Create an evolution event from a change set JSON file")
    evolution_create_parser.add_argument("json_file")

    mape_parser = subparsers.add_parser("mape", help="Run MAPE-K cycles")
    mape_subparsers = mape_parser.add_subparsers(dest="mape_command", required=True)
    mape_run_parser = mape_subparsers.add_parser("run", help="Run a MAPE cycle from a JSON array of learning signals")
    mape_run_parser.add_argument("json_file", nargs="?")
    mape_run_parser.add_argument("--from-path", dest="from_path", default=None)
    mape_show_parser = mape_subparsers.add_parser("show", help="Show a MAPE cycle")
    mape_show_parser.add_argument("cycle_id")

    intents_parser = subparsers.add_parser("intents", help="Intent graph commands")
    intents_subparsers = intents_parser.add_subparsers(dest="intents_command", required=True)
    intents_detect_parser = intents_subparsers.add_parser("detect", help="Detect intent from JSON input")
    intents_detect_parser.add_argument("json_file")
    intents_subparsers.add_parser("list", help="List detected intents and default patterns")

    failures_parser = subparsers.add_parser("failures", help="Failure graph commands")
    failures_subparsers = failures_parser.add_subparsers(dest="failures_command", required=True)
    failures_record_parser = failures_subparsers.add_parser("record", help="Record a failure from JSON input")
    failures_record_parser.add_argument("json_file")
    failures_subparsers.add_parser("list", help="List recorded failures")

    improvements_parser = subparsers.add_parser("improvements", help="Improvement graph commands")
    improvements_subparsers = improvements_parser.add_subparsers(dest="improvements_command", required=True)
    improvements_propose_parser = improvements_subparsers.add_parser("propose", help="Propose an improvement for a failure")
    improvements_propose_parser.add_argument("failure_id")
    improvements_review_parser = improvements_subparsers.add_parser("review", help="Approve or reject an improvement")
    improvements_review_parser.add_argument("proposal_id")
    review_group = improvements_review_parser.add_mutually_exclusive_group(required=True)
    review_group.add_argument("--approve", action="store_true")
    review_group.add_argument("--reject", action="store_true")

    loops_parser = subparsers.add_parser("loops", help="Run specialized improvement loops")
    loops_subparsers = loops_parser.add_subparsers(dest="loop_name", required=True)
    knowledge_parser = loops_subparsers.add_parser("knowledge", help="Knowledge evolution loop")
    knowledge_subparsers = knowledge_parser.add_subparsers(dest="loop_command", required=True)
    knowledge_run_parser = knowledge_subparsers.add_parser("run", help="Run knowledge loop from signals JSON")
    knowledge_run_parser.add_argument("json_file")
    skill_parser = loops_subparsers.add_parser("skill", help="Skill selection loop")
    skill_subparsers = skill_parser.add_subparsers(dest="loop_command", required=True)
    skill_run_parser = skill_subparsers.add_parser("run", help="Run skill loop from task JSON")
    skill_run_parser.add_argument("json_file")
    agent_parser = loops_subparsers.add_parser("agent", help="Agent improvement loop")
    agent_subparsers = agent_parser.add_subparsers(dest="loop_command", required=True)
    agent_run_parser = agent_subparsers.add_parser("run", help="Run agent loop from conversation JSON")
    agent_run_parser.add_argument("json_file")

    memory_parser = subparsers.add_parser("memory", help="Markdown memory commands")
    memory_subparsers = memory_parser.add_subparsers(dest="memory_command", required=True)
    memory_subparsers.add_parser("list", help="List memory files, drafts, and dreams")
    memory_search_parser = memory_subparsers.add_parser("search", help="Search Markdown memory")
    memory_search_parser.add_argument("query")
    memory_read_parser = memory_subparsers.add_parser("read", help="Read one memory file by scope and id")
    memory_read_parser.add_argument("scope")
    memory_read_parser.add_argument("subject_id", nargs="?")
    memory_draft_parser = memory_subparsers.add_parser("draft", help="Create a memory draft from JSON")
    memory_draft_parser.add_argument("json_file")
    memory_apply_parser = memory_subparsers.add_parser("apply", help="Apply a memory draft")
    memory_apply_parser.add_argument("draft_id")
    memory_apply_parser.add_argument("--approve", action="store_true", required=True)
    memory_reject_parser = memory_subparsers.add_parser("reject", help="Reject a memory draft")
    memory_reject_parser.add_argument("draft_id")

    dream_parser = subparsers.add_parser("dream", help="Run out-of-band memory dreaming")
    dream_subparsers = dream_parser.add_subparsers(dest="dream_command", required=True)
    dream_run_parser = dream_subparsers.add_parser("run", help="Run a deterministic dream from audit JSON")
    dream_run_parser.add_argument("json_file")
    dream_scheduler_parser = dream_subparsers.add_parser("scheduler", help="Run or inspect the Dream scheduler")
    dream_scheduler_subparsers = dream_scheduler_parser.add_subparsers(dest="scheduler_command", required=True)
    dream_scheduler_status_parser = dream_scheduler_subparsers.add_parser("status", help="Show Dream scheduler state")
    dream_scheduler_run_parser = dream_scheduler_subparsers.add_parser("run", help="Run one scheduled Dream cycle")
    dream_scheduler_run_parser.add_argument("--once", action="store_true", required=True)
    dream_scheduler_run_parser.add_argument("json_file", nargs="?")
    dream_scheduler_start_parser = dream_scheduler_subparsers.add_parser("start", help="Start the local Dream scheduler loop")
    dream_scheduler_start_parser.add_argument("--interval-minutes", type=int, default=60)
    dream_scheduler_start_parser.add_argument("json_file", nargs="?")
    dream_diary_parser = dream_subparsers.add_parser("diary", help="Show a dream diary")
    dream_diary_parser.add_argument("dream_run_id")
    dream_review_parser = dream_subparsers.add_parser("review", help="Approve or reject a dream proposal")
    dream_review_parser.add_argument("proposal_id")
    dream_review_group = dream_review_parser.add_mutually_exclusive_group(required=True)
    dream_review_group.add_argument("--approve", action="store_true")
    dream_review_group.add_argument("--reject", action="store_true")
    dream_review_parser.add_argument("--notes", default=None)

    feedback_parser = subparsers.add_parser(
        "feedback",
        help="Create feedback, interpret it, and generate an update proposal",
    )
    feedback_parser.add_argument("json_file", help="Path to a UserFeedback JSON file")

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Run all evaluators and the learning gate for a proposal",
    )
    evaluate_parser.add_argument("proposal_id")

    apply_parser = subparsers.add_parser(
        "apply",
        help="Build and apply an approved proposal update plan",
    )
    apply_parser.add_argument("proposal_id")

    args = parser.parse_args()

    engine = KEEEngine(load_settings())
    if args.event_stream:
        _run_streamed(args, engine)
        return
    if args.command == "stats":
        print_json(
            {
                "feedback": len(engine.store.feedback.list()),
                "traces": len(engine.store.traces.list()),
                "proposals": len(engine.store.proposals.list()),
                "evaluations": len(engine.store.evaluations.list()),
                "patterns": len(engine.store.patterns.list()),
                "schema_suggestions": len(engine.store.schema_suggestions.list()),
                "skills": len(engine.store.skills.list()),
                "actions": len(engine.store.action_definitions.list()),
                "workflow_runs": len(engine.store.workflow_runs.list()),
                "evolution_events": len(engine.store.evolution_events.list()),
                "mape_cycles": len(engine.store.mape_cycles.list()),
                "intent_patterns": len(engine.store.intent_patterns.list()),
                "conversation_intents": len(engine.store.conversation_intents.list()),
                "task_intents": len(engine.store.task_intents.list()),
                "failure_records": len(engine.store.failure_records.list()),
                "improvement_proposals": len(engine.store.improvement_proposals.list()),
                "knowledge_evolution_cycles": len(engine.store.knowledge_evolution_cycles.list()),
                "skill_selection_cycles": len(engine.store.skill_selection_cycles.list()),
                "agent_improvement_cycles": len(engine.store.agent_improvement_cycles.list()),
                "memory_files": len(engine.store.memory_files.list()),
                "memory_drafts": len(engine.store.memory_drafts.list()),
                "dream_runs": len(engine.store.dream_runs.list()),
                "dream_proposals": len(engine.store.dream_proposals.list()),
                "dream_diaries": len(engine.store.dream_diaries.list()),
            }
        )
    elif args.command == "skills" and args.skills_command == "list":
        print_json(engine.store.skills.list())
    elif args.command == "actions" and args.actions_command == "list":
        print_json(engine.store.action_definitions.list())
    elif args.command == "action" and args.action_command == "run":
        run = engine.run_action(args.action_type, read_json(args.json_file))
        print_json(run)
    elif args.command == "action" and args.action_command == "list-runs":
        runs = engine.store.action_runs.list()
        if args.action_type:
            runs = [run for run in runs if run.action_type == args.action_type]
        print_json(runs)
    elif args.command == "action" and args.action_command == "artifacts":
        artifacts = engine.store.action_artifacts.list()
        if args.run_id:
            artifacts = [artifact for artifact in artifacts if artifact.action_run_id == args.run_id]
        print_json(artifacts)
    elif args.command == "action" and args.action_command == "show-artifact":
        artifact = engine.store.action_artifacts.get(args.artifact_id)
        if not artifact:
            raise SystemExit(f"Artifact not found: {args.artifact_id}")
        print_json(artifact)
    elif args.command == "workflow" and args.workflow_command == "plan":
        workflow = engine.plan_workflow(read_json(args.json_file))
        print_json(workflow)
    elif args.command == "workflow" and args.workflow_command == "run":
        workflow = WorkflowDefinition.model_validate(read_json(args.json_file))
        run = engine.run_workflow(workflow)
        print_json(run)
    elif args.command == "monitor" and args.monitor_command == "scan":
        print_json(engine.monitor_path(args.path))
    elif args.command == "monitor" and args.monitor_command == "diff":
        print_json(engine.monitor.diff(args.path))
    elif args.command == "monitor" and args.monitor_command == "signals":
        print_json(engine.monitor_path(args.path))
    elif args.command == "evolution" and args.evolution_command == "history":
        print_json(engine.evolution.history(args.target_id))
    elif args.command == "evolution" and args.evolution_command == "create":
        event = engine.create_evolution_event(ChangeSet.model_validate(read_json(args.json_file)))
        print_json(event)
    elif args.command == "mape" and args.mape_command == "run":
        if args.from_path:
            signals = engine.monitor_path(args.from_path)
        elif args.json_file:
            raw = json.loads(Path(args.json_file).read_text(encoding="utf-8"))
            signals = [LearningSignal.model_validate(item) for item in raw]
        else:
            raise SystemExit("mape run requires signals.json or --from-path PATH")
        print_json(engine.run_mape_cycle(signals))
    elif args.command == "mape" and args.mape_command == "show":
        cycle = engine.store.mape_cycles.get(args.cycle_id)
        if not cycle:
            raise SystemExit(f"MAPE cycle not found: {args.cycle_id}")
        print_json(cycle)
    elif args.command == "intents" and args.intents_command == "detect":
        print_json(engine.detect_intent(read_json(args.json_file)))
    elif args.command == "intents" and args.intents_command == "list":
        print_json(
            {
                "patterns": engine.store.intent_patterns.list(),
                "conversation_intents": engine.store.conversation_intents.list(),
                "task_intents": engine.store.task_intents.list(),
            }
        )
    elif args.command == "failures" and args.failures_command == "record":
        print_json(engine.record_failure(read_json(args.json_file)))
    elif args.command == "failures" and args.failures_command == "list":
        print_json(engine.store.failure_records.list())
    elif args.command == "improvements" and args.improvements_command == "propose":
        print_json(engine.propose_improvement(args.failure_id))
    elif args.command == "improvements" and args.improvements_command == "review":
        if args.approve:
            print_json(engine.approve_improvement(args.proposal_id))
        else:
            print_json(engine.reject_improvement(args.proposal_id))
    elif args.command == "loops" and args.loop_name == "knowledge" and args.loop_command == "run":
        raw = read_json(args.json_file)
        signals = [LearningSignal.model_validate(item) for item in raw]
        print_json(engine.run_knowledge_evolution(signals))
    elif args.command == "loops" and args.loop_name == "skill" and args.loop_command == "run":
        print_json(engine.run_skill_selection(read_json(args.json_file)))
    elif args.command == "loops" and args.loop_name == "agent" and args.loop_command == "run":
        print_json(engine.run_agent_improvement(read_json(args.json_file)))
    elif args.command == "memory" and args.memory_command == "list":
        print_json(engine.list_memory())
    elif args.command == "memory" and args.memory_command == "search":
        print_json(engine.search_memory(args.query))
    elif args.command == "memory" and args.memory_command == "read":
        print_json(engine.read_memory(args.scope, args.subject_id))
    elif args.command == "memory" and args.memory_command == "draft":
        print_json(engine.draft_memory(read_json(args.json_file)))
    elif args.command == "memory" and args.memory_command == "apply":
        print_json(engine.apply_memory_draft(args.draft_id))
    elif args.command == "memory" and args.memory_command == "reject":
        print_json(engine.reject_memory_draft(args.draft_id))
    elif args.command == "dream" and args.dream_command == "run":
        print_json(engine.run_dream(read_json(args.json_file)))
    elif args.command == "dream" and args.dream_command == "scheduler" and args.scheduler_command == "status":
        print_json(engine.dream_scheduler_status())
    elif args.command == "dream" and args.dream_command == "scheduler" and args.scheduler_command == "run":
        payload = read_dream_payload(args.json_file)
        print_json(engine.run_dream_scheduler_once(payload))
    elif args.command == "dream" and args.dream_command == "scheduler" and args.scheduler_command == "start":
        payload = read_dream_payload(args.json_file)
        print_json(engine.start_dream_scheduler(args.interval_minutes, payload))
    elif args.command == "dream" and args.dream_command == "diary":
        print_json(engine.dream_diary(args.dream_run_id))
    elif args.command == "dream" and args.dream_command == "review":
        print_json(engine.review_dream_proposal(args.proposal_id, approve=args.approve, notes=args.notes))
    elif args.command == "feedback":
        feedback = UserFeedback.model_validate(read_json(args.json_file))
        saved_feedback, proposal = engine.accept_feedback(feedback)
        print_json({"feedback": saved_feedback, "proposal": proposal})
    elif args.command == "evaluate":
        proposal = engine.store.proposals.get(args.proposal_id)
        if not proposal:
            raise SystemExit(f"Proposal not found: {args.proposal_id}")
        results, aggregate = engine.run_evaluations(proposal)
        decision = engine.store.decisions.list()[-1]
        print_json({"results": results, "aggregate": aggregate, "decision": decision})
    elif args.command == "apply":
        proposal = engine.store.proposals.get(args.proposal_id)
        if not proposal:
            raise SystemExit(f"Proposal not found: {args.proposal_id}")
        result = engine.safe_apply.apply(proposal)
        engine.store.proposals.upsert(proposal)
        print_json(result)


def _run_streamed(args, engine: KEEEngine) -> None:
    command = load_runtime_command(args.runtime_context)
    if command is None:
        raise ValueError("--runtime-context is required with --event-stream")
    if args.idempotency_key:
        command.idempotency_key = args.idempotency_key
    if command.engine != "llm-kee":
        raise ValueError(f"Runtime command engine mismatch: {command.engine}")
    emitter = RuntimeEmitter(command, sys.stdout)
    receipts = OperationReceiptStore(engine.settings.workspace, ".llm_kee")
    receipt = receipts.begin(command)
    if receipt.status in {"succeeded", "insufficient"}:
        emitter.emit("step.completed", receipt.status, references=_runtime_references(receipt.output), payload={"replayed": True})
        return
    emitter.emit("step.started", "running", payload={"command": args.command})
    try:
        output, status = _execute_runtime_command(args, engine, command, emitter)
        receipts.complete(receipt, status, output)
        if output.get("artifact_ids") or output.get("proposal_ids"):
            emitter.emit("artifact.created", status, references=_runtime_references(output))
        emitter.emit("step.completed", status, references=_runtime_references(output), payload=_runtime_counts(output))
    except KeyboardInterrupt:
        receipts.interrupt(receipt, "LLM-KEE operation interrupted")
        emitter.emit("step.failed", "interrupted", error=RuntimeError(code="interrupted", message="LLM-KEE operation interrupted", retryable=True))
    except Exception as exc:
        receipts.complete(receipt, "failed", {"error": str(exc)})
        emitter.emit("step.failed", "failed", error=RuntimeError(code="kee_failed", message=str(exc), retryable=True))
        raise


def _execute_runtime_command(args, engine: KEEEngine, command, emitter: RuntimeEmitter) -> tuple[dict[str, Any], str]:
    if args.command == "mape" and args.mape_command == "run":
        signals = _runtime_signals(args, engine, command)
        for phase in ("monitor", "analyze", "plan", "execute", "knowledge"):
            emitter.emit("step.progress", "running", payload={"phase": phase})
        cycle = engine.run_mape_cycle(signals)
        execution_status = str(cycle.execution.get("status") or "")
        status = "succeeded" if cycle.status == "completed" and execution_status == "completed" else "insufficient"
        return {
            "mape_cycle_id": cycle.id,
            "signal_ids": [signal.id for signal in signals],
            "action_run_ids": cycle.action_run_ids,
            "artifact_ids": cycle.artifact_ids,
            "proposal_ids": cycle.proposal_ids,
            "execution_status": execution_status,
        }, status
    if args.command == "loops" and args.loop_command == "run":
        if args.loop_name == "knowledge":
            signals = _runtime_signals(args, engine, command)
            cycle = engine.run_knowledge_evolution(signals)
            return {"knowledge_evolution_cycle_id": cycle.id, "mape_cycle_id": cycle.mape_cycle_id, "signal_ids": cycle.signal_ids, "proposal_ids": cycle.proposal_ids}, "succeeded" if cycle.status == "completed" and cycle.mape_cycle_id else "insufficient"
        if args.loop_name == "skill":
            cycle = engine.run_skill_selection(read_json(args.json_file))
            return {"skill_cycle_id": cycle.id}, "succeeded"
        cycle = engine.run_agent_improvement(read_json(args.json_file))
        return {"agent_cycle_id": cycle.id, "proposal_ids": getattr(cycle, "proposal_ids", [])}, "succeeded"
    if args.command == "failures" and args.failures_command == "record":
        failure = engine.record_failure(read_json(args.json_file))
        return {"failure_id": failure.id}, "succeeded"
    if args.command == "memory" and args.memory_command == "search":
        result = engine.search_memory(args.query)
        matches = result.get("matches", [])
        return {"memory_paths": [str(item.get("path")) for item in matches if isinstance(item, dict)]}, "succeeded" if matches else "insufficient"
    if args.command == "dream" and args.dream_command == "run":
        result = engine.run_dream(read_dream_payload(args.json_file))
        run = result["run"]
        return {"dream_run_id": run.id, "proposal_ids": run.proposal_ids, "diary_id": run.diary_id or ""}, "succeeded"
    raise ValueError(f"Runtime event streaming is not supported for command: {args.command}")


def _runtime_signals(args, engine: KEEEngine, command) -> list[LearningSignal]:
    if getattr(args, "from_path", None):
        signals = engine.monitor_path(args.from_path)
    else:
        raw = read_json(args.json_file)
        signals = [LearningSignal.model_validate(item) for item in raw]
    for signal in signals:
        signal.payload = {
            **signal.payload,
            "nox_decision_id": command.decision_id,
            "nox_run_id": command.run_id,
            "nox_step_id": command.step_id,
            "input_hash": command.input_hash,
            "claw_reference_ids": command.input.get("claw_reference_ids", []),
            "kg_trace_id": command.input.get("kg_trace_id"),
        }
    return signals


def _runtime_references(output: dict[str, Any]) -> dict[str, str | list[str]]:
    return {key: value for key, value in output.items() if isinstance(value, str) or (isinstance(value, list) and all(isinstance(item, str) for item in value))}


def _runtime_counts(output: dict[str, Any]) -> dict[str, Any]:
    return {f"{key[:-4]}_count": len(value) for key, value in output.items() if key.endswith("_ids") and isinstance(value, list)}
