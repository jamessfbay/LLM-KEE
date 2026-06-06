import argparse
import json
from pathlib import Path
from typing import Any

from llm_kee.config import load_settings
from llm_kee.models import ChangeSet, LearningSignal, UserFeedback, WorkflowDefinition
from llm_kee.services import KEEEngine


def read_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


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
