from llm_kee.models import SkillDefinition


def default_skills() -> list[SkillDefinition]:
    return [
        SkillDefinition(
            id="rag_evidence_retrieval",
            name="RAG Evidence Retrieval",
            description="Retrieve source-linked evidence and citations for a task.",
            use_cases=["evidence retrieval", "cited answer", "source lookup"],
            inputs=["task", "query", "source_ids"],
            outputs=["evidence_pack", "citation_candidates"],
            dependencies=[],
            failure_modes=["missing sources", "weak citation match"],
            evaluation_metrics=["citation_precision", "evidence_coverage"],
            tags=["rag", "evidence", "retrieval"],
        ),
        SkillDefinition(
            id="event_sourcing_timeline",
            name="Event Sourcing Timeline",
            description="Reconstruct state history and timeline from fragmented events.",
            use_cases=["timeline", "state history", "version reconstruction"],
            inputs=["target_id", "events", "evidence_ids"],
            outputs=["timeline", "missing_events", "state_history"],
            dependencies=["rag_evidence_retrieval"],
            failure_modes=["missing dates", "duplicated events", "ambiguous status language"],
            evaluation_metrics=["timeline_completeness", "date_accuracy"],
            tags=["timeline", "event", "evolution"],
        ),
        SkillDefinition(
            id="evidence_evaluation",
            name="Evidence Evaluation",
            description="Check whether evidence supports claims, outputs, or proposed changes.",
            use_cases=["claim validation", "conflict detection", "missing evidence"],
            inputs=["claim", "evidence_ids", "artifact"],
            outputs=["validation_result", "conflicts", "evidence_gaps"],
            dependencies=["rag_evidence_retrieval"],
            failure_modes=["stale evidence", "citation does not support claim"],
            evaluation_metrics=["support_level", "citation_precision", "conflict_rate"],
            tags=["validation", "evidence", "conflict"],
        ),
        SkillDefinition(
            id="ontology_engineering",
            name="Ontology Engineering",
            description="Define and validate domain-neutral entity, relation, and schema choices.",
            use_cases=["schema design", "entity typing", "relation typing"],
            inputs=["domain", "entities", "relations", "examples"],
            outputs=["ontology_suggestion", "schema_gap_report"],
            dependencies=[],
            failure_modes=["over-specific schema", "ambiguous relation predicates"],
            evaluation_metrics=["schema_fit", "reuse_score"],
            tags=["ontology", "schema", "knowledge_graph"],
        ),
    ]
