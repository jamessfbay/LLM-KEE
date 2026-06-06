from llm_kee.models import ActionDefinition


def default_actions() -> list[ActionDefinition]:
    return [
        ActionDefinition(
            id="generate_intelligence_pack",
            action_type="generate_intelligence_pack",
            name="Generate Intelligence Pack",
            description="Generate a source-linked intelligence artifact for any target.",
            required_skills=["rag_evidence_retrieval", "evidence_evaluation", "ontology_engineering"],
        ),
        ActionDefinition(
            id="rebuild_timeline",
            action_type="rebuild_timeline",
            name="Rebuild Timeline",
            description="Reconstruct a target timeline with evidence links and gaps.",
            required_skills=["rag_evidence_retrieval", "event_sourcing_timeline", "evidence_evaluation"],
        ),
        ActionDefinition(
            id="detect_missing_or_conflicting_information",
            action_type="detect_missing_or_conflicting_information",
            name="Detect Missing or Conflicting Information",
            description="Identify evidence gaps, schema gaps, and conflicting claims.",
            required_skills=["rag_evidence_retrieval", "evidence_evaluation", "ontology_engineering"],
        ),
    ]
