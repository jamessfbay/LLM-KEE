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
            id="generate_validation_questions",
            action_type="generate_validation_questions",
            name="Generate Validation Questions",
            description="Generate exactly five neutral, evidence-grounded customer research questions.",
            required_skills=["rag_evidence_retrieval", "evidence_evaluation"],
        ),
        ActionDefinition(
            id="generate_outreach_emails",
            action_type="generate_outreach_emails",
            name="Generate Outreach Emails",
            description="Generate one editable research email for each validated question.",
            required_skills=["rag_evidence_retrieval", "evidence_evaluation"],
        ),
        ActionDefinition(
            id="synthesize_opportunity",
            action_type="synthesize_opportunity",
            name="Synthesize Opportunity",
            description="Summarize an opportunity from source-linked evidence and explicit unknowns.",
            required_skills=["rag_evidence_retrieval", "evidence_evaluation"],
        ),
        ActionDefinition(
            id="generate_product_definition",
            action_type="generate_product_definition",
            name="Generate Product Definition",
            description="Generate an evidence-backed minimum product definition.",
            required_skills=["rag_evidence_retrieval", "evidence_evaluation"],
        ),
        ActionDefinition(
            id="explain_opportunity_delta",
            action_type="explain_opportunity_delta",
            name="Explain Opportunity Delta",
            description="Explain how new evidence changes an existing opportunity.",
            required_skills=["rag_evidence_retrieval", "evidence_evaluation"],
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
