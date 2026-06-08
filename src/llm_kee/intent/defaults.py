from llm_kee.models import IntentPattern


def default_intent_patterns() -> list[IntentPattern]:
    return [
        IntentPattern(
            id="project_status_tracking",
            intent_type="project_status_tracking",
            description="Track current status, approvals, delays, and project changes.",
            trigger_terms=["status", "approved", "delayed", "planning", "commission", "hearing"],
            success_criteria=["current status is identified", "status evidence is cited"],
            default_skill_ids=["rag_evidence_retrieval", "event_sourcing_timeline"],
        ),
        IntentPattern(
            id="approval_risk_analysis",
            intent_type="approval_risk_analysis",
            description="Assess approval, entitlement, planning, or development risk.",
            trigger_terms=["risk", "approval", "entitlement", "objection", "appeal", "support"],
            success_criteria=["risk drivers are grounded", "uncertainty is surfaced"],
            default_skill_ids=["evidence_evaluation", "event_sourcing_timeline"],
        ),
        IntentPattern(
            id="evidence_lookup",
            intent_type="evidence_lookup",
            description="Find citations, source records, and evidence for a claim.",
            trigger_terms=["evidence", "source", "citation", "quote", "page", "document"],
            success_criteria=["relevant evidence is returned", "source gaps are listed"],
            default_skill_ids=["rag_evidence_retrieval"],
        ),
        IntentPattern(
            id="timeline_reconstruction",
            intent_type="timeline_reconstruction",
            description="Rebuild a chronology from dated facts and evidence.",
            trigger_terms=["timeline", "chronology", "history", "sequence", "when"],
            success_criteria=["events are ordered", "dates are source-linked"],
            default_skill_ids=["event_sourcing_timeline", "evidence_evaluation"],
        ),
        IntentPattern(
            id="permit_history_lookup",
            intent_type="permit_history_lookup",
            description="Find permit, application, inspection, or record history.",
            trigger_terms=["permit", "application", "inspection", "record", "parcel", "apn"],
            success_criteria=["permit sources are checked", "missing permit data is flagged"],
            default_skill_ids=["rag_evidence_retrieval", "evidence_evaluation"],
        ),
    ]
