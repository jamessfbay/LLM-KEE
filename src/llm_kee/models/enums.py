from enum import StrEnum


class TargetType(StrEnum):
    CLAIM = "claim"
    ENTITY = "entity"
    RELATION = "relation"
    EVIDENCE = "evidence"
    WIKI_PAGE = "wiki_page"
    SCHEMA = "schema"
    PATTERN = "pattern"


class FeedbackType(StrEnum):
    CORRECTION = "correction"
    APPROVAL = "approval"
    REJECTION = "rejection"
    MISSING_EVIDENCE = "missing_evidence"
    DUPLICATE = "duplicate"
    CONFLICT = "conflict"


class FeedbackStatus(StrEnum):
    RECEIVED = "received"
    INTERPRETED = "interpreted"
    PROPOSED = "proposed"
    CLOSED = "closed"


class SignalType(StrEnum):
    FEEDBACK = "feedback"
    REASONING_TRACE = "reasoning_trace"
    NEW_SOURCE = "new_source"
    GRAPH_CONFLICT = "graph_conflict"
    LOW_CONFIDENCE_CLAIM = "low_confidence_claim"
    SCHEMA_GAP = "schema_gap"
    ORPHAN_NODE = "orphan_node"
    DUPLICATE_ENTITY = "duplicate_entity"


class ProposalType(StrEnum):
    CREATE_CLAIM = "create_claim"
    UPDATE_CLAIM = "update_claim"
    RETIRE_CLAIM = "retire_claim"
    UPDATE_RELATION = "update_relation"
    MERGE_ENTITY = "merge_entity"
    ADD_EVIDENCE = "add_evidence"
    UPDATE_WIKI_PAGE = "update_wiki_page"
    SCHEMA_CHANGE = "schema_change"
    PATTERN_PROPOSAL = "pattern_proposal"


class ProposalStatus(StrEnum):
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    NEED_MORE_EVIDENCE = "need_more_evidence"
    CONFLICT_REVIEW = "conflict_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"


class EvaluatorType(StrEnum):
    LLM_JUDGE = "llm_judge"
    RULE_ENGINE = "rule_engine"
    EVIDENCE_CHECKER = "evidence_checker"
    CONFLICT_CHECKER = "conflict_checker"
    DOMAIN_EXPERT = "domain_expert"
    BEHAVIOR_SIGNAL = "behavior_signal"


class EvaluationDecision(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    NEED_MORE_EVIDENCE = "need_more_evidence"
    CONFLICT = "conflict"
    REVIEW = "review"


class LearningDecisionType(StrEnum):
    AUTO_APPLY = "auto_apply"
    PENDING_REVIEW = "pending_review"
    NEED_MORE_EVIDENCE = "need_more_evidence"
    CONFLICT_REVIEW = "conflict_review"
    REJECT = "reject"


class ReviewStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    RETIRED = "retired"
