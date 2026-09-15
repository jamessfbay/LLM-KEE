from __future__ import annotations

import json
from typing import Any

from llm_kee.experience.hashing import canonical_hash, seal_model
from llm_kee.experience.models import (
    ActorLearningHandoff,
    CausalStatus,
    ExperienceCandidate,
    ExperienceCandidateStatus,
    ExperienceKind,
    TaskVerdict,
)


_RESTRICTED_CHANGE_MARKERS = (
    "change permission",
    "modify permission",
    "bypass safety",
    "disable safety",
    "change policy",
    "modify policy",
    "external action",
    "send externally",
    "execute trade",
    "place order",
)

_ALLOWED_STRUCTURED_FIELDS = {
    "condition",
    "action",
    "expected_outcome",
    "kind",
    "causal_status",
    "mechanism_hypothesis",
    "counterexample_refs",
    "invalidation_conditions",
    "applicability",
    "uncertainty",
}


class ExperienceDistiller:
    """Distill only the terminal actor's signed learning handoff.

    The distiller never consumes a verifier result or outcome independently;
    doing so would let another component impersonate the experience-owning
    actor and lose the original model/prompt/tool binding.
    """

    def distill(self, handoff: ActorLearningHandoff | dict[str, Any]) -> list[ExperienceCandidate]:
        source = (
            handoff
            if isinstance(handoff, ActorLearningHandoff)
            else ActorLearningHandoff.model_validate(handoff)
        )
        candidates: list[ExperienceCandidate] = []
        for index, conclusion in enumerate(source.conclusions):
            claim = self._parse_claim(conclusion, source)
            self._enforce_learning_boundary(claim)
            identity = canonical_hash(
                {
                    "handoff_hash": source.handoff_hash,
                    "conclusion_index": index,
                    "conclusion": conclusion,
                }
            )
            candidates.append(
                seal_model(
                    ExperienceCandidate,
                    "candidate_hash",
                    id=f"experience_{identity[:24]}",
                    handoff_id=source.id,
                    tenant_id=source.tenant_id,
                    domain=source.domain,
                    kind=claim["kind"],
                    causal_status=claim["causal_status"],
                    condition=claim["condition"],
                    action=claim["action"],
                    expected_outcome=claim["expected_outcome"],
                    mechanism_hypothesis=claim.get("mechanism_hypothesis"),
                    evidence_refs=list(source.evidence_refs),
                    counterexample_refs=claim["counterexample_refs"],
                    applicability=claim["applicability"],
                    invalidation_conditions=claim["invalidation_conditions"],
                    uncertainty=claim["uncertainty"],
                    parent_memory_hash=source.parent_memory_hash,
                    status=ExperienceCandidateStatus.PROPOSED,
                )
            )
        return candidates

    def _parse_claim(
        self,
        conclusion: str,
        handoff: ActorLearningHandoff,
    ) -> dict[str, Any]:
        structured: dict[str, Any] | None = None
        try:
            decoded = json.loads(conclusion)
            if isinstance(decoded, dict):
                unexpected = set(decoded) - _ALLOWED_STRUCTURED_FIELDS
                if unexpected:
                    raise ValueError(
                        "structured actor conclusion contains unsupported fields: "
                        + ", ".join(sorted(unexpected))
                    )
                structured = decoded
        except json.JSONDecodeError:
            pass

        if structured is None:
            passed = handoff.verdict == TaskVerdict.PASS
            return {
                "kind": ExperienceKind.PROCEDURE if passed else ExperienceKind.FAILURE_LESSON,
                "causal_status": CausalStatus.OBSERVED_ASSOCIATION,
                "condition": (
                    f"For {handoff.domain} subject {handoff.subject_id}, under the verified "
                    f"evidence set from decision {handoff.decision_id}"
                ),
                "action": (
                    "Reuse the verified internal procedure represented by "
                    + ", ".join(handoff.action_refs)
                    if passed
                    else "Avoid or revise the failed internal procedure represented by "
                    + ", ".join(handoff.action_refs)
                ),
                "expected_outcome": conclusion,
                "counterexample_refs": [],
                "applicability": {
                    "domain": handoff.domain,
                    "subject_id": handoff.subject_id,
                    "source_decision_id": handoff.decision_id,
                },
                "invalidation_conditions": list(handoff.uncertainties),
                "uncertainty": 0.25 if passed and not handoff.uncertainties else 0.6,
            }

        condition = str(structured.get("condition") or "").strip()
        action = str(structured.get("action") or "").strip()
        outcome = str(structured.get("expected_outcome") or "").strip()
        if not condition or not action or not outcome:
            raise ValueError("structured actor conclusion requires condition, action and expected_outcome")
        kind = ExperienceKind(str(structured.get("kind") or ExperienceKind.PROCEDURE))
        causal_status = CausalStatus(
            str(structured.get("causal_status") or CausalStatus.OBSERVED_ASSOCIATION)
        )
        mechanism = structured.get("mechanism_hypothesis")
        if causal_status != CausalStatus.OBSERVED_ASSOCIATION and not str(mechanism or "").strip():
            raise ValueError("non-associational actor conclusion requires a mechanism hypothesis")
        applicability = structured.get("applicability")
        if not isinstance(applicability, dict):
            applicability = {"domain": handoff.domain, "subject_id": handoff.subject_id}
        if applicability.get("domain", handoff.domain) != handoff.domain:
            raise ValueError("actor conclusion cannot expand to another domain")
        return {
            "kind": kind,
            "causal_status": causal_status,
            "condition": condition,
            "action": action,
            "expected_outcome": outcome,
            "mechanism_hypothesis": str(mechanism).strip() if mechanism else None,
            "counterexample_refs": self._strings(structured.get("counterexample_refs", [])),
            "applicability": applicability,
            "invalidation_conditions": self._strings(
                structured.get("invalidation_conditions", handoff.uncertainties)
            ),
            "uncertainty": float(structured.get("uncertainty", 0.5)),
        }

    @staticmethod
    def _strings(value: Any) -> list[str]:
        if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
            raise ValueError("experience list fields must contain non-empty strings")
        return list(dict.fromkeys(value))

    @staticmethod
    def _enforce_learning_boundary(claim: dict[str, Any]) -> None:
        text = " ".join(
            str(claim.get(field, ""))
            for field in ("condition", "action", "expected_outcome", "mechanism_hypothesis")
        ).lower()
        if any(marker in text for marker in _RESTRICTED_CHANGE_MARKERS):
            raise ValueError("experience cannot modify policy, permission, safety or external action scope")
