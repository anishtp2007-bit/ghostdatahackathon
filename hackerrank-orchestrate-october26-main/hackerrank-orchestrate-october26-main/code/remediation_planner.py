"""Build evidence-backed findings, future fixes, and verification steps.

The planner does not write output.csv. It consumes the dataset, identity
resolver, evidence scan, and policy engine, then returns one auditable plan per
request. Unknown intent or evidence is carried as an unresolved result instead
of being converted into a made-up finding or a false COMPLETE.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from typing import Any, Iterable, Mapping

from data_loader import LoadedDataset, parse_iso_timestamp
from evidence_scanner import EvidenceRecord, ScanResult
from identity_resolver import IdentityResolver
from policy_engine import DecisionStatus, PolicyDecision, PolicyEngine


OUTPUT_STATUSES = {"COMPLETE", "INCOMPLETE", "NOT_APPLICABLE"}
STEP_FIELDS = {
    "step_no", "phase", "system_id", "resource_id", "operation", "state", "evidence_id",
}
PHASE_ORDER = {"FINDING": 0, "REMEDIATION": 1, "VERIFICATION": 2}
OPERATION_ORDER = {
    "REPORT_REMAINING_RECORD": 0,
    "REPORT_EXPOSED_PII": 1,
    "REPORT_ACTIVE_DATA_FLOW": 2,
    "REPORT_RESTORE_RISK": 3,
    "DISABLE_DATA_FLOW": 0,
    "BLOCK_RESTORE": 1,
    "DELETE_RECORD": 2,
    "REDACT_RECORD": 3,
    "ANONYMIZE_RECORD": 4,
    "VERIFY_RECORD_ABSENT": 0,
    "VERIFY_RECORD_PRESENT": 1,
    "VERIFY_RECORD_REDACTED": 2,
    "VERIFY_RECORD_ANONYMIZED": 3,
    "VERIFY_DATA_FLOW_DISABLED": 4,
    "VERIFY_RESTORE_BLOCKED": 5,
}
_FINDING_OPERATION = {
    "DELETE": "REPORT_REMAINING_RECORD",
    "REDACT": "REPORT_EXPOSED_PII",
    "ANONYMIZE": "REPORT_EXPOSED_PII",
}
_REMEDIATION_OPERATION = {
    "DELETE": ("DELETE_RECORD", "ABSENT"),
    "REDACT": ("REDACT_RECORD", "REDACTED"),
    "ANONYMIZE": ("ANONYMIZE_RECORD", "ANONYMIZED"),
}
ALLOWED_OPERATION_STATES = {
    ("FINDING", "REPORT_REMAINING_RECORD", "PRESENT"),
    ("FINDING", "REPORT_EXPOSED_PII", "PRESENT"),
    ("FINDING", "REPORT_ACTIVE_DATA_FLOW", "ACTIVE"),
    ("FINDING", "REPORT_RESTORE_RISK", "ACTIVE"),
    ("REMEDIATION", "DISABLE_DATA_FLOW", "DISABLED"),
    ("REMEDIATION", "BLOCK_RESTORE", "BLOCKED"),
    ("REMEDIATION", "DELETE_RECORD", "ABSENT"),
    ("REMEDIATION", "REDACT_RECORD", "REDACTED"),
    ("REMEDIATION", "ANONYMIZE_RECORD", "ANONYMIZED"),
    ("VERIFICATION", "VERIFY_RECORD_ABSENT", "ABSENT"),
    ("VERIFICATION", "VERIFY_RECORD_PRESENT", "PRESENT"),
    ("VERIFICATION", "VERIFY_RECORD_REDACTED", "REDACTED"),
    ("VERIFICATION", "VERIFY_RECORD_ANONYMIZED", "ANONYMIZED"),
    ("VERIFICATION", "VERIFY_DATA_FLOW_DISABLED", "DISABLED"),
    ("VERIFICATION", "VERIFY_RESTORE_BLOCKED", "BLOCKED"),
}
_VERIFICATION_OPERATION = {
    "DELETE": ("VERIFY_RECORD_ABSENT", "ABSENT"),
    "REDACT": ("VERIFY_RECORD_REDACTED", "REDACTED"),
    "ANONYMIZE": ("VERIFY_RECORD_ANONYMIZED", "ANONYMIZED"),
    "RETAIN": ("VERIFY_RECORD_PRESENT", "PRESENT"),
    "DISABLE_DATA_FLOW": ("VERIFY_DATA_FLOW_DISABLED", "DISABLED"),
    "BLOCK_RESTORE": ("VERIFY_RESTORE_BLOCKED", "BLOCKED"),
}
_RISK_OPERATION = {
    "DATA_FLOW": ("REPORT_ACTIVE_DATA_FLOW", "DISABLE_DATA_FLOW", "DISABLED"),
    "RESTORE_PATH": ("REPORT_RESTORE_RISK", "BLOCK_RESTORE", "BLOCKED"),
}
_DIRECT_FIELD_NAMES = {
    "email", "email_address", "customer_email", "requester_email", "owner_email",
    "phone", "phone_number", "customer_phone", "requester_phone", "owner_phone",
    "name", "full_name", "display_name", "requester_name", "customer_name",
    "email_hash", "stable_email_hash",
}
_SEARCHABLE_TEXT_FIELDS = {
    "subject_selector", "selector", "query", "lookup_value", "message_text",
    "message_excerpt", "message", "document", "document_text", "profile", "details",
    "observation_detail",
}
_SYSTEM_CONTEXT_MAP = (
    ("customer_profiles", "account_service"),
    ("customer_accounts", "account_service"),
    ("customer_search", "search_service"),
    ("account_service", "account_service"),
    ("analytics_platform", "analytics_platform"),
    ("backup_service", "backup_service"),
    ("export_service", "export_service"),
    ("integration_service", "integration_service"),
    ("search_service", "search_service"),
    ("security_service", "security_service"),
    ("support_platform", "support_platform"),
)


class RequestIntent(StrEnum):
    """Whether the request establishes that deletion was authorized."""

    DELETION = "DELETION"
    NON_DELETION = "NON_DELETION"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True, slots=True)
class PlanStep:
    """One allowed output action with exactly the challenge's fields."""

    step_no: int
    phase: str
    system_id: str
    resource_id: str
    operation: str
    state: str
    evidence_id: str

    def as_dict(self) -> dict[str, int | str]:
        """Serialize to exactly the required output object fields."""
        return {
            "step_no": self.step_no,
            "phase": self.phase,
            "system_id": self.system_id,
            "resource_id": self.resource_id,
            "operation": self.operation,
            "state": self.state,
            "evidence_id": self.evidence_id,
        }


@dataclass(frozen=True, slots=True)
class CasePlan:
    """Plan result. ``deletion_status=None`` means evidence/intent is unresolved."""

    case_id: str
    deletion_status: str | None
    steps: tuple[PlanStep, ...]
    user_ids: tuple[str, ...] = ()
    unresolved_reasons: tuple[str, ...] = ()

    @property
    def finalizable(self) -> bool:
        """Whether the plan can safely be serialized as the final challenge row."""
        return self.deletion_status in OUTPUT_STATUSES and not self.unresolved_reasons

    def output_steps(self) -> list[dict[str, int | str]]:
        return [step.as_dict() for step in self.steps]

    def as_output_row(self) -> dict[str, str]:
        """Return a contract-shaped row, refusing to serialize unresolved cases."""
        if not self.finalizable:
            raise ValueError(f"Case {self.case_id} is unresolved and cannot be serialized as a final answer")
        return {
            "case_id": self.case_id,
            "deletion_status": self.deletion_status,
            "steps_json": "[]" if self.deletion_status == "NOT_APPLICABLE" else self._steps_json(),
        }

    def _steps_json(self) -> str:
        import json

        return json.dumps(self.output_steps(), ensure_ascii=False, separators=(",", ":"))


class RequestIntentClassifier:
    """Conservative multilingual lexical check for deletion authorization.

    Explicit deletion phrases win over access/correction/marketing language.
    A request with no clear deletion or non-deletion phrase stays unresolved.
    """

    _DELETION_PATTERNS = tuple(re.compile(pattern) for pattern in (
        r"\b(delete|deletion|erase|erasure|purge|wipe|destroy)\b",
        r"\b(remove|removal)\b.{0,50}\b(account|profile|personal data|personal information|my data|my details|my records)\b",
        r"\b(account|profile|personal data|personal information|my data|my records)\b.{0,50}\b(remove|removal|erase|delete|purge|wipe|clear)\b",
        r"\b(forget me|right to be forgotten|close (?:my|this|the) account|close my account permanently|permanent account closure)\b",
        r"\b(suppress|suppression)\b.{0,40}\b(account|personal data|personal information|my data)\b",
        r"\b(borr(ar|ado|ada)|elimin(ar|aci[oó]n)|suprim(ir|ir mi)|apagar|exclu(ir|s[aã]o))\b",
        r"\b(supprim(er|e|ez|ession)|suppression|effac(er|ement)|oubli(?:er|ez)|cl[oô]turer)\b",
        r"\b(l[oö]sch(?:en|ung)|entfern(?:en|ung)|tilg(?:en|ung)|konto schlie(?:ss|ß)en)\b",
        r"\b(cancell(?:are|azione)|elimin(?:are|azione)|rimuov(?:ere|i)|chiudere (?:il )?account)\b",
        r"\b(verwijder(?:en)?|wissen|account sluiten)\b",
        r"\b(hapus|menghapus|menghilangkan|x[oó]a|xo[aá]|xoá)\b",
        r"\b(delete|remove|erase) my account\b",
        r"(削除|消去|抹消|退会|アカウント.{0,8}(?:閉鎖|削除))",
        r"(删除|刪除|注销|註銷|消除).{0,8}(账号|帳號|账户|帳戶|个人资料|個人資料|数据|資料)",
        r"(계정|개인정보|데이터).{0,8}(삭제|제거)",
    ))
    _NON_DELETION_PATTERNS = tuple(re.compile(pattern) for pattern in (
        r"\b(access|access to|copy of|download|export|portability|portable copy)\b.{0,45}\b(my |personal )?(data|information|records|account)\b",
        r"\b(correct|correction|rectify|rectification|update|change)\b.{0,45}\b(my |personal )?(data|information|details|address|phone|email)\b",
        r"\b(unsubscribe|opt[ -]?out|stop (?:sending|using) (?:me )?(?:marketing|promotional)|marketing opt[ -]?out)\b",
        r"\b(acceso|acceder|copia de|exportar|rectificar|corregir|actualizar|baja de publicidad)\b",
        r"\b(acc[eè]s|copie de|exporter|rectifier|corriger|mettre [aà] jour|d[eé]sabonner)\b",
        r"\b(auskunft|zugriff|kopie|export|berichtigen|korrigieren|aktualisieren|abmelden)\b",
        r"\b(acesso|c[oó]pia|exportar|corrigir|atualizar|cancelar marketing)\b",
        r"\b(accesso|copia|esportare|correggere|aggiornare|disiscrivere)\b",
        r"(訂閱|退訂|访问|訪問|复制|複製|更正|修正|更新).{0,8}(数据|資料|邮件|郵件|营销|行銷)",
        r"(정보.{0,4}(열람|정정|이동)|수정|정정|마케팅.{0,4}(수신거부|거부))",
    ))
    _NEGATED_DELETION_PATTERNS = tuple(re.compile(pattern) for pattern in (
        r"\b(do not|don't|never|not)\b.{0,25}\b(delete|deletion|erase|erasure|remove|purge)\b",
        r"\b(not requesting|not a request for|no request to)\b.{0,30}\b(deletion|delete|erasure|erase)\b",
        r"\bnot\s+(?:requesting\s+)?(?:a\s+)?(?:data\s+)?deletion\b",
        r"\b(no quiero|no solicito|sin)\b.{0,25}\b(eliminar|borrar|suprimir)\b",
        r"\b(ne pas|ne demande pas)\b.{0,20}\b(supprimer|effacer)\b",
        r"(削除しない|消去しない|删除不要|不要删除)",
    ))

    @staticmethod
    def _normalize(text: str) -> str:
        folded = unicodedata.normalize("NFKD", text.casefold())
        return "".join(char for char in folded if not unicodedata.combining(char))

    def classify(self, request_text: str) -> RequestIntent:
        normalized = self._normalize(request_text)
        has_deletion = any(pattern.search(normalized) for pattern in self._DELETION_PATTERNS)
        has_non_deletion = any(pattern.search(normalized) for pattern in self._NON_DELETION_PATTERNS)
        if any(pattern.search(normalized) for pattern in self._NEGATED_DELETION_PATTERNS):
            return RequestIntent.NON_DELETION
        if has_deletion:
            return RequestIntent.DELETION
        if has_non_deletion:
            return RequestIntent.NON_DELETION
        return RequestIntent.UNRESOLVED


class RemediationPlanner:
    """Combine identities, evidence, activity, and policy into ordered plans."""

    def __init__(
        self,
        dataset: LoadedDataset,
        *,
        identity_resolver: IdentityResolver | None = None,
        policy_engine: PolicyEngine | None = None,
        scan_result: ScanResult | None = None,
        intent_classifier: RequestIntentClassifier | None = None,
    ) -> None:
        self.dataset = dataset
        self.identity_resolver = identity_resolver or IdentityResolver(
            dataset.user_identities, dataset.user_audit_log
        )
        self.policy_engine = policy_engine or PolicyEngine.from_dataset(
            dataset.deletion_rules, self.identity_resolver
        )
        self.scan_result = scan_result
        self.intent_classifier = intent_classifier or RequestIntentClassifier()
        self._case_components: dict[str, tuple[IdentityResolver, PolicyEngine]] = {}
        self._valid_evidence_ids = {
            row["file_id"] for row in dataset.evidence_files
        } | {row["activity_id"] for row in dataset.system_activity} | {
            row["audit_id"] for row in dataset.user_audit_log
        }
        if self.scan_result is None:
            self._scan_evidence()

    @classmethod
    def from_dataset(cls, dataset: LoadedDataset) -> RemediationPlanner:
        """Build the resolver, policy engine, and one shared evidence scan."""
        return cls(dataset)

    def _scan_evidence(self) -> None:
        from evidence_scanner import EvidenceScanner, load_manifest

        scanner = EvidenceScanner(
            self.dataset.dataset_dir,
            load_manifest(self.dataset.dataset_dir),
            self.identity_resolver,
            self.dataset.system_activity,
        )
        self.scan_result = scanner.scan_all()

    def plan_case(self, request: Mapping[str, str]) -> CasePlan:
        """Plan one request without writing to disk or manufacturing evidence."""
        case_id = request.get("case_id", "")
        if not case_id:
            raise ValueError("A deletion request must contain case_id")
        intent = self.intent_classifier.classify(request.get("request_text", ""))
        if intent == RequestIntent.NON_DELETION:
            return CasePlan(case_id, "NOT_APPLICABLE", (), unresolved_reasons=())
        if intent == RequestIntent.UNRESOLVED:
            return CasePlan(
                case_id, None, (), unresolved_reasons=(
                    "Request text does not clearly authorize deletion or establish a different request type.",
                ),
            )

        audit_at = parse_iso_timestamp(request["audit_at"], location=f"{case_id}.audit_at")
        case_resolver, case_policy_engine = self._components_for_cutoff(audit_at)
        email = request.get("email", "")
        user_ids = tuple(sorted(case_resolver.resolve_email(email)))
        if not user_ids:
            return CasePlan(
                case_id, None, (), unresolved_reasons=(
                    "The request email did not resolve to a current or historical user_id.",
                ),
            )
        if len(user_ids) > 1:
            return CasePlan(
                case_id, None, (), user_ids=user_ids, unresolved_reasons=(
                    "The request email maps to multiple user IDs; identity is ambiguous.",
                ),
            )
        user_id = user_ids[0]

        assert self.scan_result is not None
        # Claims may connect a person to a resource, but are never used as
        # resource-state proof. A matching snapshot still has to support the
        # resulting finding or verification.
        linked_resource_keys = {
            (system_id, record.resource_id)
            for record in self.scan_result.records
            if record.evidence_kind == "claim"
            and record.observed_at is not None
            and record.observed_at <= audit_at
            and self._claim_resource_is_grounded(record)
            and self._record_matches_with_resolver(record, user_id, case_resolver)
            for system_id in self._claim_systems(record)
        }
        records = [
            self._with_resolved_identifiers(
                record, user_id, case_resolver,
                linked_by_claim=(self._canonical_system(record), record.resource_id) in linked_resource_keys,
            )
            for record in self.scan_result.records
            if record.evidence_kind == "observation"
            and record.observed_at is not None
            and record.observed_at <= audit_at
            and (
                self._record_matches_with_resolver(record, user_id, case_resolver)
                or (self._canonical_system(record), record.resource_id) in linked_resource_keys
            )
        ]
        direct_records = [record for record in records if self._has_grounded_resource_id(record)]
        if not direct_records:
            if self._has_request_linked_activity(user_id, audit_at, case_resolver):
                unresolved_reason = (
                    "Activity evidence links the request to a customer resource, but there is no direct snapshot/readback "
                    "for that resource by audit_at; claims cannot establish its actual state."
                )
            else:
                unresolved_reason = "No identity-matched direct observations with grounded resource IDs were available by audit_at."
            return CasePlan(
                case_id, None, (), user_ids=user_ids, unresolved_reasons=(unresolved_reason,),
            )

        findings: list[PlanStep] = []
        remediations: list[PlanStep] = []
        verifications: list[PlanStep] = []
        unresolved: list[str] = []
        all_compliant = True
        applicable_count = 0

        groups: dict[tuple[str, str, str, str], list[EvidenceRecord]] = defaultdict(list)
        for record in direct_records:
            category = case_policy_engine.classify_record(record)
            if category == "UNKNOWN":
                unresolved.append(
                    f"Evidence {record.evidence_id} for resource {record.resource_id} has no supported policy category."
                )
                all_compliant = False
                continue
            key = (self._canonical_system(record), record.resource_id, record.topology, category)
            groups[key].append(record)

        for (system_id, resource_id, _topology, category), group in sorted(groups.items()):
            decisions = self._resource_decisions(group, user_id, audit_at, case_policy_engine)
            if not decisions:
                unresolved.append(f"No usable direct state for {system_id}/{resource_id} at audit_at.")
                all_compliant = False
                continue
            # Recreation controls are an independent gate. Check them before
            # handling a conditional data-policy result such as RETAIN/UNKNOWN.
            latest = self._latest_record(group)
            risk_decisions = case_policy_engine.evaluate_recreation_risks(latest, as_of=audit_at)
            for risk in risk_decisions:
                if risk.status == DecisionStatus.NONCOMPLIANT:
                    all_compliant = False
                    finding, remediation, verification = self._steps_for_risk(
                        risk, system_id, resource_id
                    )
                    if finding is None or remediation is None or verification is None:
                        unresolved.append(f"{system_id}/{resource_id}: recreation risk could not be represented safely.")
                    else:
                        findings.append(finding)
                        remediations.append(remediation)
                        verifications.append(verification)
                elif risk.status == DecisionStatus.UNKNOWN:
                    unresolved.append(f"{system_id}/{resource_id}: {risk.reason}")
                    all_compliant = False
                elif risk.status == DecisionStatus.COMPLIANT:
                    verification = self._verification_for_risk(risk, system_id, resource_id)
                    if verification is not None:
                        verifications.append(verification)

            for decision in decisions:
                if decision.status == DecisionStatus.NOT_APPLICABLE:
                    continue
                applicable_count += 1
                if decision.status in {DecisionStatus.UNKNOWN, DecisionStatus.NEEDS_CONTEXT, DecisionStatus.OUT_OF_SCOPE}:
                    unresolved.append(f"{system_id}/{resource_id}: {decision.reason}")
                    all_compliant = False
                    continue
                if decision.status == DecisionStatus.NONCOMPLIANT:
                    all_compliant = False
                    if decision.required_action in {"DISABLE_DATA_FLOW", "BLOCK_RESTORE"}:
                        finding, remediation, verification = self._steps_for_risk(
                            decision, system_id, resource_id
                        )
                    else:
                        source_record = next(
                            record for record in group if record.evidence_id == decision.evidence_id
                        )
                        finding, remediation, verification = self._steps_for_violation(
                            decision, source_record, system_id, resource_id
                        )
                    if finding is None or remediation is None or verification is None:
                        unresolved.append(
                            f"{system_id}/{resource_id}: the policy violation has no permitted, evidence-backed planning operation."
                        )
                        continue
                    findings.append(finding)
                    remediations.append(remediation)
                    verifications.append(verification)
                elif decision.status == DecisionStatus.COMPLIANT:
                    verification = self._verification_for_compliant(decision, system_id, resource_id)
                    if verification is None:
                        unresolved.append(f"{system_id}/{resource_id}: no allowed verification operation matches the policy.")
                        all_compliant = False
                    else:
                        verifications.append(verification)

        if self.scan_result.issues:
            partial_ids = sorted({issue.evidence_id for issue in self.scan_result.issues if issue.evidence_id})
            unresolved.append(
                "Some participant evidence could not be fully parsed, so the case cannot be certified complete: "
                + ", ".join(partial_ids)
            )

        if applicable_count == 0 and not unresolved:
            unresolved.append("No applicable policy-backed record state was available to verify this deletion.")
        if not all_compliant and (not findings or not remediations) and not unresolved:
            unresolved.append("The deletion is known to be incomplete, but evidence does not support a permitted remediation finding.")

        if findings and remediations:
            # One proven violation is enough to establish INCOMPLETE, even if
            # unresolved evidence prevents a complete remediation/output row.
            status: str | None = "INCOMPLETE"
        elif unresolved:
            status = None
        elif all_compliant and applicable_count > 0:
            status = "COMPLETE"
        else:
            status = None

        if status == "COMPLETE":
            selected_steps = self._sort_steps(verifications)
        elif status == "INCOMPLETE":
            # All verification in an incomplete plan is future work, including
            # checks of already-compliant required-retention records.
            future_verifications = [replace(step, evidence_id="") for step in verifications]
            selected_steps = self._sort_steps(findings + remediations + future_verifications)
        else:
            # Keep evidence-backed partial plan visible for debugging, but this
            # unresolved result is deliberately not serializable as final output.
            selected_steps = self._sort_steps(findings + remediations + verifications)
        numbered = tuple(replace(step, step_no=index) for index, step in enumerate(selected_steps, start=1))
        self._validate_steps(numbered, status, case_id)
        return CasePlan(
            case_id=case_id,
            deletion_status=status,
            steps=numbered,
            user_ids=user_ids,
            unresolved_reasons=tuple(sorted(set(unresolved))),
        )

    def plan_cases(self, requests: Iterable[Mapping[str, str]]) -> list[CasePlan]:
        """Plan an iterable deterministically; does not write output.csv."""
        return [self.plan_case(request) for request in sorted(requests, key=lambda row: row["case_id"])]

    def _components_for_cutoff(self, audit_at: datetime) -> tuple[IdentityResolver, PolicyEngine]:
        key = audit_at.isoformat()
        if key not in self._case_components:
            resolver = IdentityResolver(
                self.dataset.user_identities,
                self.dataset.user_audit_log,
                as_of=audit_at,
            )
            policy = PolicyEngine.from_dataset(self.dataset.deletion_rules, resolver)
            self._case_components[key] = (resolver, policy)
        return self._case_components[key]

    def _record_matches_with_resolver(
        self,
        record: EvidenceRecord,
        user_id: str,
        resolver: IdentityResolver,
    ) -> bool:
        for path, value in self._scalar_items(record.fields):
            if not isinstance(value, str) or not value.strip():
                continue
            leaf = self._leaf(path)
            if leaf in _DIRECT_FIELD_NAMES:
                canonical = self._identity_field(leaf)
                if canonical and user_id in resolver.resolve(value, canonical):
                    return True
            elif leaf in _SEARCHABLE_TEXT_FIELDS:
                if user_id in resolver.match_text(value):
                    return True
            elif leaf in {"user_id", "subject_reference", "customer_reference", "stable_user_id"}:
                if user_id in resolver.resolve(value, "user_id"):
                    return True
        return False

    def _has_request_linked_activity(
        self,
        user_id: str,
        audit_at: datetime,
        resolver: IdentityResolver,
    ) -> bool:
        assert self.scan_result is not None
        return any(
            record.evidence_kind == "claim"
            and record.observed_at is not None
            and record.observed_at <= audit_at
            and self._record_matches_with_resolver(record, user_id, resolver)
            for record in self.scan_result.records
        )

    @staticmethod
    def _claim_resource_is_grounded(record: EvidenceRecord) -> bool:
        """Accept only resource IDs present in an activity field or parsed log line."""
        for path, value in RemediationPlanner._scalar_items(record.fields):
            if RemediationPlanner._leaf(path) == "resource_id" and str(value).strip() == record.resource_id:
                return True
        return (
            record.line_number is not None
            and bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9-]*_\d+", record.resource_id))
        )

    def _with_resolved_identifiers(
        self,
        record: EvidenceRecord,
        user_id: str,
        resolver: IdentityResolver,
        *,
        linked_by_claim: bool = False,
    ) -> EvidenceRecord:
        # Recompute against the case's audit_at-limited resolver; don't inherit
        # links discovered only through identity changes recorded later.
        matches = {user_id} if linked_by_claim or self._record_matches_with_resolver(record, user_id, resolver) else set()
        return replace(record, matched_user_ids=matches)

    @staticmethod
    def _identity_field(leaf: str) -> str | None:
        if "email" in leaf:
            return "email_hash" if "hash" in leaf else "email"
        if "phone" in leaf:
            return "phone"
        if "name" in leaf:
            return "name"
        return None

    @staticmethod
    def _has_grounded_resource_id(record: EvidenceRecord) -> bool:
        grounded_fields = {
            "resource_id", "record_id", "ticket_id", "request_id", "archive_key", "object_key",
        }
        return any(
            RemediationPlanner._leaf(path) in grounded_fields and bool(str(value).strip())
            for path, value in RemediationPlanner._scalar_items(record.fields)
        )

    def _resource_decisions(
        self,
        group: list[EvidenceRecord],
        user_id: str,
        audit_at: datetime,
        policy_engine: PolicyEngine,
    ) -> list[PolicyDecision]:
        latest_time = max(record.observed_at for record in group if record.observed_at is not None)
        latest = [record for record in group if record.observed_at == latest_time]
        if len({record.observed_state for record in latest}) > 1:
            # Let PolicyEngine construct a conflict result from only the tied snapshots.
            decision = policy_engine.evaluate_resource(
                latest,
                user_id=user_id,
                system_id=latest[0].system_id,
                resource_id=latest[0].resource_id,
                as_of=audit_at,
            )
            return [decision]
        return [
            policy_engine.evaluate_record(record, user_id=user_id, as_of=audit_at)
            for record in sorted(latest, key=lambda item: item.evidence_id)
        ]

    @staticmethod
    def _latest_record(group: list[EvidenceRecord]) -> EvidenceRecord:
        newest_time = max(record.observed_at for record in group if record.observed_at is not None)
        latest = [record for record in group if record.observed_at == newest_time]
        return latest[-1]

    def _steps_for_violation(
        self,
        decision: PolicyDecision,
        record: EvidenceRecord,
        system_id: str,
        resource_id: str,
    ) -> tuple[PlanStep | None, PlanStep | None, PlanStep | None]:
        action = decision.required_action
        if action in _REMEDIATION_OPERATION:
            finding_operation = _FINDING_OPERATION[action]
            remediation_operation, remediation_state = _REMEDIATION_OPERATION[action]
            verification_operation, verification_state = _VERIFICATION_OPERATION[action]
            if action == "REDACT" and not decision.exposed_identifiers:
                # No grounded prohibited identifier means there is no evidence for
                # REPORT_EXPOSED_PII; do not make up a finding.
                return None, None, None
            finding = self._step(
                "FINDING", system_id, resource_id, finding_operation,
                "PRESENT", decision.evidence_id,
            )
            remediation = self._step(
                "REMEDIATION", system_id, resource_id, remediation_operation,
                remediation_state, "",
            )
            verification = self._step(
                "VERIFICATION", system_id, resource_id, verification_operation,
                verification_state, "",
            )
            return finding, remediation, verification
        return None, None, None

    def _steps_for_risk(
        self,
        decision: PolicyDecision,
        system_id: str,
        resource_id: str,
    ) -> tuple[PlanStep | None, PlanStep | None, PlanStep | None]:
        risk_category = decision.data_category
        operations = _RISK_OPERATION.get(risk_category)
        if operations is None:
            return None, None, None
        finding_operation, remediation_operation, verification_state = operations
        remediation_state = "DISABLED" if remediation_operation == "DISABLE_DATA_FLOW" else "BLOCKED"
        verification_operation, verification_state = _VERIFICATION_OPERATION[remediation_operation]
        return (
            self._step("FINDING", system_id, resource_id, finding_operation, "ACTIVE", decision.evidence_id),
            self._step("REMEDIATION", system_id, resource_id, remediation_operation, remediation_state, ""),
            self._step("VERIFICATION", system_id, resource_id, verification_operation, verification_state, ""),
        )

    def _verification_for_compliant(
        self,
        decision: PolicyDecision,
        system_id: str,
        resource_id: str,
    ) -> PlanStep | None:
        action = decision.required_action
        if action not in _VERIFICATION_OPERATION:
            return None
        operation, state = _VERIFICATION_OPERATION[action]
        return self._step("VERIFICATION", system_id, resource_id, operation, state, decision.evidence_id)

    def _verification_for_risk(
        self,
        decision: PolicyDecision,
        system_id: str,
        resource_id: str,
    ) -> PlanStep | None:
        remediation = decision.required_action
        if remediation not in {"DISABLE_DATA_FLOW", "BLOCK_RESTORE"}:
            return None
        operation, state = _VERIFICATION_OPERATION[remediation]
        return self._step("VERIFICATION", system_id, resource_id, operation, state, decision.evidence_id)

    @staticmethod
    def _step(
        phase: str,
        system_id: str,
        resource_id: str,
        operation: str,
        state: str,
        evidence_id: str,
    ) -> PlanStep:
        if operation not in OPERATION_ORDER:
            raise ValueError(f"Operation is not allowed by the challenge contract: {operation}")
        return PlanStep(0, phase, system_id, resource_id, operation, state, evidence_id)

    @staticmethod
    def _canonical_system(record: EvidenceRecord) -> str:
        context = record.source_context.casefold()
        for marker, system_id in _SYSTEM_CONTEXT_MAP:
            if marker in context:
                return system_id
        field_system = record.fields.get("system")
        if isinstance(field_system, str) and field_system.strip():
            return field_system.strip()
        raw_system = RemediationPlanner._normalize_system(record.system_id)
        return raw_system

    @staticmethod
    def _normalize_system(system_id: str) -> str:
        raw_system = system_id.strip()
        if raw_system.endswith("_capture"):
            raw_system = raw_system.removesuffix("_capture")
        capture_aliases = {
            "customer_account_service": "account_service",
            "customer_account_failover_replica": "account_service",
            "customer_data_inventory": "customer_data_service",
            "customer_data_read_replica": "customer_data_service",
            "customer_data_reporting_replica": "customer_data_service",
            "customer_data_regional_replicas": "customer_data_service",
            "customer_data_failover_replica": "customer_data_service",
        }
        return capture_aliases.get(raw_system, raw_system)

    def _claim_systems(self, record: EvidenceRecord) -> set[str]:
        systems = {self._canonical_system(record)}
        for key in ("source_system", "target_system", "system_id", "system"):
            value = record.fields.get(key)
            if isinstance(value, str) and value.strip():
                systems.add(self._normalize_system(value))
        return systems

    @staticmethod
    def _leaf(path: str) -> str:
        return path.rsplit(".", 1)[-1].split("[", 1)[0].casefold()

    @staticmethod
    def _scalar_items(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
        if isinstance(value, Mapping):
            for key, child in value.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                yield from RemediationPlanner._scalar_items(child, path)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                yield from RemediationPlanner._scalar_items(child, f"{prefix}[{index}]")
        elif value is not None and not isinstance(value, (bytes, bytearray)):
            yield prefix, value

    def _sort_steps(self, steps: Iterable[PlanStep]) -> list[PlanStep]:
        unique_steps = list(dict.fromkeys(steps))
        return sorted(
            unique_steps,
            key=lambda step: (
                PHASE_ORDER[step.phase],
                OPERATION_ORDER[step.operation],
                step.system_id,
                step.resource_id,
                step.evidence_id,
            ),
        )

    def _validate_steps(self, steps: tuple[PlanStep, ...], status: str | None, case_id: str) -> None:
        for expected_number, step in enumerate(steps, start=1):
            if step.step_no != expected_number:
                raise ValueError(f"Nonconsecutive step numbering in {case_id}")
            if set(step.as_dict()) != STEP_FIELDS:
                raise ValueError(f"Invalid output step fields in {case_id}")
            if (step.phase, step.operation, step.state) not in ALLOWED_OPERATION_STATES:
                raise ValueError(f"Disallowed operation/state combination in {case_id}: {step.as_dict()}")
            if step.evidence_id and step.evidence_id not in self._valid_evidence_ids:
                raise ValueError(f"Unknown evidence ID {step.evidence_id!r} in {case_id}")
            if status == "COMPLETE":
                if step.phase != "VERIFICATION" or not step.evidence_id:
                    raise ValueError(f"COMPLETE plan {case_id} must have evidence-backed verification steps only")
            elif status == "INCOMPLETE":
                if step.phase == "FINDING" and not step.evidence_id:
                    raise ValueError(f"Finding in {case_id} must cite actual evidence")
                if step.phase in {"REMEDIATION", "VERIFICATION"} and step.evidence_id:
                    raise ValueError(f"Future steps in {case_id} must have an empty evidence_id")
            elif status == "NOT_APPLICABLE" and steps:
                raise ValueError(f"NOT_APPLICABLE plan {case_id} must have no steps")

        if status == "COMPLETE" and any(step.phase != "VERIFICATION" for step in steps):
            raise ValueError(f"COMPLETE plan {case_id} contains non-verification steps")
        if status == "INCOMPLETE":
            if not any(step.phase == "FINDING" for step in steps):
                raise ValueError(f"INCOMPLETE plan {case_id} has no evidence-backed findings")
            if not any(step.phase == "REMEDIATION" for step in steps):
                raise ValueError(f"INCOMPLETE plan {case_id} has no remediation actions")
            if not any(step.phase == "VERIFICATION" for step in steps):
                raise ValueError(f"INCOMPLETE plan {case_id} has no future verification")


def classify_request_intent(request_text: str) -> RequestIntent:
    """Convenience wrapper for the conservative deletion-intent classifier."""
    return RequestIntentClassifier().classify(request_text)
