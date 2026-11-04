"""Apply the supplied deletion rules to observed evidence records.

This module evaluates policy and observed state only. It does not create final
challenge output, make request-intent decisions, or trust activity claims.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime, time, timezone
from enum import StrEnum
from typing import Any, Iterable, Mapping

from data_loader import DatasetValidationError, parse_iso_timestamp
from evidence_scanner import EvidenceRecord
from identity_resolver import IdentityResolver


POLICY_COLUMNS = ("rule_id", "data_category", "condition", "required_action")
POLICY_ACTIONS = {"DELETE", "REDACT", "ANONYMIZE", "RETAIN"}
DIRECT_IDENTIFIER_FIELDS = {
    "email": "email",
    "email_address": "email",
    "customer_email": "email",
    "requester_email": "email",
    "owner_email": "email",
    "phone": "phone",
    "phone_number": "phone",
    "customer_phone": "phone",
    "requester_phone": "phone",
    "owner_phone": "phone",
    "name": "name",
    "full_name": "name",
    "display_name": "name",
    "requester_name": "name",
    "customer_name": "name",
    "email_hash": "email_hash",
    "stable_email_hash": "email_hash",
}
_LINK_IDENTIFIER_FIELDS = {
    "user_id", "subject_reference", "customer_reference", "stable_user_id",
}
_COLLECTION_CATEGORIES = {
    "customer_profiles": "USER_PROFILE",
    "customer_search_documents": "SEARCH_INDEX_ENTRY",
    "customer_analytics_profiles": "ANALYTICS_PROFILE",
    "customer_analytics_reports": "ANALYTICS_PROFILE",
    "support_records": "SUPPORT_MESSAGE",
    "support_conversations": "SUPPORT_MESSAGE",
    "ticket_attachments": "SUPPORT_ATTACHMENT",
    "customer_exports": "USER_EXPORT",
    "customer_export_jobs": "USER_EXPORT",
    "connector_subscriptions": "DATA_FLOW",
    "connector_configurations": "DATA_FLOW",
    "backup_restore_entries": "BACKUP_ARCHIVE",
    "backup_catalog": "BACKUP_ARCHIVE",
    "security_audit_records": "SECURITY_AUDIT_LOG",
    "security_audit_events": "SECURITY_AUDIT_LOG",
    "deleted_customer_references": "ACCOUNT_TOMBSTONE",
    "marketing_profiles": "MARKETING_PROFILE",
    "email_delivery_logs": "EMAIL_DELIVERY_LOG",
    "customer_cache_entries": "CACHE_ENTRY",
    "product_telemetry": "PRODUCT_TELEMETRY",
}
_TABLE_CATEGORIES = {
    "customer_profiles": "USER_PROFILE",
    "customer_search_documents": "SEARCH_INDEX_ENTRY",
    "search_documents_primary": "SEARCH_INDEX_ENTRY",
    "customer_analytics_profiles": "ANALYTICS_PROFILE",
    "support_records": "SUPPORT_MESSAGE",
    "customer_exports": "USER_EXPORT",
    "connector_subscriptions": "DATA_FLOW",
    "backup_restore_entries": "BACKUP_ARCHIVE",
    "security_audit_records": "SECURITY_AUDIT_LOG",
}
_RISK_CATEGORIES = {"DATA_FLOW", "RESTORE_PATH"}


class DecisionStatus(StrEnum):
    """Result of evaluating one record against one policy rule."""

    COMPLIANT = "COMPLIANT"
    NONCOMPLIANT = "NONCOMPLIANT"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    NEEDS_CONTEXT = "NEEDS_CONTEXT"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


@dataclass(frozen=True, slots=True)
class PolicyRule:
    """One rule read from deletion_rules.csv."""

    rule_id: str
    data_category: str
    condition: str
    required_action: str


@dataclass(frozen=True, slots=True)
class PolicyContext:
    """Optional case facts that evidence alone may not establish."""

    account_deleted: bool | None = None
    stable_user_id_required: bool | None = None
    support_shared_context: bool | None = None
    attachment_belongs_only_to_subject: bool | None = None
    record_attributable: bool | None = None
    within_retention_period: bool | None = None
    legal_hold_active: bool | None = None
    restore_access_blockable: bool | None = None


@dataclass(frozen=True, slots=True)
class ExposedIdentifier:
    """A field containing a subject identifier; raw values are not copied here."""

    field_path: str
    identifier_type: str


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """Evidence-backed policy result for a record as of a cutoff timestamp."""

    status: DecisionStatus
    rule_id: str | None
    data_category: str
    required_action: str | None
    system_id: str
    resource_id: str
    evidence_id: str
    source_file: str
    observed_at: datetime | None
    observed_state: str
    reason: str
    exposed_identifiers: tuple[ExposedIdentifier, ...] = ()
    conflicting_evidence_ids: tuple[str, ...] = ()


class PolicyConfigurationError(ValueError):
    """Raised when policy rows are incomplete, malformed, or ambiguous."""


class PolicyEngine:
    """Evaluate records against the actual rules supplied by the challenge."""

    def __init__(
        self,
        rules: Iterable[Mapping[str, str] | PolicyRule],
        identity_resolver: IdentityResolver,
    ) -> None:
        self.identity_resolver = identity_resolver
        self.rules: tuple[PolicyRule, ...] = self._load_rules(rules)
        self._rules_by_category = {rule.data_category: rule for rule in self.rules}

    @classmethod
    def from_dataset(
        cls,
        deletion_rules: Iterable[Mapping[str, str]],
        identity_resolver: IdentityResolver,
    ) -> PolicyEngine:
        """Construct an engine from ``LoadedDataset.deletion_rules``."""
        return cls(deletion_rules, identity_resolver)

    @staticmethod
    def _load_rules(rules: Iterable[Mapping[str, str] | PolicyRule]) -> tuple[PolicyRule, ...]:
        parsed: list[PolicyRule] = []
        for row_number, raw_rule in enumerate(rules, start=1):
            if isinstance(raw_rule, PolicyRule):
                rule = raw_rule
            else:
                if tuple(raw_rule.keys()) != POLICY_COLUMNS:
                    raise PolicyConfigurationError(
                        f"Policy row {row_number} must have columns {POLICY_COLUMNS} in that order"
                    )
                rule = PolicyRule(
                    rule_id=str(raw_rule["rule_id"]).strip(),
                    data_category=PolicyEngine._normalize_category(raw_rule["data_category"]),
                    condition=str(raw_rule["condition"]).strip(),
                    required_action=str(raw_rule["required_action"]).strip().upper(),
                )
            if not rule.rule_id or not rule.data_category or not rule.condition:
                raise PolicyConfigurationError(f"Policy row {row_number} has an empty required value")
            if rule.required_action not in POLICY_ACTIONS:
                raise PolicyConfigurationError(
                    f"Policy rule {rule.rule_id} has unsupported action {rule.required_action!r}"
                )
            parsed.append(rule)

        ids = [rule.rule_id for rule in parsed]
        if len(ids) != len(set(ids)):
            raise PolicyConfigurationError("Policy rule_id values must be unique")
        categories = [rule.data_category for rule in parsed]
        if len(categories) != len(set(categories)):
            raise PolicyConfigurationError("Each data_category must have exactly one supplied rule")
        if not parsed:
            raise PolicyConfigurationError("At least one deletion policy rule is required")
        return tuple(parsed)

    @staticmethod
    def _normalize_category(value: str) -> str:
        return "_".join(str(value).strip().replace("-", " ").split()).upper()

    def classify_record(self, record: EvidenceRecord) -> str:
        """Map an evidence record to the policy category stated in its contents."""
        for field_name in ("data_category", "record_type", "category"):
            value = record.fields.get(field_name)
            if isinstance(value, str) and value.strip():
                normalized = self._normalize_category(value)
                if normalized in self._rules_by_category or normalized in _RISK_CATEGORIES:
                    return normalized

        collection = record.fields.get("collection_name")
        if isinstance(collection, str):
            category = _COLLECTION_CATEGORIES.get(collection.strip().casefold())
            if category:
                return category

        context = record.source_context.casefold()
        context_categories = (
            ("account_service profile_lookups", "USER_PROFILE"),
            ("analytics_platform profiles", "ANALYTICS_PROFILE"),
            ("backup_service restore_entries", "BACKUP_ARCHIVE"),
            ("export_service exports", "USER_EXPORT"),
            ("integration_service connectors", "DATA_FLOW"),
            ("search_service index_queries", "SEARCH_INDEX_ENTRY"),
            ("search_documents_primary", "SEARCH_INDEX_ENTRY"),
            ("security_service audit_records", "SECURITY_AUDIT_LOG"),
            ("support_platform tickets", "SUPPORT_MESSAGE"),
            ("marketing_platform", "MARKETING_PROFILE"),
            ("cache", "CACHE_ENTRY"),
            ("telemetry", "PRODUCT_TELEMETRY"),
            ("email_delivery", "EMAIL_DELIVERY_LOG"),
            ("notification_delivery", "EMAIL_DELIVERY_LOG"),
        )
        for marker, category in context_categories:
            if marker in context:
                return category

        source_name = record.source_file.replace("\\", "/").casefold()
        for marker, category in _COLLECTION_CATEGORIES.items():
            if marker in source_name:
                return category
        if "notification_delivery" in source_name or "email_delivery" in source_name:
            return "EMAIL_DELIVERY_LOG"

        keys = self._field_names(record.fields)
        if {"retention_reason", "order_reference"}.issubset(keys):
            return "LEGAL_HOLD_RECORD"
        if {"actor_type", "event_name", "integrity_seal"}.issubset(keys):
            return "SECURITY_AUDIT_LOG"
        if {"event_code", "retention_until", "subject_reference"}.issubset(keys):
            return "SECURITY_AUDIT_LOG"
        if "restore_allowed" in keys or "archive_path" in keys:
            return "BACKUP_ARCHIVE"
        if "ticket_id" in keys and "file_name" in keys:
            return "SUPPORT_ATTACHMENT"
        if {"requester_email", "message_excerpt"}.issubset(keys) or {"message_text", "channel"}.issubset(keys):
            return "SUPPORT_MESSAGE"
        if {"hits", "query"}.issubset(keys) or "index_name" in keys or "document" in keys:
            return "SEARCH_INDEX_ENTRY"
        if {"connector_name", "enabled"}.issubset(keys):
            return "DATA_FLOW"
        if {"metrics", "recent_activity"}.issubset(keys) or "reporting_window" in keys:
            return "ANALYTICS_PROFILE"
        if {"account_status", "email", "phone"}.issubset(keys) and ("full_name" in keys or "name" in keys):
            return "ACCOUNT_TOMBSTONE" if str(record.fields.get("account_status", "")).casefold() in {"deleted", "tombstone"} else "USER_PROFILE"
        if {"owner_email", "owner_phone"}.issubset(keys) or {"owner_email", "download_available"}.issubset(keys):
            return "USER_EXPORT"
        if {"audience_cohort", "email"}.issubset(keys):
            return "MARKETING_PROFILE" if "marketing" in record.system_id.casefold() else "ANALYTICS_PROFILE"

        table_name = record.resource_id.split(":", 1)[0].casefold()
        category = _TABLE_CATEGORIES.get(table_name)
        if category:
            return category
        # SQL table names often appear with a row identifier as the resource ID.
        for table_name, category in _TABLE_CATEGORIES.items():
            if record.resource_id.startswith(table_name + ":"):
                return category
        if record.category == "data_flow":
            return "DATA_FLOW"
        if record.category == "restore_path":
            return "RESTORE_PATH"
        return "UNKNOWN"

    def evaluate_record(
        self,
        record: EvidenceRecord,
        *,
        user_id: str,
        as_of: datetime,
        context: PolicyContext | None = None,
    ) -> PolicyDecision:
        """Evaluate one direct observation against policy at ``as_of``.

        Claim-only records, missing observation times, and post-cutoff evidence
        cannot establish compliant or noncompliant resource state.
        """
        self._validate_cutoff(as_of)
        policy_context = context or PolicyContext()
        category = self.classify_record(record)
        if record.observed_at is not None and record.observed_at > as_of:
            return self._decision(
                record, category, DecisionStatus.OUT_OF_SCOPE, None, None,
                "This evidence was captured after the audit cutoff and is excluded.",
            )
        if record.evidence_kind != "observation":
            return self._decision(
                record, category, DecisionStatus.UNKNOWN, None, None,
                "Activity or operation results are claims, not direct observations of resource state.",
            )
        if record.observed_at is None:
            return self._decision(
                record, category, DecisionStatus.UNKNOWN, None, None,
                "The observation has no usable timestamp, so its state at the audit cutoff is unknown.",
            )

        if category in _RISK_CATEGORIES:
            return self._evaluate_recreation_risk(record, category)

        rule = self._rules_by_category.get(category)
        if rule is None:
            return self._decision(
                record, category, DecisionStatus.NEEDS_CONTEXT, None, None,
                "No supplied deletion rule identifies how this evidence category should be treated.",
            )

        condition_met = self._condition_met(rule, record, user_id, as_of, policy_context)
        if condition_met is None:
            return self._decision(
                record, category, DecisionStatus.NEEDS_CONTEXT, rule, None,
                f"The supplied condition for {rule.rule_id} cannot be confirmed from this record; do not guess.",
            )
        if not condition_met:
            return self._decision(
                record, category, DecisionStatus.NOT_APPLICABLE, rule, None,
                f"The condition for {rule.rule_id} is not met; this rule does not apply to this record.",
            )

        return self._assess_action(record, user_id, category, rule, as_of, policy_context)

    def evaluate_resource(
        self,
        records: Iterable[EvidenceRecord],
        *,
        user_id: str,
        system_id: str,
        resource_id: str,
        as_of: datetime,
        context: PolicyContext | None = None,
    ) -> PolicyDecision:
        """Reconcile repeated evidence and evaluate the latest direct observation.

        Later events are ignored. Claims are excluded. If the newest direct
        observations at the same timestamp disagree, the result is UNKNOWN.
        """
        self._validate_cutoff(as_of)
        candidates = [
            record for record in records
            if record.system_id == system_id
            and record.resource_id == resource_id
            and record.evidence_kind == "observation"
            and record.observed_at is not None
            and record.observed_at <= as_of
            and user_id in record.matched_user_ids
        ]
        if not candidates:
            return PolicyDecision(
                status=DecisionStatus.UNKNOWN,
                rule_id=None,
                data_category="UNKNOWN",
                required_action=None,
                system_id=system_id,
                resource_id=resource_id,
                evidence_id="",
                source_file="",
                observed_at=None,
                observed_state="UNKNOWN",
                reason="No identity-matched direct observation exists at or before the audit cutoff.",
            )

        latest_time = max(record.observed_at for record in candidates if record.observed_at is not None)
        latest = [record for record in candidates if record.observed_at == latest_time]
        states = {record.observed_state for record in latest}
        if len(states) != 1:
            first = latest[0]
            return self._decision(
                first,
                self.classify_record(first),
                DecisionStatus.UNKNOWN,
                self._rules_by_category.get(self.classify_record(first)),
                latest_time,
                "Conflicting direct observations at the same timestamp disagree about resource state.",
                conflicting_evidence_ids=tuple(sorted(record.evidence_id for record in latest)),
            )
        # Duplicate same-state captures are equivalent for policy assessment.
        return self.evaluate_record(latest[-1], user_id=user_id, as_of=as_of, context=context)

    def evaluate_recreation_risks(
        self,
        record: EvidenceRecord,
        *,
        as_of: datetime,
    ) -> tuple[PolicyDecision, ...]:
        """Evaluate sync/restore controls separately from a record's data policy.

        This permits a backup archive to be retained by policy while its restore
        path is separately found active and requires blocking.
        """
        self._validate_cutoff(as_of)
        if record.evidence_kind != "observation" or record.observed_at is None or record.observed_at > as_of:
            return ()

        values = {self._leaf(path): value for path, value in self._scalar_items(record.fields)}
        declared_type = self._normalize_category(str(record.fields.get("record_type", "")))
        context_words = " ".join((
            record.source_file,
            str(record.fields.get("collection_name", "")),
            str(record.fields.get("connector_name", "")),
        )).casefold()
        decisions: list[PolicyDecision] = []

        restore_value = next((values[key] for key in (
            "restore_allowed", "can_restore", "restore_enabled",
        ) if key in values), None)
        restore_context = declared_type in {"BACKUP_ARCHIVE", "RESTORE_PATH"} or record.category == "restore_path" or any(
            marker in context_words for marker in ("backup", "restore", "archive")
        )
        if restore_context and (restore_value is not None or record.observed_state in {"ACTIVE", "BLOCKED"}):
            allowed = self._as_boolean(restore_value) if restore_value is not None else record.observed_state == "ACTIVE"
            state = "ACTIVE" if allowed is True else "BLOCKED" if allowed is False else "UNKNOWN"
            decisions.append(self._evaluate_recreation_risk(self._with_state(record, state), "RESTORE_PATH"))

        enabled_value = next((values[key] for key in (
            "enabled", "is_enabled", "active", "is_active",
        ) if key in values), None)
        flow_context = declared_type == "DATA_FLOW" or record.category == "data_flow" or any(
            marker in context_words for marker in (
                "connector", "subscription", "sync", "import", "export", "integration", "data_flow",
            )
        )
        if flow_context and (enabled_value is not None or record.observed_state in {"ACTIVE", "DISABLED"}):
            enabled = self._as_boolean(enabled_value) if enabled_value is not None else record.observed_state == "ACTIVE"
            state = "ACTIVE" if enabled is True else "DISABLED" if enabled is False else "UNKNOWN"
            decisions.append(self._evaluate_recreation_risk(self._with_state(record, state), "DATA_FLOW"))

        if not decisions and record.category in _RISK_CATEGORIES:
            decisions.append(self._evaluate_recreation_risk(record, record.category.upper()))
        return tuple(decisions)

    def _evaluate_recreation_risk(self, record: EvidenceRecord, category: str) -> PolicyDecision:
        if category == "DATA_FLOW":
            if record.observed_state == "ACTIVE":
                return self._decision(
                    record, category, DecisionStatus.NONCOMPLIANT, None, "DISABLE_DATA_FLOW",
                    "A directly observed active process can copy or recreate this user's data.",
                )
            if record.observed_state == "DISABLED":
                return self._decision(
                    record, category, DecisionStatus.COMPLIANT, None, "DISABLE_DATA_FLOW",
                    "The directly observed data flow is disabled.",
                )
            return self._decision(
                record, category, DecisionStatus.UNKNOWN, None, "DISABLE_DATA_FLOW",
                "The process state does not establish whether the data flow is active.",
            )

        if record.observed_state == "ACTIVE":
            return self._decision(
                record, category, DecisionStatus.NONCOMPLIANT, None, "BLOCK_RESTORE",
                "A directly observed restore path can recreate this user's data.",
            )
        if record.observed_state == "BLOCKED":
            return self._decision(
                record, category, DecisionStatus.COMPLIANT, None, "BLOCK_RESTORE",
                "The directly observed restore path is blocked.",
            )
        return self._decision(
            record, category, DecisionStatus.UNKNOWN, None, "BLOCK_RESTORE",
            "The evidence does not establish whether this restore path is blocked.",
        )

    def _assess_action(
        self,
        record: EvidenceRecord,
        user_id: str,
        category: str,
        rule: PolicyRule,
        as_of: datetime,
        context: PolicyContext,
    ) -> PolicyDecision:
        exposed = self._exposed_identifiers(record, user_id, category)
        state = record.observed_state.upper()
        if rule.required_action == "DELETE":
            status = DecisionStatus.COMPLIANT if state == "ABSENT" else (
                DecisionStatus.NONCOMPLIANT if state == "PRESENT" else DecisionStatus.UNKNOWN
            )
            reason = {
                DecisionStatus.COMPLIANT: "The directly observed record is absent as required.",
                DecisionStatus.NONCOMPLIANT: "The record is still present but policy requires deletion.",
                DecisionStatus.UNKNOWN: "The evidence does not establish whether the record is absent.",
            }[status]
        elif rule.required_action == "REDACT":
            if state == "ABSENT":
                status = DecisionStatus.NONCOMPLIANT
                reason = "Policy requires this record to remain, but the record is absent."
            elif state == "REDACTED" and not exposed:
                status = DecisionStatus.COMPLIANT
                reason = "A direct readback reports the record redacted and no prohibited subject identifiers were found."
            elif state != "PRESENT":
                status = DecisionStatus.UNKNOWN
                reason = "The evidence does not establish whether the retained record is present."
            elif exposed:
                status = DecisionStatus.NONCOMPLIANT
                reason = "The record remains, but subject identifiers prohibited by the redaction rule are exposed."
            else:
                status = DecisionStatus.COMPLIANT
                reason = "The record remains and no prohibited subject identifiers were found."
        elif rule.required_action == "ANONYMIZE":
            if state == "ABSENT":
                status = DecisionStatus.NONCOMPLIANT
                reason = "Policy requires an anonymized record to remain, but the record is absent."
            elif state == "ANONYMIZED" and not exposed:
                status = DecisionStatus.COMPLIANT
                reason = "A direct readback reports the retained record anonymized with no exposed subject identifiers."
            elif state != "PRESENT":
                status = DecisionStatus.UNKNOWN
                reason = "The evidence does not establish whether the record is present and anonymized."
            elif user_id in record.matched_user_ids or exposed:
                status = DecisionStatus.NONCOMPLIANT
                reason = "The retained record can still be linked to the subject or contains their identifier."
            else:
                status = DecisionStatus.COMPLIANT
                reason = "The record remains but is not linked to the subject by the known identifiers."
        else:  # RETAIN
            if state == "PRESENT":
                status = DecisionStatus.COMPLIANT
                reason = "The record is present as required by the supplied retention rule."
            elif state == "ABSENT":
                status = DecisionStatus.NONCOMPLIANT
                reason = "The supplied retention rule requires this record to remain, but it is absent."
            else:
                status = DecisionStatus.UNKNOWN
                reason = "The evidence does not establish whether the policy-required retained record is present."

        return self._decision(record, category, status, rule, rule.required_action, reason, exposed)

    def _condition_met(
        self,
        rule: PolicyRule,
        record: EvidenceRecord,
        user_id: str,
        as_of: datetime,
        context: PolicyContext,
    ) -> bool | None:
        condition = rule.condition.strip().casefold()
        if condition == "always":
            return True

        if condition == "conversation contains shared customer, agent, or order context that must be retained":
            if context.support_shared_context is not None:
                return context.support_shared_context
            fields = record.fields
            if any(fields.get(key) for key in (
                "shared_context", "shared_customer_context", "order_reference", "order_id",
                "agent_id", "agent_name", "agent_reference", "assigned_agent",
            )):
                return True
            if fields.get("message_text") and fields.get("channel"):
                return True
            return None

        if condition == "attachment belongs only to the deletion subject":
            if context.attachment_belongs_only_to_subject is not None:
                return context.attachment_belongs_only_to_subject
            owners: set[str] = set()
            for key in ("owner_user_id", "user_id", "subject_reference", "customer_reference", "uploaded_by"):
                value = record.fields.get(key)
                if isinstance(value, str):
                    owners.update(self.identity_resolver.resolve(value, "user_id"))
            if owners:
                return owners == {user_id}
            if user_id in record.matched_user_ids:
                return record.matched_user_ids == {user_id}
            return None

        if condition == "record is still attributable to a user":
            if context.record_attributable is not None:
                # The caller explicitly identified the record as relevant, but
                # confirmed the required anonymization has already unlinked it.
                if rule.required_action == "ANONYMIZE" and context.record_attributable is False:
                    return True
                return context.record_attributable
            return user_id in record.matched_user_ids

        if condition == "log contains a user email address":
            if self._contains_known_identifier(record, user_id, "email"):
                return True
            if user_id in record.matched_user_ids:
                return None
            return False

        if condition == "record is within the supplied retention period":
            if context.within_retention_period is not None:
                return context.within_retention_period
            for key in ("retention_until", "retained_until", "expires_at"):
                value = record.fields.get(key)
                if isinstance(value, str) and value.strip():
                    expiry = self._parse_date_or_timestamp(value)
                    if expiry is not None:
                        return expiry >= as_of
            return None

        if condition == "a supplied legal hold is active":
            if context.legal_hold_active is not None:
                return context.legal_hold_active
            for key in ("legal_hold_active", "hold_active", "is_on_legal_hold"):
                value = record.fields.get(key)
                if value is not None:
                    return self._as_boolean(value)
            status = str(record.fields.get("hold_status", "")).strip().casefold()
            if status in {"active", "open", "in_force"}:
                return True
            if status in {"inactive", "released", "closed", "expired"}:
                return False
            return None

        if condition == "account is deleted and only a stable internal user_id is needed to preserve references to retained records":
            account_deleted = context.account_deleted
            if account_deleted is None:
                status = str(record.fields.get("account_status", "")).strip().casefold()
                account_deleted = (
                    self.classify_record(record) == "ACCOUNT_TOMBSTONE"
                    or status in {"deleted", "closed", "tombstone", "inactive_deleted"}
                )
            stable_reference_required = context.stable_user_id_required
            if stable_reference_required is None:
                stable_reference_required = any(
                    isinstance(record.fields.get(key), str) and bool(record.fields[key].strip())
                    for key in ("subject_reference", "stable_user_id", "user_id", "customer_reference")
                )
            if account_deleted is None or stable_reference_required is None:
                return None
            return account_deleted and stable_reference_required

        if condition == "restore access can be blocked":
            if context.restore_access_blockable is not None:
                return context.restore_access_blockable
            for key in ("restore_access_blockable", "can_block_restore", "restore_block_supported"):
                value = record.fields.get(key)
                if value is not None:
                    return self._as_boolean(value)
            # restore_allowed only says whether restore is currently possible;
            # it does not prove that the control can block future restores.
            return None

        return None

    def _contains_known_identifier(self, record: EvidenceRecord, user_id: str, kind: str) -> bool:
        identifiers = self.identity_resolver.identifiers_for(user_id).get(kind, set())
        for path, value in self._scalar_items(record.fields):
            text = str(value).strip()
            field_kind = DIRECT_IDENTIFIER_FIELDS.get(self._leaf(path))
            if field_kind == kind and user_id in self.identity_resolver.resolve(text, kind):
                return True
            if field_kind is None and any(
                self._contains_literal(text, identifier, kind) for identifier in identifiers
            ):
                return True
        return False

    def _exposed_identifiers(
        self,
        record: EvidenceRecord,
        user_id: str,
        category: str,
    ) -> tuple[ExposedIdentifier, ...]:
        exposed: set[ExposedIdentifier] = set()
        is_tombstone = category == "ACCOUNT_TOMBSTONE"
        subject_linked = user_id in record.matched_user_ids
        for path, value in self._scalar_items(record.fields):
            text = str(value).strip()
            if not text:
                continue
            leaf = self._leaf(path)
            identifier_type = DIRECT_IDENTIFIER_FIELDS.get(leaf)
            if identifier_type:
                if is_tombstone and subject_linked:
                    # Tombstones preserve only stable internal references; every
                    # populated direct identifier field must be redacted.
                    exposed.add(ExposedIdentifier(path, identifier_type))
                elif user_id in self.identity_resolver.resolve(text, identifier_type):
                    exposed.add(ExposedIdentifier(path, identifier_type))
                elif user_id in self.identity_resolver.match_text(text):
                    exposed.add(ExposedIdentifier(path, identifier_type))
            elif leaf in _LINK_IDENTIFIER_FIELDS:
                if is_tombstone:
                    continue  # stable user references are permitted on tombstones
                if user_id in self.identity_resolver.resolve(text, "user_id"):
                    exposed.add(ExposedIdentifier(path, "user_reference"))
            else:
                if user_id in self.identity_resolver.match_text(text):
                    exposed.add(ExposedIdentifier(path, "text_identifier"))
        return tuple(sorted(exposed, key=lambda exposure: (exposure.field_path, exposure.identifier_type)))

    @staticmethod
    def _contains_literal(text: str, identifier: str, kind: str) -> bool:
        if kind == "email_hash":
            return identifier.casefold() in text.casefold()
        if kind == "email":
            return identifier.casefold() in text.casefold()
        if kind == "phone":
            normalize_digits = lambda value: "".join(char for char in value if char.isdigit())
            normalized = normalize_digits(identifier)
            return bool(normalized) and normalized in normalize_digits(text)
        return " ".join(text.casefold().split()).find(" ".join(identifier.casefold().split())) >= 0

    @staticmethod
    def _leaf(path: str) -> str:
        return path.rsplit(".", 1)[-1].split("[", 1)[0].casefold()

    @staticmethod
    def _scalar_items(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
        if isinstance(value, Mapping):
            for key, child in value.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                yield from PolicyEngine._scalar_items(child, path)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                yield from PolicyEngine._scalar_items(child, f"{prefix}[{index}]")
        elif value is not None and not isinstance(value, (bytes, bytearray)):
            yield prefix, value

    @staticmethod
    def _field_names(value: Any) -> set[str]:
        """Return declared leaf names, including fields whose value is null."""
        names: set[str] = set()
        if isinstance(value, Mapping):
            for key, child in value.items():
                names.add(PolicyEngine._leaf(str(key)))
                names.update(PolicyEngine._field_names(child))
        elif isinstance(value, (list, tuple)):
            for child in value:
                names.update(PolicyEngine._field_names(child))
        return names

    @staticmethod
    def _as_boolean(value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        normalized = str(value).strip().casefold()
        if normalized in {"true", "yes", "1", "active", "enabled", "allowed", "open"}:
            return True
        if normalized in {"false", "no", "0", "inactive", "disabled", "blocked", "released", "closed"}:
            return False
        return None

    @staticmethod
    def _parse_date_or_timestamp(value: str) -> datetime | None:
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            try:
                parsed_date = date.fromisoformat(value.strip())
            except ValueError:
                return None
            parsed = datetime.combine(parsed_date, time.min, tzinfo=timezone.utc)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed

    @staticmethod
    def _validate_cutoff(as_of: datetime) -> None:
        if as_of.tzinfo is None:
            raise ValueError("as_of must include a time zone")

    @staticmethod
    def _decision(
        record: EvidenceRecord,
        category: str,
        status: DecisionStatus,
        rule: PolicyRule | None,
        required_action: str | None,
        reason: str,
        exposed: tuple[ExposedIdentifier, ...] = (),
        *,
        conflicting_evidence_ids: tuple[str, ...] = (),
    ) -> PolicyDecision:
        return PolicyDecision(
            status=status,
            rule_id=rule.rule_id if rule else None,
            data_category=category,
            required_action=required_action or (rule.required_action if rule else None),
            system_id=record.system_id,
            resource_id=record.resource_id,
            evidence_id=record.evidence_id,
            source_file=record.source_file,
            observed_at=record.observed_at,
            observed_state=record.observed_state,
            reason=reason,
            exposed_identifiers=exposed,
            conflicting_evidence_ids=conflicting_evidence_ids,
        )

    @staticmethod
    def _with_state(record: EvidenceRecord, state: str) -> EvidenceRecord:
        """Return a shallow evidence copy with a condition-derived risk state."""
        return replace(record, observed_state=state)
