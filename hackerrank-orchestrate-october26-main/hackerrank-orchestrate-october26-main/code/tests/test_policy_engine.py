"""Tests for the data-driven deletion policy engine."""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from data_loader import load_dataset  # noqa: E402
from evidence_scanner import EvidenceRecord  # noqa: E402
from identity_resolver import IdentityResolver  # noqa: E402
from policy_engine import (  # noqa: E402
    DecisionStatus,
    PolicyConfigurationError,
    PolicyContext,
    PolicyEngine,
)


CUTOFF = datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)
USER_ID = "user_test"


class PolicyEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dataset = load_dataset()
        identities = [{
            "user_id": USER_ID,
            "name": "Test Customer",
            "email": "current@example.test",
            "phone": "+1 555 010 0100",
            "email_hash": "a" * 64,
            "updated_at": "2026-01-01T00:00:00Z",
        }]
        audit = [{
            "user_id": USER_ID,
            "event_time": "2025-12-01T00:00:00Z",
            "field_name": "email",
            "old_value": "historical@example.test",
            "new_value": "current@example.test",
        }]
        cls.resolver = IdentityResolver(identities, audit)
        cls.engine = PolicyEngine.from_dataset(dataset.deletion_rules, cls.resolver)
        cls.rules = dataset.deletion_rules

    def _record(
        self,
        *,
        category: str,
        state: str = "PRESENT",
        fields: dict[str, object] | None = None,
        matched: set[str] | None = None,
        resource_id: str = "resource_1",
        system_id: str = "account_service",
        evidence_id: str = "evidence_test",
        observed_at: datetime = datetime(2026, 9, 1, tzinfo=timezone.utc),
        kind: str = "observation",
        source_file: str = "evidence/test.json",
    ) -> EvidenceRecord:
        return EvidenceRecord(
            evidence_id=evidence_id,
            source_file=source_file,
            file_format=Path(source_file).suffix,
            system_id=system_id,
            resource_id=resource_id,
            observed_at=observed_at,
            timestamps={"captured_at": observed_at.isoformat()},
            observed_state=state,
            category="personal_record",
            topology="primary",
            evidence_kind=kind,
            matched_user_ids=set(matched or ()),
            fields={"record_type": category, **(fields or {})},
        )

    def test_reads_all_actual_dataset_rules_and_categories(self) -> None:
        self.assertEqual(len(self.engine.rules), 15)
        self.assertEqual(
            {rule.data_category for rule in self.engine.rules},
            {row["data_category"] for row in self.rules},
        )
        self.assertEqual(
            {rule.required_action for rule in self.engine.rules},
            {"DELETE", "REDACT", "ANONYMIZE", "RETAIN"},
        )
        for rule in self.engine.rules:
            with self.subTest(category=rule.data_category):
                record = self._record(category=rule.data_category)
                self.assertEqual(self.engine.classify_record(record), rule.data_category)

    def test_classifies_service_export_contexts_without_case_labels(self) -> None:
        contexts = {
            "service_exports account_service profile_lookups": "USER_PROFILE",
            "service_exports analytics_platform profiles": "ANALYTICS_PROFILE",
            "service_exports backup_service restore_entries": "BACKUP_ARCHIVE",
            "service_exports export_service exports": "USER_EXPORT",
            "service_exports integration_service connectors": "DATA_FLOW",
            "service_exports search_service index_queries": "SEARCH_INDEX_ENTRY",
            "service_exports security_service audit_records": "SECURITY_AUDIT_LOG",
            "service_exports support_platform tickets": "SUPPORT_MESSAGE",
        }
        for source_context, expected in contexts.items():
            with self.subTest(source_context=source_context):
                record = self._record(category="UNLABELED", fields={"resource_id": "record_1"})
                record = EvidenceRecord(
                    evidence_id=record.evidence_id,
                    source_file=record.source_file,
                    file_format=record.file_format,
                    system_id=record.system_id,
                    resource_id=record.resource_id,
                    observed_at=record.observed_at,
                    timestamps=record.timestamps,
                    observed_state=record.observed_state,
                    category=record.category,
                    topology=record.topology,
                    evidence_kind=record.evidence_kind,
                    matched_user_ids=record.matched_user_ids,
                    fields=record.fields,
                    source_context=source_context,
                )
                self.assertEqual(self.engine.classify_record(record), expected)

    def test_rejects_malformed_or_ambiguous_policy_rows(self) -> None:
        with self.assertRaises(PolicyConfigurationError):
            PolicyEngine([{"id": "wrong_schema"}], self.resolver)
        duplicate = [self.rules[0], {**self.rules[1], "data_category": self.rules[0]["data_category"]}]
        with self.assertRaises(PolicyConfigurationError):
            PolicyEngine(duplicate, self.resolver)

    def test_delete_rule_marks_present_noncompliant_and_absent_compliant(self) -> None:
        present = self._record(category="USER_PROFILE", matched={USER_ID}, fields={"email": "current@example.test"})
        absent = self._record(category="USER_PROFILE", state="ABSENT", matched={USER_ID})

        self.assertEqual(
            self.engine.evaluate_record(present, user_id=USER_ID, as_of=CUTOFF).status,
            DecisionStatus.NONCOMPLIANT,
        )
        self.assertEqual(
            self.engine.evaluate_record(absent, user_id=USER_ID, as_of=CUTOFF).status,
            DecisionStatus.COMPLIANT,
        )

    def test_tombstone_redacts_direct_pii_but_preserves_stable_user_reference(self) -> None:
        exposing = self._record(
            category="ACCOUNT_TOMBSTONE",
            matched={USER_ID},
            fields={
                "collection_name": "deleted_customer_references",
                "subject_reference": USER_ID,
                "email": "current@example.test",
                "name": "Test Customer",
                "phone": "+1 555 010 0100",
                "email_hash": "a" * 64,
                "account_status": "DELETED",
            },
        )
        safe = self._record(
            category="ACCOUNT_TOMBSTONE",
            matched={USER_ID},
            fields={
                "collection_name": "deleted_customer_references",
                "subject_reference": USER_ID,
                "account_status": "DELETED",
            },
        )
        exposed_decision = self.engine.evaluate_record(
            exposing, user_id=USER_ID, as_of=CUTOFF,
        )
        safe_decision = self.engine.evaluate_record(
            safe, user_id=USER_ID, as_of=CUTOFF,
        )

        self.assertEqual(exposed_decision.status, DecisionStatus.NONCOMPLIANT)
        self.assertEqual(
            {item.identifier_type for item in exposed_decision.exposed_identifiers},
            {"email", "name", "phone", "email_hash"},
        )
        self.assertEqual(safe_decision.status, DecisionStatus.COMPLIANT)
        self.assertEqual(safe_decision.exposed_identifiers, ())

    def test_shared_support_conversation_is_redacted_not_deleted(self) -> None:
        record = self._record(
            category="SUPPORT_MESSAGE",
            matched={USER_ID},
            fields={
                "message_text": "Customer current@example.test asked about order ORD-22.",
                "channel": "support_chat",
                "agent_name": "Support Agent",
                "order_reference": "ORD-22",
            },
        )
        decision = self.engine.evaluate_record(record, user_id=USER_ID, as_of=CUTOFF)

        self.assertEqual(decision.required_action, "REDACT")
        self.assertEqual(decision.status, DecisionStatus.NONCOMPLIANT)
        self.assertTrue(any(item.field_path == "message_text" for item in decision.exposed_identifiers))

    def test_attachment_owned_only_by_subject_uses_delete_rule(self) -> None:
        record = self._record(
            category="SUPPORT_ATTACHMENT",
            matched={USER_ID},
            fields={"owner_user_id": USER_ID, "file_name": "attachment.bin"},
        )
        decision = self.engine.evaluate_record(record, user_id=USER_ID, as_of=CUTOFF)
        self.assertEqual(decision.required_action, "DELETE")
        self.assertEqual(decision.status, DecisionStatus.NONCOMPLIANT)

    def test_attachment_condition_without_ownership_evidence_needs_context(self) -> None:
        record = self._record(category="SUPPORT_ATTACHMENT", fields={"file_name": "attachment.bin"})
        decision = self.engine.evaluate_record(record, user_id=USER_ID, as_of=CUTOFF)
        self.assertEqual(decision.status, DecisionStatus.NEEDS_CONTEXT)

    def test_anonymize_detects_link_and_accepts_explicitly_verified_unlinking(self) -> None:
        linked = self._record(
            category="ANALYTICS_PROFILE",
            matched={USER_ID},
            fields={"user_id": USER_ID, "email": "current@example.test"},
        )
        unlinked = self._record(
            category="ANALYTICS_PROFILE",
            matched=set(),
            fields={"user_id": "unlinked_subject", "metrics": {"orders": 2}},
        )
        linked_decision = self.engine.evaluate_record(linked, user_id=USER_ID, as_of=CUTOFF)
        unlinked_decision = self.engine.evaluate_record(
            unlinked, user_id=USER_ID, as_of=CUTOFF,
            context=PolicyContext(record_attributable=False),
        )

        self.assertEqual(linked_decision.required_action, "ANONYMIZE")
        self.assertEqual(linked_decision.status, DecisionStatus.NONCOMPLIANT)
        self.assertEqual(unlinked_decision.status, DecisionStatus.COMPLIANT)

    def test_direct_redacted_and_anonymized_readback_states_are_compliant(self) -> None:
        redacted = self._record(
            category="SUPPORT_MESSAGE", state="REDACTED", matched=set(),
            fields={"message_text": "Customer identifiers removed", "channel": "support_chat"},
        )
        anonymized = self._record(
            category="ANALYTICS_PROFILE", state="ANONYMIZED", matched=set(),
            fields={"metrics": {"orders": 2}},
        )
        redacted_decision = self.engine.evaluate_record(redacted, user_id=USER_ID, as_of=CUTOFF)
        anonymized_decision = self.engine.evaluate_record(
            anonymized, user_id=USER_ID, as_of=CUTOFF,
            context=PolicyContext(record_attributable=False),
        )
        self.assertEqual(redacted_decision.status, DecisionStatus.COMPLIANT)
        self.assertEqual(anonymized_decision.status, DecisionStatus.COMPLIANT)

    def test_retention_rule_requires_the_record_to_remain(self) -> None:
        retained = self._record(
            category="SECURITY_AUDIT_LOG",
            matched={USER_ID},
            fields={"retention_until": "2026-10-01", "subject_reference": USER_ID},
        )
        missing = self._record(
            category="SECURITY_AUDIT_LOG", state="ABSENT", matched={USER_ID},
            fields={"retention_until": "2026-10-01"},
        )
        self.assertEqual(
            self.engine.evaluate_record(retained, user_id=USER_ID, as_of=CUTOFF).status,
            DecisionStatus.COMPLIANT,
        )
        self.assertEqual(
            self.engine.evaluate_record(missing, user_id=USER_ID, as_of=CUTOFF).status,
            DecisionStatus.NONCOMPLIANT,
        )

    def test_email_delivery_log_redacts_historical_email_in_text(self) -> None:
        record = self._record(
            category="EMAIL_DELIVERY_LOG",
            matched={USER_ID},
            fields={"message": "Delivery failed for historical@example.test"},
        )
        decision = self.engine.evaluate_record(record, user_id=USER_ID, as_of=CUTOFF)
        self.assertEqual(decision.required_action, "REDACT")
        self.assertEqual(decision.status, DecisionStatus.NONCOMPLIANT)

    def test_legal_hold_record_is_retained_when_hold_is_active(self) -> None:
        record = self._record(category="LEGAL_HOLD_RECORD", matched={USER_ID}, fields={"record_id": "hold_1"})
        decision = self.engine.evaluate_record(
            record, user_id=USER_ID, as_of=CUTOFF,
            context=PolicyContext(legal_hold_active=True),
        )
        self.assertEqual(decision.required_action, "RETAIN")
        self.assertEqual(decision.status, DecisionStatus.COMPLIANT)

    def test_unknown_legal_hold_and_restore_conditions_are_not_guessed(self) -> None:
        legal = self._record(category="LEGAL_HOLD_RECORD", matched={USER_ID})
        backup = self._record(category="BACKUP_ARCHIVE", matched={USER_ID}, fields={"restore_allowed": True})
        self.assertEqual(
            self.engine.evaluate_record(legal, user_id=USER_ID, as_of=CUTOFF).status,
            DecisionStatus.NEEDS_CONTEXT,
        )
        self.assertEqual(
            self.engine.evaluate_record(backup, user_id=USER_ID, as_of=CUTOFF).status,
            DecisionStatus.NEEDS_CONTEXT,
        )

    def test_retained_backup_and_restore_risk_are_separate_assessments(self) -> None:
        record = self._record(
            category="BACKUP_ARCHIVE",
            state="PRESENT",
            matched={USER_ID},
            fields={"restore_allowed": True, "restore_access_blockable": True},
            source_file="evidence/databases/backup_catalog.sql",
        )
        policy = self.engine.evaluate_record(record, user_id=USER_ID, as_of=CUTOFF)
        risks = self.engine.evaluate_recreation_risks(record, as_of=CUTOFF)

        self.assertEqual(policy.required_action, "RETAIN")
        self.assertEqual(policy.status, DecisionStatus.COMPLIANT)
        self.assertEqual(len(risks), 1)
        self.assertEqual(risks[0].required_action, "BLOCK_RESTORE")
        self.assertEqual(risks[0].status, DecisionStatus.NONCOMPLIANT)

    def test_active_data_flow_is_a_separate_condition(self) -> None:
        record = self._record(
            category="DATA_FLOW",
            state="PRESENT",
            matched={USER_ID},
            fields={"connector_name": "customer_sync", "enabled": True},
            source_file="evidence/records/connectors.json",
        )
        risk, = self.engine.evaluate_recreation_risks(record, as_of=CUTOFF)
        self.assertEqual(risk.required_action, "DISABLE_DATA_FLOW")
        self.assertEqual(risk.status, DecisionStatus.NONCOMPLIANT)

    def test_audit_cutoff_ignores_later_snapshots_and_claimed_results(self) -> None:
        before = self._record(
            category="USER_PROFILE", state="PRESENT", matched={USER_ID},
            evidence_id="before", observed_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
        )
        after = self._record(
            category="USER_PROFILE", state="ABSENT", matched={USER_ID},
            evidence_id="after", observed_at=datetime(2026, 10, 1, tzinfo=timezone.utc),
        )
        claim = self._record(
            category="USER_PROFILE", state="ABSENT", matched={USER_ID},
            evidence_id="claim", kind="claim",
            observed_at=datetime(2026, 9, 20, tzinfo=timezone.utc),
        )
        decision = self.engine.evaluate_resource(
            [before, after, claim], user_id=USER_ID, system_id="account_service",
            resource_id="resource_1", as_of=CUTOFF,
        )
        self.assertEqual(decision.status, DecisionStatus.NONCOMPLIANT)
        self.assertEqual(decision.evidence_id, "before")

    def test_conflicting_same_time_direct_evidence_is_unknown(self) -> None:
        present = self._record(
            category="USER_PROFILE", state="PRESENT", matched={USER_ID}, evidence_id="present",
        )
        absent = self._record(
            category="USER_PROFILE", state="ABSENT", matched={USER_ID}, evidence_id="absent",
        )
        decision = self.engine.evaluate_resource(
            [present, absent], user_id=USER_ID, system_id="account_service",
            resource_id="resource_1", as_of=CUTOFF,
        )
        self.assertEqual(decision.status, DecisionStatus.UNKNOWN)
        self.assertEqual(set(decision.conflicting_evidence_ids), {"present", "absent"})

    def test_post_cutoff_single_record_is_out_of_scope(self) -> None:
        future = self._record(
            category="USER_PROFILE", matched={USER_ID},
            observed_at=datetime(2026, 10, 1, tzinfo=timezone.utc),
        )
        decision = self.engine.evaluate_record(future, user_id=USER_ID, as_of=CUTOFF)
        self.assertEqual(decision.status, DecisionStatus.OUT_OF_SCOPE)


if __name__ == "__main__":
    unittest.main()
