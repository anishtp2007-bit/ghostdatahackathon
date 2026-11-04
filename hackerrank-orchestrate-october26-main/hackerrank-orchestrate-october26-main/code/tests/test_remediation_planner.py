"""Tests for the findings and remediation planner."""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from data_loader import load_dataset  # noqa: E402
from evidence_scanner import EvidenceRecord, ScanResult  # noqa: E402
from identity_resolver import IdentityResolver  # noqa: E402
from policy_engine import PolicyEngine  # noqa: E402
from remediation_planner import (  # noqa: E402
    OPERATION_ORDER,
    PHASE_ORDER,
    PlanStep,
    RemediationPlanner,
    RequestIntent,
    RequestIntentClassifier,
)


CUTOFF = datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc)


class RequestIntentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.classifier = RequestIntentClassifier()

    def test_explicit_deletion_in_english_spanish_french_and_japanese(self) -> None:
        examples = (
            "Please permanently delete my account and personal information.",
            "Por favor, eliminar mi cuenta y mis datos personales.",
            "Je demande la suppression de mon compte et de mes données.",
            "このアカウントと個人データの削除をお願いします。",
        )
        for text in examples:
            with self.subTest(language_text=text.encode("ascii", "backslashreplace")):
                self.assertEqual(self.classifier.classify(text), RequestIntent.DELETION)

    def test_pure_access_correction_and_marketing_opt_out_are_not_applicable(self) -> None:
        requests = (
            "Please send me a copy of my personal data.",
            "Please correct the phone number on my profile.",
            "Unsubscribe me from marketing emails.",
        )
        for text in requests:
            with self.subTest(text=text):
                self.assertEqual(self.classifier.classify(text), RequestIntent.NON_DELETION)

    def test_deletion_wins_when_request_mentions_other_actions_too(self) -> None:
        self.assertEqual(
            self.classifier.classify("Delete my account and send me a copy of the records first."),
            RequestIntent.DELETION,
        )

    def test_explicit_request_not_to_delete_is_non_deletion(self) -> None:
        self.assertEqual(
            self.classifier.classify("Do not delete my account; only correct my address."),
            RequestIntent.NON_DELETION,
        )

    def test_unrecognized_request_is_left_unresolved(self) -> None:
        self.assertEqual(self.classifier.classify("Please review my account request."), RequestIntent.UNRESOLVED)


class RemediationPlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset()
        cls.identity = cls.dataset.user_identities[0]
        cls.user_id = cls.identity["user_id"]
        cls.email = cls.identity["email"]
        cls.resolver = IdentityResolver([cls.identity], [])
        cls.policy = PolicyEngine.from_dataset(cls.dataset.deletion_rules, cls.resolver)

    def _record(
        self,
        *,
        resource_id: str,
        record_type: str,
        system_id: str,
        state: str = "PRESENT",
        fields: dict[str, object] | None = None,
        evidence_id: str = "evidence_01",
        observed_at: datetime = datetime(2026, 9, 20, tzinfo=timezone.utc),
        kind: str = "observation",
        source_context: str = "",
        source_file: str = "evidence/test.json",
    ) -> EvidenceRecord:
        data_fields: dict[str, object] = {
            "record_type": record_type,
            "resource_id": resource_id,
            **(fields or {}),
        }
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
            matched_user_ids=set(),
            fields=data_fields,
            source_context=source_context,
        )

    def _planner(self, records: list[EvidenceRecord], *, issues: list | None = None) -> RemediationPlanner:
        result = ScanResult(
            records=records,
            issues=list(issues or []),
            discovered_files=len(self.dataset.evidence_files),
            parsed_files=len(self.dataset.evidence_files) - len(issues or []),
            partial_files=len(issues or []),
        )
        return RemediationPlanner(
            self.dataset,
            identity_resolver=self.resolver,
            policy_engine=self.policy,
            scan_result=result,
        )

    def _request(self, case_id: str, text: str = "Please delete my account and personal data.") -> dict[str, str]:
        return {
            "case_id": case_id,
            "email": self.email,
            "requested_at": "2026-09-01T00:00:00Z",
            "reported_status": "COMPLETE",
            "reported_at": "2026-09-02T00:00:00Z",
            "audit_at": CUTOFF.isoformat(),
            "request_text": text,
        }

    def test_not_applicable_has_an_empty_step_list(self) -> None:
        planner = self._planner([])
        plan = planner.plan_case(self._request("case_access", "Please provide a copy of my personal data."))
        self.assertEqual(plan.deletion_status, "NOT_APPLICABLE")
        self.assertEqual(plan.output_steps(), [])
        self.assertEqual(plan.as_output_row()["steps_json"], "[]")

    def test_unresolved_request_is_not_mislabeled_not_applicable(self) -> None:
        plan = self._planner([]).plan_case(self._request("case_unknown", "Please review my account."))
        self.assertIsNone(plan.deletion_status)
        self.assertTrue(plan.unresolved_reasons)
        with self.assertRaises(ValueError):
            plan.as_output_row()

    def test_complete_plan_has_only_evidence_backed_verification(self) -> None:
        record = self._record(
            resource_id="account_700",
            record_type="USER_PROFILE",
            system_id="account_service",
            state="ABSENT",
            fields={"user_id": self.user_id},
        )
        plan = self._planner([record]).plan_case(self._request("case_complete"))

        self.assertEqual(plan.deletion_status, "COMPLETE")
        self.assertTrue(plan.steps)
        self.assertTrue(all(step.phase == "VERIFICATION" for step in plan.steps))
        self.assertTrue(all(step.evidence_id == "evidence_01" for step in plan.steps))
        self.assertEqual(plan.steps[0].operation, "VERIFY_RECORD_ABSENT")

    def test_activity_claim_links_resource_but_never_proves_its_state(self) -> None:
        direct = self._record(
            resource_id="account_705", record_type="USER_PROFILE", system_id="account_service",
            state="ABSENT", evidence_id="evidence_01", fields={},
        )
        claim = self._record(
            resource_id="account_705", record_type="USER_PROFILE", system_id="account_service",
            state="UNKNOWN", evidence_id="activity_1", kind="claim",
            fields={"resource_id": "account_705", "message": self.email},
        )
        claim = __import__("dataclasses").replace(claim, line_number=12)
        plan = self._planner([direct, claim]).plan_case(self._request("case_link"))

        self.assertEqual(plan.deletion_status, "COMPLETE")
        self.assertEqual(len(plan.steps), 1)
        self.assertEqual(plan.steps[0].evidence_id, "evidence_01")
        self.assertNotEqual(plan.steps[0].evidence_id, "activity_1")

    def test_activity_resource_link_does_not_cross_unrelated_systems(self) -> None:
        direct = self._record(
            resource_id="shared_705", record_type="USER_PROFILE", system_id="search_service",
            state="PRESENT", fields={"resource_id": "shared_705", "search_term": "unrelated"},
            evidence_id="evidence_01",
        )
        claim = self._record(
            resource_id="shared_705", record_type="USER_PROFILE", system_id="account_service",
            kind="claim", evidence_id="activity_1",
            fields={"resource_id": "shared_705", "message": self.email, "source_system": "account_service"},
        )
        claim = __import__("dataclasses").replace(claim, line_number=2)
        plan = self._planner([direct, claim]).plan_case(self._request("case_system_collision"))

        self.assertIsNone(plan.deletion_status)
        self.assertFalse(plan.steps)
        self.assertTrue(any("Activity evidence links" in reason for reason in plan.unresolved_reasons))

    def test_incomplete_plan_covers_all_allowed_step_operations_and_exact_fields(self) -> None:
        records = [
            self._record(
                resource_id="account_701", record_type="USER_PROFILE", system_id="account_service",
                fields={"user_id": self.user_id, "email": self.email},
            ),
            self._record(
                resource_id="support_701", record_type="SUPPORT_MESSAGE", system_id="support_platform",
                fields={"subject_reference": self.user_id, "message_text": f"Email {self.email}", "channel": "email", "order_reference": "order_1"},
            ),
            self._record(
                resource_id="analytics_701", record_type="ANALYTICS_PROFILE", system_id="analytics_platform",
                fields={"user_id": self.user_id, "email": self.email},
            ),
            self._record(
                resource_id="audit_701", record_type="SECURITY_AUDIT_LOG", system_id="security_service",
                fields={"subject_reference": self.user_id, "retention_until": "2026-12-01"},
            ),
            self._record(
                resource_id="connector_701", record_type="DATA_FLOW", system_id="integration_service",
                state="ACTIVE", fields={"subject_selector": self.email, "enabled": True},
                source_context="connector_configurations",
            ),
            self._record(
                resource_id="backup_701", record_type="BACKUP_ARCHIVE", system_id="backup_service",
                state="PRESENT", fields={"subject_reference": self.user_id, "restore_allowed": True, "restore_access_blockable": True},
                source_context="backup_catalog", source_file="evidence/databases/backup_catalog.sql",
            ),
        ]
        plan = self._planner(records).plan_case(self._request("case_incomplete"))

        self.assertEqual(plan.deletion_status, "INCOMPLETE")
        self.assertFalse(plan.unresolved_reasons)
        step_dicts = plan.output_steps()
        self.assertTrue(all(set(step) == {
            "step_no", "phase", "system_id", "resource_id", "operation", "state", "evidence_id",
        } for step in step_dicts))
        self.assertEqual([step["step_no"] for step in step_dicts], list(range(1, len(step_dicts) + 1)))
        self.assertEqual(
            {step.operation for step in plan.steps},
            {
                "REPORT_REMAINING_RECORD", "REPORT_EXPOSED_PII", "REPORT_ACTIVE_DATA_FLOW", "REPORT_RESTORE_RISK",
                "DISABLE_DATA_FLOW", "BLOCK_RESTORE", "DELETE_RECORD", "REDACT_RECORD", "ANONYMIZE_RECORD",
                "VERIFY_RECORD_ABSENT", "VERIFY_RECORD_PRESENT", "VERIFY_RECORD_REDACTED",
                "VERIFY_RECORD_ANONYMIZED", "VERIFY_DATA_FLOW_DISABLED", "VERIFY_RESTORE_BLOCKED",
            },
        )
        self.assertTrue(all(step.evidence_id == "evidence_01" for step in plan.steps if step.phase == "FINDING"))
        self.assertTrue(all(step.evidence_id == "" for step in plan.steps if step.phase in {"REMEDIATION", "VERIFICATION"}))
        ordering = [(PHASE_ORDER[step.phase], OPERATION_ORDER[step.operation], step.system_id, step.resource_id) for step in plan.steps]
        self.assertEqual(ordering, sorted(ordering))
        self.assertTrue(all(step.evidence_id in {row["file_id"] for row in self.dataset.evidence_files} for step in plan.steps if step.evidence_id))

    def test_unknown_policy_evidence_does_not_create_a_finding(self) -> None:
        record = self._record(
            resource_id="mystery_701", record_type="UNSUPPORTED_CATEGORY", system_id="unknown_system",
            fields={"user_id": self.user_id},
        )
        plan = self._planner([record]).plan_case(self._request("case_ambiguous"))
        self.assertIsNone(plan.deletion_status)
        self.assertEqual(plan.output_steps(), [])
        self.assertTrue(any("policy category" in reason for reason in plan.unresolved_reasons))

    def test_conflicting_latest_resource_states_are_unresolved(self) -> None:
        records = [
            self._record(resource_id="account_702", record_type="USER_PROFILE", system_id="account_service", state="PRESENT", fields={"email": self.email}, evidence_id="evidence_01"),
            self._record(resource_id="account_702", record_type="USER_PROFILE", system_id="account_service", state="ABSENT", fields={"email": self.email}, evidence_id="evidence_02"),
        ]
        plan = self._planner(records).plan_case(self._request("case_conflict"))
        self.assertIsNone(plan.deletion_status)
        self.assertTrue(any("Conflicting" in reason for reason in plan.unresolved_reasons))

    def test_future_observations_do_not_affect_audit_time_plan(self) -> None:
        record = self._record(
            resource_id="account_703", record_type="USER_PROFILE", system_id="account_service",
            state="ABSENT", fields={"user_id": self.user_id},
            observed_at=datetime(2026, 10, 3, tzinfo=timezone.utc),
        )
        plan = self._planner([record]).plan_case(self._request("case_future"))
        self.assertIsNone(plan.deletion_status)
        self.assertTrue(any("No identity-matched direct observations" in reason for reason in plan.unresolved_reasons))

    def test_partial_evidence_blocks_a_false_complete_result(self) -> None:
        record = self._record(
            resource_id="account_704", record_type="USER_PROFILE", system_id="account_service",
            state="ABSENT", fields={"user_id": self.user_id},
        )
        issue = type("Issue", (), {"evidence_id": "evidence_image", "status": "partial"})()
        plan = self._planner([record], issues=[issue]).plan_case(self._request("case_partial"))
        self.assertIsNone(plan.deletion_status)
        self.assertTrue(any("could not be fully parsed" in reason for reason in plan.unresolved_reasons))

    def test_confirmed_violation_is_incomplete_but_unresolved_plan_cannot_serialize(self) -> None:
        record = self._record(
            resource_id="account_706", record_type="USER_PROFILE", system_id="account_service",
            state="PRESENT", fields={"email": self.email},
        )
        issue = type("Issue", (), {"evidence_id": "evidence_image", "status": "partial"})()
        plan = self._planner([record], issues=[issue]).plan_case(self._request("case_incomplete_partial"))
        self.assertEqual(plan.deletion_status, "INCOMPLETE")
        self.assertFalse(plan.finalizable)
        self.assertTrue(plan.unresolved_reasons)
        with self.assertRaises(ValueError):
            plan.as_output_row()

    def test_step_constructor_rejects_non_contract_operations(self) -> None:
        with self.assertRaises(ValueError):
            RemediationPlanner._step("FINDING", "account_service", "account_1", "DELETE", "PRESENT", "evidence_01")


if __name__ == "__main__":
    unittest.main()
