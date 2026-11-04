"""Tests for extracting and time-bounding participant evidence."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from evidence_scanner import EvidenceScanner  # noqa: E402
from identity_resolver import IdentityResolver  # noqa: E402


class EvidenceScannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.dataset_dir = Path(self.temporary_directory.name)
        (self.dataset_dir / "evidence").mkdir()
        identities = [{
            "user_id": "user_test",
            "name": "Test Customer",
            "email": "current@example.test",
            "phone": "+1 555 010 0100",
            "email_hash": "a" * 64,
            "updated_at": "2026-01-01T00:00:00Z",
        }]
        audit = [{
            "user_id": "user_test",
            "event_time": "2025-12-01T00:00:00Z",
            "field_name": "email",
            "old_value": "historical@example.test",
            "new_value": "current@example.test",
        }]
        self.resolver = IdentityResolver(identities, audit)

    def _manifest_item(self, file_id: str, relative_path: str, captured_at: str) -> dict[str, str]:
        return {
            "file_id": file_id,
            "file_path": f"evidence/{relative_path}",
            "origin": "test_source",
            "captured_at": captured_at,
        }

    def _write_json(self, name: str, data: object) -> None:
        (self.dataset_dir / "evidence" / name).write_text(
            json.dumps(data), encoding="utf-8"
        )

    def test_extracts_json_record_and_matches_historical_email(self) -> None:
        self._write_json("records.json", {
            "captured_at": "2026-01-10T00:00:00Z",
            "records": [{
                "resource_id": "record_1",
                "email": "historical@example.test",
                "status": "ACTIVE",
            }],
        })
        scanner = EvidenceScanner(
            self.dataset_dir,
            [self._manifest_item("evidence_1", "records.json", "2026-01-10T00:00:00Z")],
            self.resolver,
        )

        result = scanner.scan_all()

        self.assertEqual(result.parsed_files, 1)
        self.assertEqual(len(result.records), 1)
        self.assertEqual(result.records[0].resource_id, "record_1")
        self.assertEqual(result.records[0].evidence_id, "evidence_1")
        self.assertEqual(result.records[0].matched_user_ids, {"user_test"})
        self.assertEqual(result.records[0].observed_state, "PRESENT")

    def test_extracts_native_pdf_text_and_matches_historical_email(self) -> None:
        try:
            import pymupdf
        except ImportError:
            self.skipTest("PyMuPDF is an optional dependency")
        pdf_path = self.dataset_dir / "evidence" / "report.pdf"
        document = pymupdf.open()
        page = document.new_page()
        page.insert_text((72, 72), "Customer email: historical@example.test")
        document.save(pdf_path)
        document.close()
        scanner = EvidenceScanner(
            self.dataset_dir,
            [self._manifest_item("pdf_evidence", "report.pdf", "2026-01-10T00:00:00Z")],
            self.resolver,
        )

        result = scanner.scan_all()

        self.assertEqual(result.parsed_files, 1)
        self.assertEqual(result.partial_files, 0)
        self.assertEqual(result.records[0].file_format, ".pdf")
        self.assertEqual(result.records[0].matched_user_ids, {"user_test"})

    def test_image_only_pdf_is_reported_as_ocr_required(self) -> None:
        try:
            import pymupdf
        except ImportError:
            self.skipTest("PyMuPDF is an optional dependency")
        pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 20), False)
        pixmap.clear_with(255)
        image_bytes = pixmap.tobytes("png")
        pdf_path = self.dataset_dir / "evidence" / "scan.pdf"
        document = pymupdf.open()
        page = document.new_page()
        page.insert_image(page.rect, stream=image_bytes)
        document.save(pdf_path)
        document.close()
        scanner = EvidenceScanner(
            self.dataset_dir,
            [self._manifest_item("scan_evidence", "scan.pdf", "2026-01-10T00:00:00Z")],
            self.resolver,
        )

        result = scanner.scan_all()

        self.assertEqual(result.discovered_files, 1)
        self.assertEqual(result.failed_files, 0)
        self.assertEqual(result.partial_files, 1)
        self.assertIn("OCR", result.issues[0].message)

    def test_png_is_sent_through_ocr_or_reported_partial(self) -> None:
        try:
            import pymupdf
        except ImportError:
            self.skipTest("PyMuPDF is an optional dependency")
        pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 20), False)
        pixmap.clear_with(255)
        (self.dataset_dir / "evidence" / "screen.png").write_bytes(pixmap.tobytes("png"))
        scanner = EvidenceScanner(
            self.dataset_dir,
            [self._manifest_item("image_evidence", "screen.png", "2026-01-10T00:00:00Z")],
            self.resolver,
        )

        result = scanner.scan_all()

        self.assertEqual(result.discovered_files, 1)
        self.assertEqual(result.failed_files, 0)
        self.assertEqual(result.partial_files + result.parsed_files, 1)
        if result.partial_files:
            self.assertIn("OCR", result.issues[0].message)

    def test_tesseract_can_be_found_from_configured_windows_path(self) -> None:
        fake_tesseract = self.dataset_dir / "tools" / "tesseract.exe"
        fake_tesseract.parent.mkdir()
        fake_tesseract.write_bytes(b"test placeholder; discovery only")
        with patch.dict(os.environ, {"TESSERACT_CMD": str(fake_tesseract), "PATH": ""}):
            scanner = EvidenceScanner(self.dataset_dir, [], self.resolver)

        self.assertEqual(scanner.ocr_engine_path, fake_tesseract.resolve())

    def test_pymupdf_uses_discovered_tesseract_engine(self) -> None:
        fake_tesseract = self.dataset_dir / "tools" / "tesseract.exe"
        fake_tesseract.parent.mkdir()
        fake_tesseract.write_bytes(b"test placeholder; discovery only")
        with patch.dict(os.environ, {"TESSERACT_CMD": str(fake_tesseract), "PATH": ""}):
            scanner = EvidenceScanner(self.dataset_dir, [], self.resolver)
            page = Mock()
            page.get_textpage_ocr.return_value = object()
            page.get_text.return_value = "OCR text"

            text = scanner._ocr_page(page)

        self.assertEqual(text, "OCR text")
        page.get_textpage_ocr.assert_called_once_with(language="eng", dpi=200, full=True)

    def test_ocr_text_keeps_original_manifest_evidence_id(self) -> None:
        try:
            import pymupdf
        except ImportError:
            self.skipTest("PyMuPDF is an optional dependency")
        pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 20), False)
        pixmap.clear_with(255)
        (self.dataset_dir / "evidence" / "ocr_screen.png").write_bytes(pixmap.tobytes("png"))
        scanner = EvidenceScanner(
            self.dataset_dir,
            [self._manifest_item("evidence_ocr_11", "ocr_screen.png", "2026-01-10T00:00:00Z")],
            self.resolver,
        )
        with patch.object(EvidenceScanner, "_ocr_page", return_value="Customer email: historical@example.test"):
            result = scanner.scan_all()

        self.assertEqual(result.parsed_files, 1)
        self.assertEqual(len(result.records), 1)
        self.assertEqual(result.records[0].evidence_id, "evidence_ocr_11")
        self.assertEqual(result.records[0].matched_user_ids, {"user_test"})
        self.assertEqual(result.records[0].file_format, ".png")
        self.assertEqual(
            result.records[0].fields["document_text"],
            "Customer email: historical@example.test",
        )

    def test_audit_cutoff_excludes_later_snapshot_state(self) -> None:
        self._write_json("before.json", {"records": [{
            "resource_id": "profile_1", "email": "current@example.test", "state": "PRESENT",
        }]})
        self._write_json("after.json", {"records": [{
            "resource_id": "profile_1", "email": "current@example.test", "state": "ABSENT",
        }]})
        scanner = EvidenceScanner(
            self.dataset_dir,
            [
                self._manifest_item("before", "before.json", "2026-01-10T00:00:00Z"),
                self._manifest_item("after", "after.json", "2026-02-10T00:00:00Z"),
            ],
            self.resolver,
        )
        scanner.scan_all()

        january_state = scanner.reconstruct_resource_state(
            "test_source", "profile_1", as_of=datetime.fromisoformat("2026-01-20T00:00:00+00:00")
        )
        february_state = scanner.reconstruct_resource_state(
            "test_source", "profile_1", as_of=datetime.fromisoformat("2026-02-20T00:00:00+00:00")
        )
        self.assertEqual(january_state, "PRESENT")
        self.assertEqual(february_state, "ABSENT")

    def test_sql_replica_is_parsed_and_primary_is_not_mislabeled(self) -> None:
        sql = '''
        CREATE TABLE customer_data_primary (
            resource_id TEXT, origin_system TEXT, email TEXT, updated_at TEXT
        );
        CREATE TABLE customer_data_read_replica (
            resource_id TEXT, origin_system TEXT, email TEXT, updated_at TEXT
        );
        INSERT INTO customer_data_primary VALUES (
            'record_1', 'support_platform', 'current@example.test', '2026-01-01T00:00:00Z'
        );
        INSERT INTO customer_data_read_replica VALUES (
            'record_1', 'support_platform', 'current@example.test', '2026-01-01T00:00:00Z'
        );
        '''
        (self.dataset_dir / "evidence" / "snapshot.sql").write_text(sql, encoding="utf-8")
        scanner = EvidenceScanner(
            self.dataset_dir,
            [self._manifest_item("sql_evidence", "snapshot.sql", "2026-01-02T00:00:00Z")],
            self.resolver,
        )

        result = scanner.scan_all()

        self.assertEqual(result.parsed_files, 1)
        self.assertEqual(len(result.records), 2)
        self.assertEqual({record.topology for record in result.records}, {"primary", "read_replica"})
        self.assertTrue(all(record.system_id == "support_platform" for record in result.records))

    def test_enabled_connector_is_detected_as_active_data_flow(self) -> None:
        self._write_json("connectors.json", {"records": [{
            "resource_id": "connector_1",
            "user_id": "user_test",
            "connector_name": "analytics_export",
            "enabled": True,
        }]})
        scanner = EvidenceScanner(
            self.dataset_dir,
            [self._manifest_item("connector_evidence", "connectors.json", "2026-01-03T00:00:00Z")],
            self.resolver,
        )

        result = scanner.scan_all()

        self.assertEqual(result.records[0].category, "data_flow")
        self.assertEqual(result.records[0].observed_state, "ACTIVE")

    def test_allowed_backup_restore_is_detected_as_active_restore_risk(self) -> None:
        self._write_json("backups.json", {"records": [{
            "resource_id": "backup_1",
            "user_id": "user_test",
            "archive_key": "archive_1",
            "restore_allowed": True,
        }]})
        scanner = EvidenceScanner(
            self.dataset_dir,
            [self._manifest_item("backup_evidence", "backups.json", "2026-01-03T00:00:00Z")],
            self.resolver,
        )

        result = scanner.scan_all()

        self.assertEqual(result.records[0].category, "restore_path")
        self.assertEqual(result.records[0].observed_state, "ACTIVE")

    def test_activity_claim_cannot_override_direct_snapshot_observation(self) -> None:
        self._write_json("snapshot.json", {"records": [{
            "resource_id": "profile_1", "email": "current@example.test", "state": "PRESENT",
        }]})
        (self.dataset_dir / "evidence" / "activity.log").write_text(
            "2026-01-11T00:00:00Z account_service profile_1 DELETE_RECORD SUCCESS current@example.test\n",
            encoding="utf-8",
        )
        scanner = EvidenceScanner(
            self.dataset_dir,
            [
                self._manifest_item("snapshot", "snapshot.json", "2026-01-10T00:00:00Z"),
                self._manifest_item("activity", "activity.log", "2026-01-11T00:00:00Z"),
            ],
            self.resolver,
        )
        scanner.scan_all()

        state = scanner.reconstruct_resource_state(
            "test_source", "profile_1", as_of=datetime.fromisoformat("2026-01-12T00:00:00+00:00")
        )
        self.assertEqual(state, "PRESENT")
        self.assertTrue(any(record.evidence_kind == "claim" for record in scanner.find_for_user("user_test")))

    def test_activity_sync_event_is_a_risk_hint_not_proof_of_active_flow(self) -> None:
        scanner = EvidenceScanner(
            self.dataset_dir,
            [],
            self.resolver,
            activity_rows=[{
                "activity_id": "activity_test",
                "event_time": "2026-01-11T00:00:00Z",
                "event_type": "SYNC_RECORD",
                "source_system": "account_service",
                "target_system": "analytics_platform",
                "resource_id": "profile_1",
                "reported_result": "SUCCESS",
                "details": "Sync delivered for historical@example.test",
            }],
        )

        result = scanner.scan_all()

        self.assertEqual(result.records[0].category, "data_flow")
        self.assertEqual(result.records[0].evidence_id, "activity_test")
        self.assertEqual(result.records[0].matched_user_ids, {"user_test"})
        self.assertEqual(result.records[0].observed_state, "UNKNOWN")
        self.assertEqual(result.records[0].evidence_kind, "claim")

    def test_structured_handover_readback_is_a_direct_observation(self) -> None:
        (self.dataset_dir / "evidence" / "handover.txt").write_text(
            "CUSTOMER PLATFORM - SHIFT HANDOVER\n"
            "- 2026-01-11T00:00:00Z | account_service | account_1 | USER_PROFILE | NOT_FOUND | historical@example.test\n",
            encoding="utf-8",
        )
        scanner = EvidenceScanner(
            self.dataset_dir,
            [self._manifest_item("handover_id", "handover.txt", "2026-01-11T00:00:00Z")],
            self.resolver,
        )

        result = scanner.scan_all()

        observed = next(record for record in result.records if record.resource_id == "account_1")
        self.assertEqual(observed.evidence_id, "handover_id")
        self.assertEqual(observed.system_id, "account_service")
        self.assertEqual(observed.observed_state, "ABSENT")
        self.assertEqual(observed.evidence_kind, "observation")
        self.assertEqual(observed.matched_user_ids, {"user_test"})

    def test_unsupported_format_is_reported_instead_of_skipped(self) -> None:
        (self.dataset_dir / "evidence" / "unknown.bin").write_bytes(b"not parsed")
        scanner = EvidenceScanner(
            self.dataset_dir,
            [self._manifest_item("unknown", "unknown.bin", "2026-01-03T00:00:00Z")],
            self.resolver,
        )

        result = scanner.scan_all()

        self.assertEqual(result.discovered_files, 1)
        self.assertEqual(result.parsed_files, 0)
        self.assertEqual(result.issues[0].status, "unsupported")


if __name__ == "__main__":
    unittest.main()