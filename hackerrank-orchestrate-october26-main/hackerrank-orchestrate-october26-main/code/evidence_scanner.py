"""Extract and time-bound evidence records without making deletion decisions.

CSV, JSON, text/log, and SQLite-compatible SQL are handled with Python's
standard library. PDF text and image OCR are optional adapters; if their free
open-source dependencies are not installed, the scanner reports that clearly.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sqlite3
import struct
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from data_loader import DatasetValidationError, parse_iso_timestamp
from identity_resolver import IdentityResolver


_TIMESTAMP_KEYS = {
    "captured_at", "event_time", "timestamp", "observed_at", "updated_at",
    "created_at", "exported_at", "indexed_at", "deleted_at",
}
_IDENTIFIER_FIELDS = {
    "email": "email", "email_address": "email", "customer_email": "email",
    "requester_email": "email", "owner_email": "email", "contact_email": "email",
    "phone": "phone", "phone_number": "phone", "customer_phone": "phone",
    "requester_phone": "phone", "owner_phone": "phone",
    "email_hash": "email_hash", "hash": "email_hash",
    "name": "name", "full_name": "name", "display_name": "name",
    "requester_name": "name", "customer_name": "name",
    "user_id": "user_id", "subject_reference": "user_id",
    "customer_reference": "user_id", "stable_user_id": "user_id",
}
_RESOURCE_KEYS = ("resource_id", "record_id", "ticket_id", "request_id", "archive_key", "object_key")
_STATE_KEYS = {"state", "observed_state", "resource_state"}
_CLAIM_KEYS = {"reported_result", "reported_status", "workflow_status", "claimed_result"}
_TABLE_SYSTEMS = {
    "customer_profiles": "account_service",
    "customer_search_documents": "search_service",
    "search_documents_primary": "search_service",
    "customer_analytics_profiles": "analytics_platform",
    "support_records": "support_platform",
    "customer_exports": "export_service",
    "connector_subscriptions": "integration_management_service",
    "backup_restore_entries": "backup_service",
    "security_audit_records": "security_service",
}
_REPLICA_MARKERS = (
    ("regional_replicas", "regional_replica"),
    ("regional_replica", "regional_replica"),
    ("read_replica", "read_replica"),
    ("reporting_replica", "reporting_replica"),
    ("failover_replica", "failover_replica"),
    ("replica_ap_south", "regional_replica"),
    ("replica_eu_west", "regional_replica"),
)
_RESOURCE_PATTERN = re.compile(r"\b[a-zA-Z][a-zA-Z0-9-]*_\d+\b")
_ISO_PREFIX_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2}T[^\s|]+)")
_STRUCTURED_OBSERVATION_PATTERN = re.compile(
    r"^\s*[-*]\s*(\d{4}-\d{2}-\d{2}T[^\s|]+)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*(.*?)\s*$"
)
_KEY_VALUE_PATTERN = re.compile(r"\b([A-Za-z][A-Za-z0-9_.-]*)\s*[=:]\s*([^|,;]+)")
_ACTIVE_STATES = {"ACTIVE", "ENABLED", "RUNNING", "ALLOW", "ALLOWED", "TRUE", "1"}
_INACTIVE_STATES = {"INACTIVE", "DISABLED", "STOPPED", "BLOCKED", "FALSE", "0"}
_KNOWN_STATES = {
    "PRESENT", "ABSENT", "ACTIVE", "INACTIVE", "DISABLED", "BLOCKED",
    "REDACTED", "ANONYMIZED", "DELETED", "REMOVED", "RUNNING", "ENABLED",
}
_FLOW_ACTIVITY_TYPES = {
    "SYNC_RECORD", "CREATE_EXPORT", "IMPORT_RECORD", "EXPORT_RECORD",
    "CHECK_DATA_FLOW", "DISABLE_DATA_FLOW", "REFRESH_CACHE", "INDEX_RECORD",
}
_RESTORE_ACTIVITY_TYPES = {"RESTORE_BACKUP", "CHECK_RESTORE_ACCESS", "BLOCK_RESTORE"}


def find_tesseract_executable() -> Path | None:
    """Find a local Tesseract executable on Windows or another supported OS.

    ``TESSERACT_CMD`` may point directly to the executable if it is installed
    outside PATH. No network service or paid API is used.
    """
    configured = os.environ.get("TESSERACT_CMD", "").strip().strip('"')
    if configured:
        configured_path = Path(configured).expanduser()
        if configured_path.is_dir():
            configured_path = configured_path / ("tesseract.exe" if os.name == "nt" else "tesseract")
        if configured_path.is_file():
            return configured_path.resolve()

    for executable_name in ("tesseract.exe", "tesseract"):
        located = shutil.which(executable_name)
        if located:
            return Path(located).resolve()

    if os.name == "nt":
        program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        program_files_x86 = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
        local_app_data = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
        candidates = (
            program_files / "Tesseract-OCR" / "tesseract.exe",
            program_files_x86 / "Tesseract-OCR" / "tesseract.exe",
            local_app_data / "Programs" / "Tesseract-OCR" / "tesseract.exe",
            Path(os.environ.get("ChocolateyInstall", r"C:\ProgramData\chocolatey")) / "bin" / "tesseract.exe",
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
    return None


@dataclass(slots=True)
class ScanIssue:
    """A parser warning or error associated with one manifest item."""

    evidence_id: str
    source_file: str
    file_format: str
    status: str
    message: str


@dataclass(slots=True)
class EvidenceRecord:
    """One extracted resource, event, or snapshot row."""

    evidence_id: str
    source_file: str
    file_format: str
    system_id: str
    resource_id: str
    observed_at: datetime | None
    timestamps: dict[str, str]
    observed_state: str
    category: str
    topology: str
    evidence_kind: str
    matched_user_ids: set[str] = field(default_factory=set)
    fields: dict[str, Any] = field(default_factory=dict)
    line_number: int | None = None
    source_context: str = ""


@dataclass(slots=True)
class ScanResult:
    """Results for all manifest-listed evidence files."""

    records: list[EvidenceRecord] = field(default_factory=list)
    issues: list[ScanIssue] = field(default_factory=list)
    discovered_files: int = 0
    parsed_files: int = 0
    partial_files: int = 0
    failed_files: int = 0

    @property
    def unparsed_files(self) -> int:
        return self.partial_files + self.failed_files


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().upper() in _ACTIVE_STATES


def _falsey(value: Any) -> bool:
    if isinstance(value, bool):
        return not value
    return str(value).strip().upper() in _INACTIVE_STATES


def _scalar_items(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield from _scalar_items(child, path)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _scalar_items(child, f"{prefix}[{index}]")
    elif value is not None and not isinstance(value, (bytes, bytearray)):
        yield prefix, value


def _leaf_key(path: str) -> str:
    return re.sub(r"\[\d+\]", "", path.rsplit(".", 1)[-1]).casefold()


def _derive_topology(source_file: str, table_name: str = "") -> str:
    table = table_name.casefold()
    if "primary" in table:
        return "primary"
    for marker, topology in _REPLICA_MARKERS:
        if marker in table:
            return topology
    candidate = source_file.casefold()
    for marker, topology in _REPLICA_MARKERS:
        if marker in candidate:
            return topology
    if "primary" in candidate:
        return "primary"
    return "unknown"


def _parse_timestamp_values(fields: Mapping[str, Any]) -> dict[str, str]:
    timestamps: dict[str, str] = {}
    for key, value in _scalar_items(fields):
        leaf = _leaf_key(key)
        if leaf not in _TIMESTAMP_KEYS or not isinstance(value, str) or not value.strip():
            continue
        try:
            parse_iso_timestamp(value, location=f"evidence.{leaf}")
        except DatasetValidationError:
            continue
        timestamps[leaf] = value.strip()
    return timestamps


def _record_state(fields: Mapping[str, Any], category: str) -> str:
    scalars = {_leaf_key(key): value for key, value in _scalar_items(fields)}

    # Explicit read-state fields take priority; reported workflow claims do not.
    for key in _STATE_KEYS:
        value = scalars.get(key)
        normalized = str(value).strip().upper() if value is not None else ""
        if normalized in _KNOWN_STATES:
            if normalized in {"DELETED", "REMOVED"}:
                return "ABSENT"
            if normalized == "INACTIVE":
                return "DISABLED"
            if normalized in {"RUNNING", "ENABLED"}:
                return "ACTIVE"
            return normalized

    if category == "data_flow":
        status = str(scalars.get("status", scalars.get("job_state", ""))).strip().upper()
        if status in {"ACTIVE", "ENABLED", "RUNNING", "IN_PROGRESS"}:
            return "ACTIVE"
        if status in {"INACTIVE", "DISABLED", "STOPPED", "CANCELLED", "COMPLETED"}:
            return "DISABLED"
        for key in ("enabled", "is_enabled", "active", "is_active"):
            value = scalars.get(key)
            if value is not None:
                if _truthy(value):
                    return "ACTIVE"
                if _falsey(value):
                    return "DISABLED"

    if category == "restore_path":
        for key in ("restore_allowed", "can_restore", "restore_enabled"):
            value = scalars.get(key)
            if value is not None:
                return "ACTIVE" if _truthy(value) else "BLOCKED" if _falsey(value) else "UNKNOWN"

    if category in {"personal_record", "data_flow", "restore_path"}:
        return "PRESENT"
    return "UNKNOWN"


def _classify_category(fields: Mapping[str, Any], context: str, matched: set[str]) -> str:
    items = list(_scalar_items(fields))
    keys = {_leaf_key(key) for key, _ in items}
    values = " ".join(str(value).casefold() for _, value in items if isinstance(value, (str, int, bool)))
    words = f"{context} {values}".casefold()

    if ("restore_allowed" in keys or "can_restore" in keys or "restore_enabled" in keys
            or any(term in words for term in ("backup_restore", "restore_path", "backup archive"))):
        return "restore_path"

    has_flow_config = any(key in keys for key in (
        "enabled", "is_enabled", "active", "is_active", "job_state", "sync_status",
    ))
    has_flow_context = any(term in words for term in (
        "connector", "subscription", "sync_job", "sync process", "data_flow",
        "import_job", "export_job", "webhook", "integration", "import", "export",
        "sync", "pipeline", "process", "flow",
    ))
    status_value = str(next((value for key, value in items if _leaf_key(key) in {"status", "job_state", "sync_status"}), "")).upper()
    has_active_status = status_value in {"ACTIVE", "ENABLED", "RUNNING", "IN_PROGRESS"}
    if has_flow_context and (has_flow_config or has_active_status):
        return "data_flow"

    if matched or any(key in keys for key in _IDENTIFIER_FIELDS):
        return "personal_record"
    return "unknown"


def _flatten_json_records(node: Any, context: str = "") -> Iterable[tuple[dict[str, Any], str]]:
    if isinstance(node, list):
        for item in node:
            yield from _flatten_json_records(item, context)
        return
    if not isinstance(node, dict):
        return

    keys = {str(key).casefold() for key in node}
    looks_like_record = bool(keys.intersection({
        "resource_id", "record_id", "ticket_id", "request_id", "archive_key",
        "object_key", "user_id", "customer_reference", "subject_reference",
    }))
    if looks_like_record:
        yield node, context
        return

    for key, child in node.items():
        if isinstance(child, (dict, list)):
            child_context = f"{context} {key}".strip()
            yield from _flatten_json_records(child, child_context)


class EvidenceScanner:
    """Scan participant-provided evidence and match it to known identities."""

    def __init__(
        self,
        dataset_dir: Path | str,
        evidence_manifest: Iterable[Mapping[str, str]],
        identity_resolver: IdentityResolver,
        activity_rows: Iterable[Mapping[str, str]] = (),
    ) -> None:
        self.dataset_dir = Path(dataset_dir).resolve()
        self.evidence_root = (self.dataset_dir / "evidence").resolve()
        self.manifest = [dict(row) for row in evidence_manifest]
        self.identity_resolver = identity_resolver
        self.activity_rows = [dict(row) for row in activity_rows]
        self.ocr_engine_path = find_tesseract_executable()
        if self.ocr_engine_path is not None:
            # PyMuPDF invokes the local executable by name; add its folder when
            # it was found via TESSERACT_CMD or a standard install location.
            engine_directory = str(self.ocr_engine_path.parent)
            current_path = os.environ.get("PATH", "")
            path_entries = {entry.casefold() for entry in current_path.split(os.pathsep) if entry}
            if engine_directory.casefold() not in path_entries:
                os.environ["PATH"] = engine_directory + os.pathsep + current_path
            tessdata_directory = self.ocr_engine_path.parent / "tessdata"
            if tessdata_directory.is_dir() and not os.environ.get("TESSDATA_PREFIX"):
                os.environ["TESSDATA_PREFIX"] = str(tessdata_directory)
        self._result = ScanResult(discovered_files=len(self.manifest))

    def scan_all(self) -> ScanResult:
        """Parse every evidence file listed by file_id in evidence_files.csv."""
        self._result = ScanResult(discovered_files=len(self.manifest))
        for item in self.manifest:
            self._scan_manifest_item(item)
        self._scan_activity_rows()
        return self._result

    def find_for_user(self, user_id: str, *, as_of: datetime | None = None) -> list[EvidenceRecord]:
        """Return this user's matched evidence, excluding evidence after the cutoff."""
        if as_of is not None and as_of.tzinfo is None:
            raise ValueError("as_of must include a time zone")
        records = self._result.records
        return [
            record for record in records
            if user_id in record.matched_user_ids
            and (as_of is None or (record.observed_at is not None and record.observed_at <= as_of))
        ]

    def reconstruct_resource_state(
        self,
        system_id: str,
        resource_id: str,
        *,
        as_of: datetime,
        user_id: str | None = None,
    ) -> str:
        """Return the last directly observed state at or before ``as_of``.

        Activity lines and operation results are marked as claims and are never
        allowed to overwrite a directly observed snapshot state.
        """
        if as_of.tzinfo is None:
            raise ValueError("as_of must include a time zone")
        candidates = [
            record for record in self._result.records
            if record.system_id == system_id
            and record.resource_id == resource_id
            and record.evidence_kind == "observation"
            and record.observed_at is not None
            and record.observed_at <= as_of
            and (user_id is None or user_id in record.matched_user_ids)
        ]
        if not candidates:
            return "UNKNOWN"
        newest_time = max(record.observed_at for record in candidates if record.observed_at is not None)
        newest = [record for record in candidates if record.observed_at == newest_time]
        states = {record.observed_state for record in newest}
        return states.pop() if len(states) == 1 else "UNKNOWN"

    def _scan_manifest_item(self, item: Mapping[str, str]) -> None:
        evidence_id = item.get("file_id", "")
        relative = item.get("file_path", "")
        file_path = (self.dataset_dir / relative).resolve()
        source_file = relative.replace("\\", "/")

        if not file_path.is_relative_to(self.evidence_root):
            self._issue(evidence_id, source_file, "unknown", "error", "Path is outside dataset/evidence.")
            return
        if not file_path.is_file():
            self._issue(evidence_id, source_file, file_path.suffix.lower(), "error", "Manifest file is missing.")
            return

        extension = file_path.suffix.lower()
        origin = item.get("origin", "")
        try:
            if extension == ".json":
                self._parse_json(file_path, item, source_file, origin)
            elif extension == ".sql":
                self._parse_sql(file_path, item, source_file, origin)
            elif extension == ".csv":
                self._parse_csv(file_path, item, source_file, origin)
            elif extension in {".log", ".txt"}:
                self._parse_text(file_path, item, source_file, origin)
            elif extension == ".pdf":
                self._parse_pdf(file_path, item, source_file, origin)
            elif extension in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}:
                self._parse_image(file_path, item, source_file, origin)
            else:
                self._issue(evidence_id, source_file, extension or "unknown", "unsupported", "No parser is available for this file format.")
                return
        except (OSError, UnicodeError, csv.Error, json.JSONDecodeError, sqlite3.Error, ValueError) as exc:
            self._issue(evidence_id, source_file, extension, "error", f"Parser failed: {type(exc).__name__}: {exc}")
            return

    def _scan_activity_rows(self) -> None:
        """Record global activity as a claim, never as proof of resource state."""
        for row in self.activity_rows:
            event_type = row.get("event_type", "").strip().upper()
            fields = dict(row)
            if event_type in _FLOW_ACTIVITY_TYPES:
                context = "data_flow_activity"
            elif event_type in _RESTORE_ACTIVITY_TYPES:
                context = "restore_path_activity"
            else:
                context = "activity"
            record = self._make_record(
                {"file_id": row.get("activity_id", ""), "captured_at": row.get("event_time", "")},
                "dataset/system_activity.csv",
                row.get("target_system") or row.get("source_system") or "unknown_system",
                fields,
                context=context,
                resource_id=row.get("resource_id", ""),
                observed_at=self._timestamp_or_none(row.get("event_time", ""), "system_activity.event_time"),
                evidence_kind="claim",
                free_text=row.get("details", ""),
            )
            record.evidence_id = row.get("activity_id", "")
            if context == "data_flow_activity":
                record.category = "data_flow"
            elif context == "restore_path_activity":
                record.category = "restore_path"
            record.observed_state = "UNKNOWN"
            self._result.records.append(record)

    @staticmethod
    def _timestamp_or_none(value: str, location: str) -> datetime | None:
        try:
            return parse_iso_timestamp(value, location=location)
        except DatasetValidationError:
            return None

    def _issue(self, evidence_id: str, source_file: str, file_format: str, status: str, message: str) -> None:
        self._result.issues.append(ScanIssue(evidence_id, source_file, file_format, status, message))
        if status in {"partial", "unsupported"}:
            self._result.partial_files += 1
        elif status == "error":
            self._result.failed_files += 1

    def _parsed(self) -> None:
        self._result.parsed_files += 1

    def _capture_time(self, item: Mapping[str, str]) -> datetime | None:
        value = item.get("captured_at", "").strip()
        if not value:
            return None
        try:
            return parse_iso_timestamp(value, location=f"evidence_files.csv:{item.get('file_id', '')}")
        except DatasetValidationError:
            return None

    def _matched_users(self, fields: Mapping[str, Any], free_text: str = "") -> set[str]:
        matches: set[str] = set()
        for key, value in _scalar_items(fields):
            canonical_field = _IDENTIFIER_FIELDS.get(_leaf_key(key))
            if canonical_field is not None:
                matches.update(self.identity_resolver.resolve(str(value), canonical_field))
        if free_text:
            matches.update(self.identity_resolver.match_text(free_text))
        return matches

    def _make_record(
        self,
        item: Mapping[str, str],
        source_file: str,
        origin: str,
        fields: Mapping[str, Any],
        *,
        context: str = "",
        table_name: str = "",
        resource_id: str = "",
        observed_at: datetime | None = None,
        evidence_kind: str = "observation",
        line_number: int | None = None,
        free_text: str = "",
    ) -> EvidenceRecord:
        field_copy = dict(fields)
        scalar_values = dict(_scalar_items(field_copy))
        lowered_keys = {_leaf_key(key): value for key, value in scalar_values.items()}
        if not resource_id:
            for key in _RESOURCE_KEYS:
                value = lowered_keys.get(key)
                if value is not None and str(value).strip():
                    resource_id = str(value).strip()
                    break
        if not resource_id:
            resource_id = f"{Path(source_file).stem}:record:{len(self._result.records) + 1}"

        matched = self._matched_users(field_copy, free_text)
        category = _classify_category(field_copy, f"{context} {table_name} {source_file}", matched)
        source_system = lowered_keys.get("origin_system") or lowered_keys.get("system_id") or lowered_keys.get("source_system")
        if source_system:
            system_id = str(source_system)
        else:
            system_id = _TABLE_SYSTEMS.get(table_name.casefold(), "")
        if not system_id:
            system_id = origin or "unknown_system"

        timestamps = _parse_timestamp_values(field_copy)
        captured_at = self._capture_time(item)
        if observed_at is None:
            observed_at = captured_at
        if captured_at is not None:
            timestamps.setdefault("captured_at", item["captured_at"])

        state = _record_state(field_copy, category) if evidence_kind == "observation" else "UNKNOWN"
        return EvidenceRecord(
            evidence_id=item.get("file_id", ""),
            source_file=source_file,
            file_format=Path(source_file).suffix.lower(),
            system_id=system_id,
            resource_id=resource_id,
            observed_at=observed_at,
            timestamps=timestamps,
            observed_state=state,
            category=category,
            topology=_derive_topology(source_file, table_name),
            evidence_kind=evidence_kind,
            matched_user_ids=matched,
            fields=field_copy,
            line_number=line_number,
            source_context=" ".join(part for part in (context, table_name) if part).strip(),
        )

    def _parse_json(self, path: Path, item: Mapping[str, str], source_file: str, origin: str) -> None:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        count = 0
        for fields, context in _flatten_json_records(data):
            record = self._make_record(item, source_file, origin, fields, context=context)
            self._result.records.append(record)
            count += 1
        # A valid JSON manifest with no row-like records is still successfully parsed.
        if count == 0:
            self._result.records.append(
                self._make_record(item, source_file, origin, {"document": "valid JSON document"}, resource_id=f"{path.stem}:document")
            )
        self._parsed()

    def _parse_sql(self, path: Path, item: Mapping[str, str], source_file: str, origin: str) -> None:
        connection = sqlite3.connect(":memory:")

        def restrict_sql(action: int, arg1: str | None, arg2: str | None, database: str | None, trigger: str | None) -> int:
            if action in {sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH}:
                return sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_FUNCTION and (arg2 or arg1 or "").casefold() == "load_extension":
                return sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_PRAGMA and (arg1 or "").casefold() not in {"foreign_keys"}:
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        connection.set_authorizer(restrict_sql)
        try:
            connection.executescript(path.read_text(encoding="utf-8-sig"))
            tables = connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
            for (table_name,) in tables:
                quoted_table = '"' + table_name.replace('"', '""') + '"'
                cursor = connection.execute(f"SELECT * FROM {quoted_table}")
                columns = [description[0] for description in cursor.description or ()]
                for row_index, values in enumerate(cursor.fetchall(), start=1):
                    fields = dict(zip(columns, values, strict=True))
                    embedded = fields.get("record_json")
                    if isinstance(embedded, str):
                        try:
                            decoded = json.loads(embedded)
                        except json.JSONDecodeError:
                            decoded = None
                        if isinstance(decoded, dict):
                            fields.update(decoded)
                    if table_name.casefold() == "snapshot_scope":
                        continue
                    record = self._make_record(
                        item, source_file, origin, fields,
                        table_name=table_name,
                        resource_id=str(fields.get("resource_id", f"{table_name}:row:{row_index}")),
                    )
                    self._result.records.append(record)
            self._parsed()
        finally:
            connection.close()

    def _parse_csv(self, path: Path, item: Mapping[str, str], source_file: str, origin: str) -> None:
        with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            if not reader.fieldnames:
                raise ValueError("CSV has no header row")
            for row_number, row in enumerate(reader, start=2):
                fields = {key: value or "" for key, value in row.items() if key is not None}
                self._result.records.append(
                    self._make_record(item, source_file, origin, fields, resource_id=f"{path.stem}:row:{row_number}")
                )
        self._parsed()

    def _parse_text(self, path: Path, item: Mapping[str, str], source_file: str, origin: str) -> None:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        had_replacements = "\ufffd" in text
        parsed_line_count = 0
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            structured = _STRUCTURED_OBSERVATION_PATTERN.match(line)
            if structured:
                timestamp, system_id, resource_id, record_type, reported_state, detail = (
                    part.strip() for part in structured.groups()
                )
                try:
                    observed_at = parse_iso_timestamp(timestamp, location=f"{path.name}:{line_number}")
                except DatasetValidationError:
                    observed_at = None
                state_name = reported_state.upper()
                observed_state = {
                    "NOT_FOUND": "ABSENT",
                    "MISSING": "ABSENT",
                    "FOUND": "PRESENT",
                    "IDENTIFIABLE": "PRESENT",
                    "ANONYMIZED": "ANONYMIZED",
                    "REDACTED": "REDACTED",
                    "ACTIVE": "ACTIVE",
                    "DISABLED": "DISABLED",
                    "BLOCKED": "BLOCKED",
                }.get(state_name, "UNKNOWN")
                fields: dict[str, Any] = {
                    "event_time": timestamp,
                    "system_id": system_id,
                    "resource_id": resource_id,
                    "record_type": record_type,
                    "observed_state": observed_state,
                    "observation_detail": detail,
                }
                record = self._make_record(
                    item,
                    source_file,
                    origin,
                    fields,
                    context="structured_handover",
                    resource_id=resource_id,
                    observed_at=observed_at,
                    evidence_kind="observation",
                    line_number=line_number,
                    free_text=detail,
                )
                # The handover is a readback snapshot (including NOT_FOUND),
                # not an operation-result claim; its status word is the observed
                # resource state at the recorded timestamp.
                record.system_id = system_id
                self._result.records.append(record)
                parsed_line_count += 1
                continue

            fields: dict[str, Any] = {"message": line}
            observed_at: datetime | None = None
            match = _ISO_PREFIX_PATTERN.match(line.strip())
            if match:
                try:
                    observed_at = parse_iso_timestamp(match.group(1), location=f"{path.name}:{line_number}")
                    fields["event_time"] = match.group(1)
                except DatasetValidationError:
                    observed_at = None
            for key, value in _KEY_VALUE_PATTERN.findall(line):
                normalized_key = key.casefold()
                if normalized_key not in _CLAIM_KEYS:
                    fields.setdefault(key, value.strip())
            resource_match = _RESOURCE_PATTERN.search(line)
            record = self._make_record(
                item, source_file, origin, fields,
                resource_id=resource_match.group() if resource_match else f"{path.stem}:line:{line_number}",
                observed_at=observed_at or self._capture_time(item),
                evidence_kind="claim",
                line_number=line_number,
                free_text=line,
            )
            # Text logs describe activity or system messages; they are not state snapshots.
            self._result.records.append(record)
            parsed_line_count += 1
        if had_replacements:
            self._issue(item.get("file_id", ""), source_file, path.suffix.lower(), "partial", "Text contained undecodable bytes replaced with a marker.")
        else:
            self._parsed()

    def _parse_pdf(self, path: Path, item: Mapping[str, str], source_file: str, origin: str) -> None:
        images_present = False
        ocr_needed = False
        try:
            import pymupdf
        except ImportError:
            try:
                from pypdf import PdfReader  # type: ignore[import-not-found]
            except ImportError:
                self._issue(
                    item.get("file_id", ""), source_file, ".pdf", "unsupported",
                    "PDF text extraction requires the optional free 'PyMuPDF' package; no content was silently skipped.",
                )
                return
            reader = PdfReader(str(path))
            page_text = [page.extract_text() or "" for page in reader.pages]
            images_present = False
        else:
            document = pymupdf.open(path)
            try:
                page_text: list[str] = []
                for page in document:
                    text = page.get_text("text") or ""
                    page_images = page.get_images(full=True)
                    images_present = images_present or bool(page_images)
                    if not text.strip() and page_images:
                        text = self._ocr_page(page)
                        if not text.strip():
                            ocr_needed = True
                    page_text.append(text)
            finally:
                document.close()

        text = "\n".join(page_text)
        if not text.strip():
            if "images_present" in locals() and images_present:
                self._issue(
                    item.get("file_id", ""), source_file, ".pdf", "partial",
                    "PDF contains image-only page content; OCR is required to read it.",
                )
            else:
                self._issue(
                    item.get("file_id", ""), source_file, ".pdf", "partial",
                    "PDF has no extractable text or embedded page images.",
                )
            return
        self._parse_extracted_document_text(
            text,
            item,
            source_file,
            origin,
            ".pdf",
            coverage_complete=not ocr_needed,
            partial_message="Some PDF pages had images but could not be OCR'd; install the local Tesseract OCR engine to read them.",
        )

    def _parse_image(self, path: Path, item: Mapping[str, str], source_file: str, origin: str) -> None:
        dimensions = self._png_dimensions(path) if path.suffix.lower() == ".png" else None
        try:
            import pymupdf
        except ImportError:
            pymupdf = None  # type: ignore[assignment]

        if pymupdf is not None:
            document = pymupdf.open(path)
            try:
                page = document[0]
                text = self._ocr_page(page)
            finally:
                document.close()
            if text.strip():
                self._parse_extracted_document_text(text, item, source_file, origin, path.suffix.lower())
            else:
                detail = f"Image dimensions={dimensions[0]}x{dimensions[1]}. " if dimensions else ""
                if self.ocr_engine_path is None:
                    message = (
                        "Tesseract OCR was not found. Install the Windows Tesseract engine, "
                        "add it to PATH, or set TESSERACT_CMD to its full executable path. "
                        "The image text was not extracted."
                    )
                else:
                    message = "Tesseract was found, but OCR returned no readable text for this image."
                self._issue(
                    item.get("file_id", ""), source_file, path.suffix.lower(), "partial",
                    detail + message,
                )
            return

        try:
            import pytesseract  # type: ignore[import-not-found]
            from PIL import Image  # type: ignore[import-not-found]
        except ImportError:
            detail = f"PNG dimensions={dimensions[0]}x{dimensions[1]}. " if dimensions else ""
            self._issue(item.get("file_id", ""), source_file, path.suffix.lower(), "partial", detail + "Image OCR needs optional Pillow, pytesseract, and a local Tesseract executable; pixel contents were not readable.")
            return

        try:
            text = pytesseract.image_to_string(Image.open(path))
        except Exception as exc:  # OCR engines report varied platform-specific errors.
            self._issue(item.get("file_id", ""), source_file, path.suffix.lower(), "unsupported", f"Image OCR could not run: {type(exc).__name__}: {exc}")
            return
        self._parse_extracted_document_text(text, item, source_file, origin, path.suffix.lower())

    def _ocr_page(self, page: Any) -> str:
        """Use PyMuPDF's local OCR bridge when Tesseract is installed."""
        if self.ocr_engine_path is None:
            return ""
        try:
            text_page = page.get_textpage_ocr(language="eng", dpi=200, full=True)
            return page.get_text("text", textpage=text_page) or ""
        except Exception:
            # Missing executable or language data is reported by the caller as partial.
            return ""

    @staticmethod
    def _png_dimensions(path: Path) -> tuple[int, int] | None:
        with path.open("rb") as image_file:
            header = image_file.read(24)
        if len(header) >= 24 and header[:8] == b"\x89PNG\r\n\x1a\n":
            return struct.unpack(">II", header[16:24])
        return None

    def _parse_extracted_document_text(
        self,
        text: str,
        item: Mapping[str, str],
        source_file: str,
        origin: str,
        file_format: str,
        *,
        coverage_complete: bool = True,
        partial_message: str = "Some document content could not be extracted.",
    ) -> None:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        all_text = "\n".join(lines)
        if not lines:
            self._issue(item.get("file_id", ""), source_file, file_format, "partial", "No readable text was extracted.")
            return
        resource_match = _RESOURCE_PATTERN.search(all_text)
        fields = {"document_text": all_text}
        self._result.records.append(
            self._make_record(
                item, source_file, origin, fields,
                resource_id=resource_match.group() if resource_match else f"{Path(source_file).stem}:document",
                observed_at=self._capture_time(item),
                free_text=all_text,
            )
        )
        if coverage_complete:
            self._parsed()
        else:
            self._issue(item.get("file_id", ""), source_file, file_format, "partial", partial_message)


def load_manifest(dataset_dir: Path | str) -> list[dict[str, str]]:
    """Read evidence_files.csv with the shared CSV reader conventions."""
    path = Path(dataset_dir).resolve() / "evidence_files.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        expected = ("file_id", "file_path", "origin", "captured_at")
        if tuple(reader.fieldnames or ()) != expected:
            raise ValueError(f"Unexpected evidence manifest columns: {reader.fieldnames}")
        return [{key: value or "" for key, value in row.items()} for row in reader]