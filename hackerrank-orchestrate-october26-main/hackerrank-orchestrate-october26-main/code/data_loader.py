"""Load and validate the challenge's participant-facing CSV files."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


class DatasetValidationError(ValueError):
    """Raised when a challenge input file does not meet its expected structure."""


CSV_SCHEMAS: dict[str, tuple[str, ...]] = {
    "deletion_requests.csv": (
        "case_id", "email", "requested_at", "reported_status", "reported_at",
        "audit_at", "request_text",
    ),
    "sample_deletion_requests.csv": (
        "case_id", "email", "requested_at", "reported_status", "reported_at",
        "audit_at", "request_text",
    ),
    "user_identities.csv": (
        "user_id", "name", "email", "phone", "email_hash", "updated_at",
    ),
    "user_audit_log.csv": (
        "audit_id", "user_id", "event_time", "field_name", "old_value", "new_value",
    ),
    "deletion_rules.csv": ("rule_id", "data_category", "condition", "required_action"),
    "system_activity.csv": (
        "activity_id", "event_time", "event_type", "source_system", "target_system",
        "resource_id", "reported_result", "details",
    ),
    "evidence_files.csv": ("file_id", "file_path", "origin", "captured_at"),
    "sample_output.csv": ("case_id", "deletion_status", "steps_json"),
    "output.csv": ("case_id", "deletion_status", "steps_json"),
}

TIMESTAMP_COLUMNS: dict[str, tuple[str, ...]] = {
    "deletion_requests.csv": ("requested_at", "reported_at", "audit_at"),
    "sample_deletion_requests.csv": ("requested_at", "reported_at", "audit_at"),
    "user_identities.csv": ("updated_at",),
    "user_audit_log.csv": ("event_time",),
    "system_activity.csv": ("event_time",),
    "evidence_files.csv": ("captured_at",),
}

_ALLOWED_STATUSES = {"COMPLETE", "INCOMPLETE", "NOT_APPLICABLE"}


@dataclass(slots=True)
class LoadedDataset:
    """All challenge CSV tables, kept as rows of their original string values."""

    dataset_dir: Path
    deletion_requests: list[dict[str, str]]
    sample_deletion_requests: list[dict[str, str]]
    user_identities: list[dict[str, str]]
    user_audit_log: list[dict[str, str]]
    deletion_rules: list[dict[str, str]]
    system_activity: list[dict[str, str]]
    evidence_files: list[dict[str, str]]
    sample_output: list[dict[str, str]]
    output_template: list[dict[str, str]]


def parse_iso_timestamp(value: str, *, location: str = "timestamp") -> datetime:
    """Parse an ISO-8601 timestamp and require an explicit time zone."""
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise DatasetValidationError(f"Invalid ISO-8601 value for {location}: {value!r}") from exc
    if parsed.tzinfo is None:
        raise DatasetValidationError(f"Timestamp for {location} must include a time zone")
    return parsed


def read_csv_table(
    path: Path,
    expected_columns: Iterable[str],
    *,
    timestamp_columns: Iterable[str] = (),
    optional_empty_columns: Iterable[str] = (),
) -> list[dict[str, str]]:
    """Read one CSV file and validate its header, required cells, and timestamps."""
    if not path.is_file():
        raise DatasetValidationError(f"Required CSV file does not exist: {path}")

    expected = tuple(expected_columns)
    optional_empty = set(optional_empty_columns)
    timestamp_fields = tuple(timestamp_columns)
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            actual = tuple(reader.fieldnames or ())
            if actual != expected:
                raise DatasetValidationError(
                    f"Unexpected columns in {path.name}: expected {expected}, found {actual}"
                )

            rows: list[dict[str, str]] = []
            for row_number, raw_row in enumerate(reader, start=2):
                if None in raw_row:
                    raise DatasetValidationError(
                        f"Too many CSV values in {path.name} at row {row_number}"
                    )
                row = {key: value or "" for key, value in raw_row.items()}
                for column, value in row.items():
                    if not value.strip() and column not in optional_empty:
                        raise DatasetValidationError(
                            f"Missing value in {path.name}, row {row_number}, column {column}"
                        )
                for column in timestamp_fields:
                    parse_iso_timestamp(row[column], location=f"{path.name}:{row_number}:{column}")
                rows.append(row)
    except csv.Error as exc:
        raise DatasetValidationError(f"Could not parse {path.name}: {exc}") from exc
    return rows


def _validate_unique(rows: list[dict[str, str]], column: str, file_name: str) -> None:
    values = [row[column] for row in rows]
    if len(values) != len(set(values)):
        raise DatasetValidationError(f"Duplicate {column} values found in {file_name}")


def _validate_relationships(dataset: LoadedDataset) -> None:
    _validate_unique(dataset.user_identities, "user_id", "user_identities.csv")
    _validate_unique(dataset.deletion_requests, "case_id", "deletion_requests.csv")
    _validate_unique(dataset.sample_deletion_requests, "case_id", "sample_deletion_requests.csv")
    _validate_unique(dataset.user_audit_log, "audit_id", "user_audit_log.csv")
    _validate_unique(dataset.deletion_rules, "rule_id", "deletion_rules.csv")
    _validate_unique(dataset.system_activity, "activity_id", "system_activity.csv")
    _validate_unique(dataset.evidence_files, "file_id", "evidence_files.csv")
    _validate_unique(dataset.sample_output, "case_id", "sample_output.csv")
    _validate_unique(dataset.output_template, "case_id", "output.csv")

    identity_ids = {row["user_id"] for row in dataset.user_identities}
    unknown_audit_users = {
        row["user_id"] for row in dataset.user_audit_log if row["user_id"] not in identity_ids
    }
    if unknown_audit_users:
        raise DatasetValidationError(
            "user_audit_log.csv references unknown user_id values: "
            + ", ".join(sorted(unknown_audit_users))
        )

    sample_request_ids = {row["case_id"] for row in dataset.sample_deletion_requests}
    sample_output_ids = {row["case_id"] for row in dataset.sample_output}
    if sample_request_ids != sample_output_ids:
        raise DatasetValidationError("Sample request and sample output case_id values do not match")

    for row_number, row in enumerate(dataset.sample_output, start=2):
        if row["deletion_status"] not in _ALLOWED_STATUSES:
            raise DatasetValidationError(
                f"Invalid deletion_status in sample_output.csv at row {row_number}"
            )
        try:
            steps = json.loads(row["steps_json"])
        except json.JSONDecodeError as exc:
            raise DatasetValidationError(
                f"Invalid JSON in sample_output.csv at row {row_number}"
            ) from exc
        if not isinstance(steps, list):
            raise DatasetValidationError(
                f"steps_json must be a JSON array in sample_output.csv at row {row_number}"
            )

    evidence_root = (dataset.dataset_dir / "evidence").resolve()
    for row in dataset.evidence_files:
        evidence_path = (dataset.dataset_dir / row["file_path"]).resolve()
        if not evidence_path.is_relative_to(evidence_root):
            raise DatasetValidationError(
                f"Evidence path escapes the evidence directory: {row['file_path']}"
            )
        if not evidence_path.is_file():
            raise DatasetValidationError(f"Evidence file listed in manifest is missing: {row['file_path']}")


def load_dataset(dataset_dir: Path | str | None = None) -> LoadedDataset:
    """Load and validate every CSV in dataset/ without changing the source files."""
    if dataset_dir is None:
        dataset_dir = Path(__file__).resolve().parents[1] / "dataset"
    directory = Path(dataset_dir).resolve()

    tables: dict[str, list[dict[str, str]]] = {}
    for file_name, columns in CSV_SCHEMAS.items():
        optional_empty = ("old_value", "new_value") if file_name == "user_audit_log.csv" else ()
        tables[file_name] = read_csv_table(
            directory / file_name,
            columns,
            timestamp_columns=TIMESTAMP_COLUMNS.get(file_name, ()),
            optional_empty_columns=optional_empty,
        )

    dataset = LoadedDataset(
        dataset_dir=directory,
        deletion_requests=tables["deletion_requests.csv"],
        sample_deletion_requests=tables["sample_deletion_requests.csv"],
        user_identities=tables["user_identities.csv"],
        user_audit_log=tables["user_audit_log.csv"],
        deletion_rules=tables["deletion_rules.csv"],
        system_activity=tables["system_activity.csv"],
        evidence_files=tables["evidence_files.csv"],
        sample_output=tables["sample_output.csv"],
        output_template=tables["output.csv"],
    )
    _validate_relationships(dataset)
    return dataset