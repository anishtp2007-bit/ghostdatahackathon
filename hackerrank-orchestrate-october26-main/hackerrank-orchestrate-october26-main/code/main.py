"""Validate and summarize the participant-facing challenge data."""

from __future__ import annotations

import argparse
from collections import Counter

from data_loader import load_dataset
from evidence_scanner import EvidenceScanner, load_manifest
from identity_resolver import IdentityResolver


def main() -> None:
	"""Validate inputs, optionally scan evidence, and never write predictions."""
	parser = argparse.ArgumentParser(description="Validate challenge data or scan its evidence files.")
	parser.add_argument(
		"--scan-evidence",
		action="store_true",
		help="Parse the evidence manifest and summarize extracted records.",
	)
	args = parser.parse_args()

	dataset = load_dataset()
	print("Dataset structure is valid.")
	print(f"Deletion requests: {len(dataset.deletion_requests)}")
	print(f"Current identities: {len(dataset.user_identities)}")
	print(f"Identity history entries: {len(dataset.user_audit_log)}")
	print(f"Deletion rules: {len(dataset.deletion_rules)}")
	print(f"System activity records: {len(dataset.system_activity)}")
	print(f"Evidence files in manifest: {len(dataset.evidence_files)}")
	if args.scan_evidence:
		resolver = IdentityResolver(dataset.user_identities, dataset.user_audit_log)
		scanner = EvidenceScanner(
			dataset.dataset_dir,
			load_manifest(dataset.dataset_dir),
			resolver,
			dataset.system_activity,
		)
		result = scanner.scan_all()
		formats = Counter(row["file_path"].rsplit(".", 1)[-1].lower() for row in dataset.evidence_files)
		print(f"Evidence files discovered: {result.discovered_files}")
		print(f"Fully parsed evidence files: {result.parsed_files}")
		print(f"Partially parsed evidence files: {result.partial_files}")
		print(f"Evidence files with errors: {result.failed_files}")
		print(f"Evidence records extracted: {len(result.records)}")
		print(f"Evidence formats: {dict(sorted(formats.items()))}")
		ocr_status = str(scanner.ocr_engine_path) if scanner.ocr_engine_path else "not found; PNG OCR is reported as partial"
		print(f"Local Tesseract OCR engine: {ocr_status}")
		print(f"Directly observed active data flows: {sum(record.category == 'data_flow' and record.observed_state == 'ACTIVE' and record.evidence_kind == 'observation' for record in result.records)}")
		print(f"Directly observed active restore paths: {sum(record.category == 'restore_path' and record.observed_state == 'ACTIVE' and record.evidence_kind == 'observation' for record in result.records)}")
		print(f"Activity log rows treated only as claims: {sum(record.evidence_kind == 'claim' and record.source_file == 'dataset/system_activity.csv' for record in result.records)}")
		if result.issues:
			print("Formats needing additional support or review:")
			for file_format, count in sorted(Counter(issue.file_format for issue in result.issues).items()):
				print(f"  {file_format}: {count} file(s)")
	print("No case decisions or output.csv predictions were generated.")


if __name__ == "__main__":
	main()
