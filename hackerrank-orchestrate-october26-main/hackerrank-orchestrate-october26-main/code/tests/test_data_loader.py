"""Tests for loading and validating the supplied challenge tables."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from data_loader import DatasetValidationError, load_dataset, read_csv_table  # noqa: E402


class DataLoaderTests(unittest.TestCase):
    def test_loads_and_validates_the_real_dataset(self) -> None:
        dataset = load_dataset()
        self.assertEqual(len(dataset.deletion_requests), 518)
        self.assertEqual(len(dataset.sample_deletion_requests), 32)
        self.assertEqual(len(dataset.user_identities), 850)
        self.assertEqual(len(dataset.evidence_files), 94)
        self.assertEqual(len(dataset.sample_output), 32)
        self.assertEqual(dataset.output_template, [])

    def test_rejects_a_csv_with_the_wrong_header(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            csv_path = Path(temporary_directory) / "bad.csv"
            csv_path.write_text("wrong,value\na,b\n", encoding="utf-8")
            with self.assertRaises(DatasetValidationError):
                read_csv_table(csv_path, ("expected", "value"))

    def test_rejects_a_missing_required_cell(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            csv_path = Path(temporary_directory) / "missing.csv"
            csv_path.write_text("one,two\nvalue,\n", encoding="utf-8")
            with self.assertRaises(DatasetValidationError):
                read_csv_table(csv_path, ("one", "two"))


if __name__ == "__main__":
    unittest.main()