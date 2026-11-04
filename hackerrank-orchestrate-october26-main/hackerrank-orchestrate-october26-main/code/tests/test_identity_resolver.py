"""Tests for current and historical identifier resolution."""

from __future__ import annotations

import sys
import unittest
from datetime import datetime
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from data_loader import load_dataset  # noqa: E402
from identity_resolver import IdentityResolver  # noqa: E402


class IdentityResolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dataset = load_dataset()
        cls.resolver = IdentityResolver(dataset.user_identities, dataset.user_audit_log)
        cls.dataset = dataset

    def test_resolves_a_current_email(self) -> None:
        identity = self.dataset.user_identities[0]
        self.assertEqual(self.resolver.resolve_email(identity["email"]), {identity["user_id"]})

    def test_resolves_a_historical_email(self) -> None:
        event = next(
            row for row in self.dataset.user_audit_log
            if row["field_name"].casefold() == "email" and row["old_value"].strip()
        )
        self.assertIn(event["user_id"], self.resolver.resolve_email(event["old_value"]))

    def test_email_lookup_ignores_letter_case(self) -> None:
        identity = self.dataset.user_identities[0]
        self.assertEqual(
            self.resolver.resolve_email(identity["email"].swapcase()),
            {identity["user_id"]},
        )

    def test_unknown_email_returns_no_users(self) -> None:
        self.assertEqual(self.resolver.resolve_email("not-in-the-data@example.invalid"), set())

    def test_as_of_excludes_later_identity_changes(self) -> None:
        resolver = IdentityResolver(
            [{
                "user_id": "user_test",
                "name": "Test Person",
                "email": "now@example.test",
                "updated_at": "2026-03-01T00:00:00Z",
            }],
            [
                {
                    "user_id": "user_test",
                    "event_time": "2026-01-02T00:00:00Z",
                    "field_name": "email",
                    "old_value": "old@example.test",
                    "new_value": "now@example.test",
                },
                {
                    "user_id": "user_test",
                    "event_time": "2026-02-02T00:00:00Z",
                    "field_name": "email",
                    "old_value": "now@example.test",
                    "new_value": "future@example.test",
                },
            ],
            as_of=datetime.fromisoformat("2026-01-15T00:00:00+00:00"),
        )
        self.assertEqual(resolver.resolve_email("old@example.test"), {"user_test"})
        self.assertEqual(resolver.resolve_email("future@example.test"), set())


if __name__ == "__main__":
    unittest.main()