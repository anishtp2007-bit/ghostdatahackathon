"""Resolve current and historical customer identifiers to stable user IDs."""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime
from typing import Iterable, Mapping

from data_loader import parse_iso_timestamp


def _normalize(field_name: str, value: str) -> str:
    """Normalize an identifier enough for case and formatting differences."""
    value = value.strip()
    field = field_name.strip().casefold()
    if field in {"email", "email_hash"}:
        return value.casefold()
    if field == "phone":
        return re.sub(r"\D", "", value)
    if field == "name":
        return " ".join(value.casefold().split())
    return value.casefold()


class IdentityResolver:
    """Build a searchable index over current and historical identity values.

    Results are sets because an identifier can be ambiguous in other datasets.
    An optional ``as_of`` timestamp excludes identity changes recorded later.
    """

    _CURRENT_ID_FIELDS = ("name", "email", "phone", "email_hash")
    _TEXT_PATTERNS = {
        "email": re.compile(r"(?i)(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Z]{2,}(?![\w.-])"),
        "email_hash": re.compile(r"(?i)(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])"),
        "phone": re.compile(r"(?<!\w)\+?\d[\d\s().-]{6,}\d(?!\w)"),
        "user_id": re.compile(r"(?i)\buser_[\w-]+\b"),
    }

    def __init__(
        self,
        identities: Iterable[Mapping[str, str]],
        audit_log: Iterable[Mapping[str, str]],
        *,
        as_of: datetime | None = None,
    ) -> None:
        if as_of is not None and as_of.tzinfo is None:
            raise ValueError("as_of must include a time zone")

        self._user_ids: set[str] = set()
        self._values: dict[str, dict[str, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        self._raw_identifiers: dict[str, dict[str, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        self._name_pattern: re.Pattern[str] | None = None

        for identity in identities:
            user_id = identity["user_id"].strip()
            self._user_ids.add(user_id)
            self._add(user_id, "user_id", user_id)
            updated_at = identity.get("updated_at", "")
            if as_of is not None and updated_at:
                if parse_iso_timestamp(updated_at, location="user_identities.updated_at") > as_of:
                    continue
            for field_name in self._CURRENT_ID_FIELDS:
                self._add(user_id, field_name, identity.get(field_name, ""))

        for event in audit_log:
            event_time = event.get("event_time", "")
            if as_of is not None and event_time:
                if parse_iso_timestamp(event_time, location="user_audit_log.event_time") > as_of:
                    continue
            user_id = event["user_id"].strip()
            field_name = event["field_name"].strip()
            self._add(user_id, field_name, event.get("old_value", ""))
            self._add(user_id, field_name, event.get("new_value", ""))

        names = sorted(self._values.get("name", {}), key=len, reverse=True)
        if names:
            alternatives = "|".join(re.escape(name) for name in names if len(name) >= 3)
            if alternatives:
                self._name_pattern = re.compile(
                    r"(?<!\w)(?:" + alternatives + r")(?!\w)", re.IGNORECASE
                )

    def _add(self, user_id: str, field_name: str, value: str) -> None:
        if not value.strip():
            return
        normalized = _normalize(field_name, value)
        field = field_name.strip().casefold()
        if normalized:
            self._values[field][normalized].add(user_id)
            self._raw_identifiers[user_id][field].add(value.strip())

    def resolve(self, value: str, field_name: str = "email") -> set[str]:
        """Return all user IDs associated with this identifier value."""
        normalized = _normalize(field_name, value)
        return set(self._values.get(field_name.strip().casefold(), {}).get(normalized, set()))

    def resolve_email(self, email: str) -> set[str]:
        """Look up an email address among both current and historical values."""
        return self.resolve(email, "email")

    def match_text(self, text: str) -> set[str]:
        """Find users whose known identifiers occur in free text."""
        matches: set[str] = set()
        for field_name, pattern in self._TEXT_PATTERNS.items():
            for match in pattern.finditer(text):
                matches.update(self.resolve(match.group(), field_name))

        # Names are not reliably tokenizable, so match known full names as phrases.
        if self._name_pattern is not None:
            for match in self._name_pattern.finditer(text):
                matches.update(self._values["name"].get(match.group().casefold(), set()))
        return matches

    def identifiers_for(self, user_id: str) -> dict[str, set[str]]:
        """Return raw known identifier values grouped by their field names."""
        return {
            field_name: set(values)
            for field_name, values in self._raw_identifiers.get(user_id, {}).items()
        }