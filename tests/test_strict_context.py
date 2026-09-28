"""Regression tests for fail-closed user and thread context handling."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils.auth_utils import (  # noqa: E402
    MissingThreadContextError,
    MissingUserContextError,
    UserContextMismatchError,
    require_runtime_user_id,
    require_thread_id,
    require_user_id,
)
from utils.voice_context import extract_outbound_user_id  # noqa: E402


class TestStrictUserContext(unittest.TestCase):
    def test_resolves_authenticated_identity(self):
        config = {
            "configurable": {
                "langgraph_auth_user": {"identity": "user-a"},
                "owner": "user-a",
            },
            "metadata": {"owner": "user-a"},
        }
        self.assertEqual(require_user_id(config), "user-a")

    def test_missing_context_does_not_use_runtime_state(self):
        runtime = SimpleNamespace(config={}, state={"user_id": "state-user"})
        with self.assertRaises(MissingUserContextError):
            require_runtime_user_id(runtime)

    def test_default_user_is_rejected(self):
        with self.assertRaises(MissingUserContextError):
            require_user_id({"configurable": {"owner": "default"}})

    def test_conflicting_context_is_rejected(self):
        config = {
            "configurable": {
                "langgraph_auth_user": {"identity": "user-a"},
                "owner": "user-b",
            }
        }
        with self.assertRaises(UserContextMismatchError):
            require_user_id(config)

    def test_thread_id_is_required(self):
        with self.assertRaises(MissingThreadContextError):
            require_thread_id({"configurable": {}})
        with self.assertRaises(MissingThreadContextError):
            require_thread_id({"configurable": {"thread_id": "default_thread"}})
        self.assertEqual(
            require_thread_id({"configurable": {"thread_id": "thread-123"}}),
            "thread-123",
        )


class TestVoiceOwnershipParsing(unittest.TestCase):
    def test_extracts_all_outbound_room_formats(self):
        reminder_id = "12345678-1234-1234-1234-123456789abc"
        self.assertEqual(
            extract_outbound_user_id(f"outbound-reminder-{reminder_id}-user-a"),
            "user-a",
        )
        self.assertEqual(
            extract_outbound_user_id("outbound-supervisor-sup_123-user-b"),
            "user-b",
        )
        self.assertEqual(
            extract_outbound_user_id("outbound-user-c-1720000000"),
            "user-c",
        )

    def test_malformed_room_never_becomes_default_user(self):
        for room_name in (
            "outbound-reminder-bad",
            "outbound-supervisor-bad",
            "outbound-user-without-timestamp",
            "inbound-101",
        ):
            with self.subTest(room_name=room_name):
                with self.assertRaises((ValueError, MissingUserContextError)):
                    extract_outbound_user_id(room_name)


if __name__ == "__main__":
    unittest.main()
