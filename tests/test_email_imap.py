"""Unit tests for IMAP ID formatting (NetEase 163 Unsafe Login).

NetEase docs require a single parenthesized ID argument after LOGIN:
https://help.mail.163.com/faqDetail.do?code=d7a5dc8471cd0c0e8b4b8f4f8e49998b374173cfe9171305fa1ce630d7f67ac2eda07326646e6eb0
"""

import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aioimaplib.aioimaplib import Command, arguments_rfs2971  # noqa: E402

from services.email import (  # noqa: E402
    IMAP_CLIENT_ID,
    _extract_rfc822_bytes,
    build_imap_id_arg,
    requires_imap_id,
)


NETEASE_DOC_ID = (
    '("name" "OpenAlfred" "version" "1.0.0" '
    '"vendor" "OpenAlfred" "support-email" "openalfred@localhost")'
)


class TestImapIdPayload(unittest.TestCase):
    def test_payload_matches_netease_java_wire_form(self):
        payload = build_imap_id_arg(IMAP_CLIENT_ID)
        self.assertEqual(payload, NETEASE_DOC_ID)
        self.assertTrue(payload.startswith("("))
        self.assertTrue(payload.endswith(")"))
        self.assertFalse(payload.startswith("( "))
        self.assertFalse(payload.endswith(" )"))

    def test_aioimaplib_default_id_splits_parentheses(self):
        """Regression: library helper is NOT what 163 accepts."""
        split_args = arguments_rfs2971(**IMAP_CLIENT_ID)
        library_form = " ".join(split_args)
        self.assertEqual(split_args[0], "(")
        self.assertEqual(split_args[-1], ")")
        self.assertNotEqual(library_form, NETEASE_DOC_ID)
        self.assertTrue(library_form.startswith("( "))

    def test_command_string_is_tag_id_then_single_atom(self):
        payload = build_imap_id_arg(IMAP_CLIENT_ID)
        loop = asyncio.new_event_loop()
        try:
            cmd = Command("ID", "A1", payload, loop=loop)
            self.assertEqual(str(cmd), f"A1 ID {NETEASE_DOC_ID}")
        finally:
            loop.close()

    def test_requires_id_for_chinese_providers(self):
        self.assertTrue(requires_imap_id("imap.163.com", "user@163.com"))
        self.assertTrue(requires_imap_id("imap.qq.com", "user@qq.com"))
        self.assertTrue(requires_imap_id("imap.gmail.com", "user@126.com"))
        self.assertFalse(requires_imap_id("imap.gmail.com", "user@gmail.com"))


class TestFetchParsing(unittest.TestCase):
    def test_extract_header_literal(self):
        lines = [
            b"1 FETCH (BODY[HEADER.FIELDS (SUBJECT FROM DATE)] {32}",
            bytearray(b"Subject: hi\r\nFrom: a@b.c\r\n\r\n"),
            b")",
        ]
        raw = _extract_rfc822_bytes(lines)
        self.assertIn(b"Subject: hi", raw)
        self.assertNotIn(b"FETCH", raw)


class TestImapSessionOrder(unittest.IsolatedAsyncioTestCase):
    async def test_id_is_sent_after_login_before_select(self):
        order: list[str] = []

        async def login(_user, _password):
            order.append("login")
            return SimpleNamespace(result="OK", lines=[])

        async def execute(cmd):
            order.append(f"cmd:{cmd.name}")
            if cmd.name == "ID":
                self.assertEqual(cmd.args, (NETEASE_DOC_ID,))
            return SimpleNamespace(result="OK", lines=[b"OK"])

        async def select(_mailbox):
            order.append("select")
            return SimpleNamespace(result="OK", lines=[b"3 EXISTS"])

        fake = MagicMock()
        fake.wait_hello_from_server = AsyncMock(side_effect=lambda: order.append("hello"))
        fake.login = login
        fake.select = select
        fake.logout = AsyncMock()
        fake.protocol = MagicMock()
        fake.protocol.new_tag.return_value = "A1"
        fake.protocol.loop = asyncio.get_running_loop()
        fake.protocol.execute = execute

        with patch("services.email.aioimaplib.IMAP4_SSL", return_value=fake):
            from services.email import _imap_session

            async with _imap_session(
                "imap.163.com", 993, "user@163.com", "secret"
            ) as (_imap, exists):
                self.assertEqual(exists, 3)

        self.assertEqual(order, ["hello", "login", "cmd:ID", "select"])


if __name__ == "__main__":
    unittest.main()
