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
    def test_mixed_timezone_dates_can_be_sorted_across_accounts(self):
        from services.email import _date_sort_key

        rows = [{"date": "2026-10-02T00:00:00Z"}, {"date": "2025-01-01"}, {"date": "invalid"}]
        self.assertEqual(sorted(rows, key=_date_sort_key, reverse=True), rows)

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
        fake._client_task = asyncio.create_task(asyncio.sleep(0))
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

    async def test_connection_failure_is_reported_without_waiting_for_greeting(self):
        from services.email import _imap_session, EmailServiceException

        async def reset():
            raise ConnectionResetError("private connection details")

        fake = MagicMock()
        fake._client_task = asyncio.create_task(reset())
        fake.wait_hello_from_server = AsyncMock()
        fake.login = AsyncMock()
        fake.logout = AsyncMock()
        fake.protocol.transport = None
        with patch("services.email.aioimaplib.IMAP4_SSL", return_value=fake):
            with self.assertRaises(EmailServiceException) as caught:
                async with _imap_session("imap.163.com", 993, "user@163.com", "secret"):
                    self.fail("A failed connection must not yield a session")
        self.assertEqual(caught.exception.code, "connection_failed")
        self.assertNotIn("private", str(caught.exception))
        fake.wait_hello_from_server.assert_not_awaited()
        fake.login.assert_not_awaited()
        fake.logout.assert_not_awaited()
        self.assertTrue(fake._client_task.done())

    async def test_timeout_cancels_connection_task(self):
        from services.email import _imap_session, EmailServiceException

        fake = MagicMock()
        fake._client_task = asyncio.create_task(asyncio.sleep(60))
        fake.logout = AsyncMock()
        fake.protocol.transport = None
        with (
            patch("services.email.aioimaplib.IMAP4_SSL", return_value=fake),
            patch("services.email.IMAP_TIMEOUT", 0.01),
        ):
            with self.assertRaises(EmailServiceException) as caught:
                async with _imap_session("imap.163.com", 993, "user@163.com", "secret"):
                    self.fail("A timed out connection must not yield a session")
        self.assertEqual(caught.exception.code, "timeout")
        self.assertTrue(fake._client_task.cancelled())
        fake.logout.assert_not_awaited()


class TestAccountCoverage(unittest.IsolatedAsyncioTestCase):
    async def test_inbox_route_includes_account_failures_for_current_user(self):
        import httpx
        from fastapi import FastAPI
        from routers.email import router
        from routers.auth import get_current_user
        from schemas.responses import EmailInboxResponse
        from services import email as service

        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_current_user] = lambda: {"id": "alice"}
        batch = service.EmailBatch([], [{"account_id": "a", "email": "a@example.com", "succeeded": False, "count": 0, "error": "连接超时", "error_code": "timeout"}])
        with patch.object(service, "get_recent_emails", AsyncMock(return_value=batch)) as fetch:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                response = await client.get("/api/emails/inbox")
        self.assertEqual(response.status_code, 200)
        data = EmailInboxResponse.model_validate(response.json())
        self.assertEqual(data.accounts[0].error_code, "timeout")
        fetch.assert_awaited_once_with(user_id="alice", limit=15, per_account=True, allow_all_failed=True)

    async def test_legacy_recent_route_does_not_hide_a_read_failure(self):
        from fastapi import HTTPException
        from routers.email import get_recent_emails_api
        from services import email as service

        with patch.object(service, "get_recent_emails", AsyncMock(side_effect=service.EmailServiceException("连接超时"))):
            with self.assertRaises(HTTPException) as caught:
                await get_recent_emails_api({"id": "alice"})
        self.assertEqual(caught.exception.status_code, 502)

    async def test_inbox_preserves_older_account_mail_and_failure_details(self):
        from services import email as service

        accounts = [{"account_id": key, "email_address": key + "@example.com", "encrypted_password": "cipher"} for key in ("gmail", "qq", "163")]
        recent = [{"id": str(n), "account_id": "gmail", "date": "2026-10-02T00:00:00+00:00"} for n in range(15)]
        older = [{"id": "1", "account_id": "qq", "date": "2025-01-01T00:00:00+00:00"}]
        with (
            patch.object(service, "get_email_credentials", AsyncMock(return_value=accounts)),
            patch.object(service, "decrypt_password", return_value="secret"),
            patch.object(service, "_fetch_recent_for_account", AsyncMock(side_effect=[recent, older, service.EmailServiceException("连接被中断", code="connection_failed")])),
        ):
            batch = await service.get_recent_emails("alice", limit=15, per_account=True, allow_all_failed=True)
        self.assertEqual(len(batch), 16)
        self.assertEqual(batch[-1]["account_id"], "qq")
        self.assertEqual([r["succeeded"] for r in batch.coverage], [True, True, False])
        self.assertEqual(batch.coverage[-1]["error_code"], "connection_failed")
        self.assertEqual(batch.coverage[-1]["error"], "连接被中断")
        self.assertNotIn("password", accounts[0])

    async def test_bad_credentials_do_not_block_other_accounts(self):
        from services import email as service

        accounts = [{"account_id": key, "email_address": key + "@example.com", "encrypted_password": key} for key in ("broken", "empty")]
        with (
            patch.object(service, "get_email_credentials", AsyncMock(return_value=accounts)),
            patch.object(service, "decrypt_password", side_effect=[ValueError("secret"), "valid"]),
            patch.object(service, "_fetch_recent_for_account", AsyncMock(return_value=[])),
        ):
            batch = await service.get_recent_emails("alice", allow_all_failed=True)
        self.assertEqual(batch, [])
        self.assertEqual(batch.coverage[0]["error_code"], "credentials_invalid")
        self.assertTrue(batch.coverage[1]["succeeded"])

    async def test_all_failed_inbox_returns_coverage(self):
        from services import email as service

        accounts = [{"account_id": "a", "email_address": "a@example.com", "encrypted_password": "cipher"}]
        with (
            patch.object(service, "get_email_credentials", AsyncMock(return_value=accounts)),
            patch.object(service, "decrypt_password", return_value="secret"),
            patch.object(service, "_fetch_recent_for_account", AsyncMock(side_effect=service.EmailServiceException("连接超时", code="timeout"))),
        ):
            batch = await service.get_recent_emails("alice", allow_all_failed=True)
        self.assertEqual(batch, [])
        self.assertEqual(batch.coverage[0]["error_code"], "timeout")

    async def test_rejected_fetch_is_not_reported_as_an_empty_inbox(self):
        from services.email import _fetch_headers, EmailServiceException

        imap = MagicMock()
        imap.fetch = AsyncMock(return_value=SimpleNamespace(result="NO", lines=[]))
        with self.assertRaises(EmailServiceException) as caught:
            await _fetch_headers(imap, "1")
        self.assertEqual(caught.exception.code, "fetch_rejected")
        self.assertEqual(imap.fetch.await_count, 2)


if __name__ == "__main__":
    unittest.main()
