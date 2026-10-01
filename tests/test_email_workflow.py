"""Mail workflow boundaries: ownership, approval versions and uncertain SMTP."""
import asyncio
import json
import sys
import tempfile
import time
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import aiosmtplib
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from db import connection, email_drafts as store
from routers.auth import get_current_user
from routers.email import router as legacy_router
from routers.email_workflow import router
from schemas.email_workflow import DraftFields
from services import email as mail_service
from services.email_worker import deliver
from tools.email_tools import create_email_draft, update_email_draft


class TestEmailWorkflow(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path_patch = patch.object(connection, "DATABASE_PATH", str(Path(self.directory.name) / "mail.db"))
        self.path_patch.start()
        await connection.init_db()
        async with connection.get_db() as db:
            await db.execute("INSERT INTO email_credentials(account_id, user_id, email_address, provider, imap_server, imap_port, smtp_server, smtp_port, encrypted_password, created_at) VALUES ('mail', 'alice', 'alice@example.com', 'other', 'imap.example.com', 993, 'smtp.example.com', 465, 'fake', '2026-10-01')")
            await db.commit()
        self.fields = {"account_id": "mail", "to_address": "bob@example.com", "subject": "会议资料", "body": "附件随后补发。"}
        self.app = FastAPI()
        self.app.include_router(router)
        self.app.include_router(legacy_router)
        self.owner = "alice"
        self.app.dependency_overrides[get_current_user] = lambda: {"id": self.owner}
        self.client = AsyncClient(transport=ASGITransport(app=self.app), base_url="http://test")
        self.notify_patch = patch("services.email_worker.event_bus.publish", new_callable=AsyncMock)
        self.notify_patch.start()

    async def asyncTearDown(self):
        await self.client.aclose()
        self.notify_patch.stop()
        self.path_patch.stop()
        self.directory.cleanup()

    async def draft(self, key="draft-create-1"):
        return await store.create_draft("alice", self.fields, key)

    async def claimed(self):
        draft = await self.draft()
        await store.enqueue_send("alice", draft["id"], 1, "send-request-1")
        return await store.claim_job("worker-1")

    def smtp(self):
        return SimpleNamespace(connect=AsyncMock(), starttls=AsyncMock(), login=AsyncMock(),
                               send_message=AsyncMock(return_value=({}, "queued")), quit=AsyncMock(), close=MagicMock())

    async def send_with(self, job, smtp):
        with patch("services.email_worker._get_credentials", new_callable=AsyncMock, return_value={
            "email_address": "alice@example.com", "smtp_server": "smtp.example.com", "smtp_port": 465, "password": "fake",
        }), patch("services.email_worker.aiosmtplib.SMTP", return_value=smtp):
            await deliver(job)
        return await store.get_job("alice", job["id"])

    async def test_creation_is_idempotent_and_never_sends(self):
        first, second = await asyncio.gather(self.draft(), self.draft())
        self.assertEqual(first["id"], second["id"])
        self.assertIsNone(first["last_job"])
        self.assertEqual(await store.list_jobs("alice"), [])

    async def test_owner_is_required_and_every_resource_is_private(self):
        draft = await self.draft()
        self.assertEqual(await store.list_drafts("bob"), [])
        with self.assertRaises(store.MailNotFound):
            await store.get_draft("bob", draft["id"])
        with self.assertRaises(store.MailNotFound):
            await store.update_draft("bob", draft["id"], 1, self.fields)
        with self.assertRaises(store.MailNotFound):
            await store.create_draft("bob", self.fields, "bob-create")
        with self.assertRaises(RuntimeError):
            await store.list_drafts("default")
        job = await store.enqueue_send("alice", draft["id"], 1, "owner-send-key")
        with self.assertRaises(store.MailNotFound):
            await store.get_job("bob", job["id"])
        self.assertEqual(await store.list_jobs("bob"), [])
        self.owner = "bob"
        self.assertEqual((await self.client.get(f"/api/email-drafts/{draft['id']}")).status_code, 404)
        self.assertEqual((await self.client.post(f"/api/email-send-jobs/{job['id']}/cancel")).status_code, 404)

    async def test_stale_edit_and_stale_approval_conflict(self):
        draft = await self.draft()
        updated = await store.update_draft("alice", draft["id"], 1, {**self.fields, "body": "用户编辑的正文"})
        self.assertEqual(updated["revision"], 2)
        with self.assertRaises(store.MailConflict):
            await store.update_draft("alice", draft["id"], 1, self.fields)
        response = await self.client.post(f"/api/email-drafts/{draft['id']}/send", json={"revision": 1, "idempotency_key": "stale-approval"})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(await store.list_jobs("alice"), [])

    async def test_concurrent_clicks_and_different_tabs_freeze_one_version(self):
        draft = await self.draft()
        first, second = await asyncio.gather(
            store.enqueue_send("alice", draft["id"], 1, "send-from-tab-a"),
            store.enqueue_send("alice", draft["id"], 1, "send-from-tab-b"),
        )
        self.assertEqual(first["id"], second["id"])
        self.assertEqual(first["body"], self.fields["body"])
        with self.assertRaises(store.MailConflict):
            await store.update_draft("alice", draft["id"], 1, {**self.fields, "body": "changed"})
        claimed = await store.claim_job("worker")
        self.assertIsNone(await store.claim_job("another-worker"))
        self.assertEqual(claimed["body"], first["body"])

    async def test_idempotency_key_cannot_be_used_for_another_draft(self):
        first = await self.draft()
        second = await self.draft("second-create")
        await store.enqueue_send("alice", first["id"], 1, "same-send-key")
        with self.assertRaises(store.MailConflict):
            await store.enqueue_send("alice", second["id"], 1, "same-send-key")

    async def test_invalid_recipient_and_empty_content_never_enter_queue(self):
        for index, address in enumerate(["not-mail", "alice@example.com,bob@example.com", "x@@example.com", "a\x00@example.com"]):
            draft = await store.create_draft("alice", {**self.fields, "to_address": address}, f"bad-draft-{index}")
            with self.assertRaises(ValueError):
                await store.enqueue_send("alice", draft["id"], 1, f"bad-send-{index}")
        with self.assertRaises(ValueError):
            DraftFields(subject="Hello\r\nBcc: victim@example.com")
        self.assertEqual(await store.list_jobs("alice"), [])

    async def test_ai_proposal_preserves_user_edits_and_conflicts_if_original_changes(self):
        original = await self.draft()
        proposal = await store.create_draft("alice", {**self.fields, "body": "AI 建议"}, "ai-proposal", proposal_for=original["id"], base_revision=1)
        self.assertEqual((await store.get_draft("alice", original["id"]))["body"], self.fields["body"])
        with self.assertRaises(store.MailConflict):
            await store.enqueue_send("alice", proposal["id"], 1, "proposal-send")
        await store.update_draft("alice", original["id"], 1, {**self.fields, "body": "新的用户编辑"})
        with self.assertRaises(store.MailConflict):
            await store.adopt_proposal("alice", proposal["id"], 1)
        self.assertEqual((await store.get_draft("alice", original["id"]))["body"], "新的用户编辑")

    async def test_adoption_is_explicit_and_does_not_send(self):
        original = await self.draft()
        proposal = await store.create_draft("alice", {**self.fields, "body": "建议正文"}, "adopt-proposal", proposal_for=original["id"], base_revision=1)
        adopted = await store.adopt_proposal("alice", proposal["id"], 1)
        self.assertEqual(adopted["revision"], 2)
        self.assertEqual(adopted["body"], "建议正文")
        self.assertEqual(await store.list_jobs("alice"), [])
        self.assertEqual((await store.get_draft("alice", proposal["id"]))["status"], "adopted")
        with self.assertRaises(store.MailConflict):
            await store.enqueue_send("alice", proposal["id"], 1, "adopted-proposal-send")
        replay = await store.create_draft("alice", {**self.fields, "body": "建议正文"}, "adopt-proposal", proposal_for=original["id"], base_revision=1)
        self.assertEqual(replay["id"], proposal["id"])

    async def test_cancel_only_before_worker_claim_and_retry_keeps_old_receipt(self):
        draft = await self.draft()
        first = await store.enqueue_send("alice", draft["id"], 1, "first-send")
        await store.cancel_job("alice", first["id"])
        second = await store.enqueue_send("alice", draft["id"], 1, "retry-send")
        self.assertNotEqual(second["id"], first["id"])
        await store.claim_job("worker")
        with self.assertRaises(store.MailConflict):
            await store.cancel_job("alice", second["id"])
        self.assertEqual((await store.get_job("alice", first["id"]))["status"], "cancelled")

    async def test_quit_failure_after_acceptance_does_not_report_send_failure(self):
        job = await self.claimed()
        smtp = self.smtp()
        smtp.quit.side_effect = ConnectionError("QUIT lost")
        result = await self.send_with(job, smtp)
        self.assertEqual(result["status"], "accepted")
        self.assertIsNotNone(result["accepted_at"])
        self.assertIsNone(result["error"])
        smtp.send_message.assert_awaited_once()
        msg = smtp.send_message.call_args.args[0]
        self.assertEqual(msg["Message-ID"], result["message_id"])
        self.assertEqual(msg["From"], "alice@example.com")
        replay = await store.enqueue_send("alice", job["draft_id"], 1, "new-click-after-reload")
        self.assertEqual(replay["id"], job["id"])
        self.assertIsNone(await store.claim_job("again"))

    async def test_disconnect_during_submission_is_unknown_and_cannot_auto_retry(self):
        job = await self.claimed()
        smtp = self.smtp()
        smtp.send_message.side_effect = ConnectionError("final acknowledgement lost")
        result = await self.send_with(job, smtp)
        self.assertEqual(result["status"], "unknown")
        self.assertEqual((await store.enqueue_send("alice", job["draft_id"], 1, "ambiguous-retry"))["id"], job["id"])
        self.assertIsNone(await store.claim_job("again"))

    async def test_connection_failure_is_known_not_submitted(self):
        job = await self.claimed()
        smtp = self.smtp()
        smtp.connect.side_effect = ConnectionError("offline")
        result = await self.send_with(job, smtp)
        self.assertEqual(result["status"], "failed")
        smtp.send_message.assert_not_awaited()

    async def test_changed_sender_cannot_submit_a_previously_confirmed_job(self):
        job = await self.claimed()
        smtp = self.smtp()
        with patch("services.email_worker._get_credentials", new_callable=AsyncMock, return_value={"email_address": "changed@example.com"}), patch("services.email_worker.aiosmtplib.SMTP", return_value=smtp):
            await deliver(job)
        self.assertEqual((await store.get_job("alice", job["id"]))["status"], "failed")
        smtp.connect.assert_not_awaited()
        smtp.send_message.assert_not_awaited()

    async def test_verification_uses_one_explicit_tls_upgrade_without_sending(self):
        @asynccontextmanager
        async def imap(*args, **kwargs):
            yield None, 0
        for port in [465, 587]:
            smtp = self.smtp()
            with patch.object(mail_service, "_imap_session", imap), patch.object(mail_service.aiosmtplib, "SMTP", return_value=smtp) as factory:
                self.assertTrue(await mail_service.verify_account("imap.example.com", 993, "smtp.example.com", port, "alice@example.com", "fake"))
            self.assertEqual(factory.call_args.kwargs["use_tls"], port == 465)
            self.assertFalse(factory.call_args.kwargs["start_tls"])
            self.assertEqual(smtp.starttls.await_count, 0 if port == 465 else 1)
            smtp.send_message.assert_not_awaited()

    async def test_rejection_is_known_failure_and_partial_result_is_preserved(self):
        job = await self.claimed()
        smtp = self.smtp()
        smtp.send_message.side_effect = aiosmtplib.SMTPDataError(550, "rejected")
        self.assertEqual((await self.send_with(job, smtp))["status"], "failed")
        await store.enqueue_send("alice", job["draft_id"], 1, "explicit-retry")
        next_job = await store.claim_job("worker-2")
        smtp = self.smtp()
        smtp.send_message.return_value = ({"bob@example.com": (550, "rejected")}, "accepted others")
        self.assertEqual((await self.send_with(next_job, smtp))["status"], "partial")

    async def test_expired_lease_fences_late_submit_and_preserves_unknown_result(self):
        job = await self.claimed()
        await store.submitting(job)
        async with connection.get_db() as db:
            await db.execute("UPDATE email_send_jobs SET lease_until = ? WHERE id = ?", (time.time() - 1, job["id"]))
            await db.commit()
        await store.expire_jobs()
        await store.finish_job(job, "accepted")
        self.assertEqual((await store.get_job("alice", job["id"]))["status"], "unknown")
        with self.assertRaises(store.MailConflict):
            await store.submitting(job)

    async def test_legacy_send_endpoint_cannot_bypass_draft_confirmation(self):
        response = await self.client.post("/api/emails/send", json=self.fields)
        self.assertEqual(response.status_code, 410)
        self.assertEqual(await store.list_jobs("alice"), [])

    async def test_graph_tools_save_references_and_propose_without_sending(self):
        runtime = SimpleNamespace(config={"configurable": {"owner": "alice"}}, tool_call_id="mail-tool-call")
        value = json.loads(await create_email_draft.coroutine(runtime, **self.fields))
        self.assertEqual(value["type"], "email_draft")
        self.assertNotIn("body", value)
        runtime.tool_call_id = "proposal-tool-call"
        proposed = json.loads(await update_email_draft.coroutine(runtime, draft_id=value["draft_id"], revision=1, **{**self.fields, "body": "更正式"}))
        self.assertEqual((await store.get_draft("alice", proposed["draft_id"]))["status"], "proposal")
        self.assertEqual(await store.list_jobs("alice"), [])
