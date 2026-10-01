"""Durable mail delivery; ambiguous submissions are never automatically replayed."""
import asyncio
import uuid
from contextlib import suppress
from datetime import datetime, timezone
from email.message import EmailMessage
from email.utils import format_datetime

import aiosmtplib

from core.event_bus import EventType, event_bus
from db import email_drafts as store
from services.email import _get_credentials
from utils.logger import get_logger

logger = get_logger("email-worker")


async def notify(user_id: str, draft_id: str) -> None:
    await event_bus.publish(EventType.EMAIL_UPDATED, {"user_id": user_id, "draft_id": draft_id})


async def deliver(job: dict) -> None:
    """Persist acceptance before closing the SMTP session, with a frozen sender."""
    smtp = None
    try:
        creds = await _get_credentials(job["user_id"], job["account_id"])
        if creds["email_address"] != job["from_address"]:
            raise ValueError("发件邮箱已经改变，请重新确认草稿")
        msg = EmailMessage()
        msg.set_content(job["body"])
        msg["From"], msg["To"], msg["Subject"] = job["from_address"], job["to_address"], job["subject"]
        msg["Message-ID"] = job["message_id"]
        msg["Date"] = format_datetime(datetime.now(timezone.utc))
        implicit_tls = creds["smtp_port"] == 465
        smtp = aiosmtplib.SMTP(hostname=creds["smtp_server"], port=creds["smtp_port"], use_tls=implicit_tls,
                             start_tls=False, timeout=30)
        await smtp.connect()
        if not implicit_tls:
            await smtp.starttls()
        await smtp.login(creds["email_address"], creds["password"])
        await store.submitting(job)
        refused, _response = await smtp.send_message(msg, timeout=600)
        await store.finish_job(job, "partial" if refused else "accepted", "部分收件人被服务器拒绝，请核对发送记录" if refused else None)
        # QUIT has no bearing on the already acknowledged delivery.
        with suppress(Exception):
            await smtp.quit(timeout=10)
    except asyncio.CancelledError:
        await store.finish_job(job, "unknown" if job["phase"] == "submitting" else "failed",
                               "服务停止，提交结果待确认；未自动重发" if job["phase"] == "submitting" else "服务停止，邮件尚未提交")
        raise
    except (aiosmtplib.SMTPRecipientsRefused, aiosmtplib.SMTPDataError, aiosmtplib.SMTPSenderRefused):
        await store.finish_job(job, "failed", "邮箱服务器明确拒绝本次邮件，请检查收件地址或邮件内容后重试")
    except aiosmtplib.SMTPAuthenticationError:
        await store.finish_job(job, "failed", "邮箱认证失败，请在设置中重新连接邮箱后重试")
    except Exception as error:
        ambiguous = job["phase"] == "submitting"
        await store.finish_job(job, "unknown" if ambiguous else "failed",
                               "提交时连接中断，邮件可能已发送；请核对邮箱，未自动重发" if ambiguous else "连接邮箱失败，邮件尚未提交；请检查邮箱配置后重试")
        logger.warning("Mail job %s stopped (%s, phase=%s)", job["id"], type(error).__name__, job["phase"])
    finally:
        if smtp is not None:
            with suppress(Exception):
                smtp.close()
        await notify(job["user_id"], job["draft_id"])


class EmailWorker:
    def __init__(self) -> None:
        self.runner_id = str(uuid.uuid4())
        self.task: asyncio.Task | None = None

    async def start(self) -> None:
        self.task = asyncio.create_task(self._loop(), name="email-worker")

    async def close(self) -> None:
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
            self.task = None

    async def _beat(self, job: dict, delivery: asyncio.Task) -> None:
        while not delivery.done():
            await asyncio.sleep(10)
            if not await store.heartbeat(job):
                delivery.cancel()
                return

    async def _loop(self) -> None:
        while True:
            try:
                await store.expire_jobs()
                job = await store.claim_job(self.runner_id)
                if job is None:
                    await asyncio.sleep(1)
                    continue
                await notify(job["user_id"], job["draft_id"])
                delivery = asyncio.create_task(deliver(job))
                beat = asyncio.create_task(self._beat(job, delivery))
                try:
                    await delivery
                finally:
                    beat.cancel()
                    await asyncio.gather(beat, return_exceptions=True)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Mail worker error; expired submissions will become visible, without replay")
                await asyncio.sleep(2)
