import email
import asyncio
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
from email.header import decode_header
from email.utils import parsedate_to_datetime
import aioimaplib
from aioimaplib.aioimaplib import Command, extract_exists, quoted
import aiosmtplib
from core.database import get_email_credentials
from utils.crypto import decrypt_password
from utils.logger import get_logger

logger = get_logger("email-service")

# NetEase (163/126/yeah) rejects SELECT unless RFC 2971 ID is sent after LOGIN.
# Official Java sample: https://help.mail.163.com/faqDetail.do?code=d7a5dc8471cd0c0e8b4b8f4f8e49998b374173cfe9171305fa1ce630d7f67ac2eda07326646e6eb0
IMAP_TIMEOUT = 30.0
IMAP_CLIENT_ID = {
    "name": "OpenAlfred",
    "version": "1.0.0",
    "vendor": "OpenAlfred",
    "support-email": "openalfred@localhost",
}
_ID_REQUIRED_MARKERS = ("163.com", "126.com", "yeah.net", "188.com", "qq.com")


def build_imap_id_arg(fields: dict[str, str]) -> str:
    """Build the ID argument NetEase/JavaMail send as a single atom.

    Wire form: ("name" "OpenAlfred" "version" "1.0.0" "vendor" "OpenAlfred" ...)
    aioimaplib.id() instead emits: ( "name" "OpenAlfred" ... ) with split parens,
    which 163 treats as missing ID and answers SELECT Unsafe Login.
    """
    parts: list[str] = []
    for key, value in fields.items():
        parts.append(quoted(str(key)))
        parts.append("NIL" if value is None else quoted(str(value)))
    return "(" + " ".join(parts) + ")"


def requires_imap_id(host: str, email_address: str = "") -> bool:
    blob = f"{host} {email_address}".lower()
    return any(marker in blob for marker in _ID_REQUIRED_MARKERS)


class EmailServiceException(Exception):
    pass

async def _get_credentials(user_id: str, account_id: str = None) -> dict:
    """Helper to fetch and decrypt email credentials for a given user."""
    creds_list = await get_email_credentials(user_id)
    if not creds_list:
        raise EmailServiceException("No email accounts configured for this user.")
        
    if account_id == "undefined" or account_id == "":
        account_id = None
        
    creds = creds_list[0] if account_id is None else next((c for c in creds_list if c["account_id"] == account_id), None)
    
    if not creds:
        raise EmailServiceException(f"Account with ID {account_id} not found.")
        
    plain_password = decrypt_password(creds["encrypted_password"])
    creds["password"] = plain_password
    return creds

def _decode_header_str(header_str) -> str:
    """Decodes email header text according to its encoding."""
    if not header_str:
        return ""
    parts = decode_header(header_str)
    decoded = ""
    for part, encoding in parts:
        if isinstance(part, bytes):
            decoded += part.decode(encoding or "utf-8", errors="replace")
        else:
            decoded += part
    return decoded


def _parse_email_date(date_str: str) -> str:
    if not date_str:
        return ""
    try:
        return parsedate_to_datetime(date_str).isoformat()
    except Exception:
        return date_str


def _date_sort_key(item: dict) -> datetime:
    raw = item.get("date") or ""
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        try:
            return parsedate_to_datetime(raw)
        except Exception:
            return datetime.min.replace(tzinfo=timezone.utc)


def _extract_rfc822_bytes(lines) -> bytes:
    """Pull RFC822/header literals out of an aioimaplib FETCH response."""
    chunks = []
    for part in lines or []:
        if isinstance(part, tuple) and len(part) >= 2:
            part = part[1]
        if not isinstance(part, (bytes, bytearray)):
            continue
        data = bytes(part)
        stripped = data.strip()
        if not stripped or stripped in (b")", b"OK", b"NO", b"BAD"):
            continue
        upper = stripped.upper()
        if upper.startswith(b"* ") or b" FETCH (" in upper or upper.endswith(b" FETCH"):
            continue
        chunks.append(data)
    return b"".join(chunks)


def _response_text(resp) -> str:
    parts = []
    for line in getattr(resp, "lines", None) or []:
        if isinstance(line, (bytes, bytearray)):
            parts.append(line.decode("utf-8", errors="replace"))
        else:
            parts.append(str(line))
    return " ".join(parts)


async def _send_imap_id(imap: aioimaplib.IMAP4, *, required: bool = False) -> None:
    """Send RFC 2971 ID in the single-atom form NetEase documents."""
    payload = build_imap_id_arg(IMAP_CLIENT_ID)
    cmd = Command("ID", imap.protocol.new_tag(), payload, loop=imap.protocol.loop)
    try:
        resp = await asyncio.wait_for(imap.protocol.execute(cmd), IMAP_TIMEOUT)
    except Exception as e:
        if required:
            raise EmailServiceException(f"IMAP ID command failed: {e}") from e
        logger.warning("IMAP ID command failed (continuing): %s", e)
        return

    if resp.result != "OK":
        detail = _response_text(resp)
        if required:
            raise EmailServiceException(f"IMAP ID rejected: {detail}")
        logger.warning("IMAP ID rejected (continuing): %s", detail)
        return
    logger.info("IMAP ID accepted")


@asynccontextmanager
async def _imap_session(
    host: str,
    port: int,
    email_address: str,
    password: str,
    *,
    select_inbox: bool = True,
):
    imap = aioimaplib.IMAP4_SSL(host=host, port=port, timeout=IMAP_TIMEOUT)
    id_required = requires_imap_id(host, email_address)
    try:
        await imap.wait_hello_from_server()
        login_resp = await imap.login(email_address, password)
        if login_resp.result != "OK":
            raise EmailServiceException(
                f"IMAP login failed for {email_address}: {_response_text(login_resp)}"
            )
        await _send_imap_id(imap, required=id_required)

        exists = None
        if select_inbox:
            select_resp = await imap.select("INBOX")
            if select_resp.result != "OK":
                raise EmailServiceException(
                    f"IMAP SELECT INBOX failed for {email_address}: {_response_text(select_resp)}"
                )
            exists = extract_exists(select_resp) or 0
        yield imap, exists
    finally:
        try:
            await imap.logout()
        except Exception:
            logger.debug("IMAP logout failed for %s", email_address, exc_info=True)


async def verify_account(imap_server, imap_port, smtp_server, smtp_port, email_address, password):
    """Verifies that the provided IMAP and SMTP settings work."""
    try:
        async with _imap_session(
            imap_server, imap_port, email_address, password, select_inbox=True
        ) as (_imap, _exists):
            pass
    except EmailServiceException:
        raise
    except Exception as e:
        raise EmailServiceException(f"IMAP Verification failed: {str(e)}")

    # Match delivery: TLS on 465, explicit STARTTLS on other configured ports.
    # Disable opportunistic STARTTLS so the library cannot upgrade twice.
    implicit_tls = smtp_port == 465
    smtp = aiosmtplib.SMTP(hostname=smtp_server, port=smtp_port, use_tls=implicit_tls,
                         start_tls=False, timeout=30)
    try:
        await smtp.connect()
        if not implicit_tls:
            await smtp.starttls()
        await smtp.login(email_address, password)
        with suppress(Exception):
            await smtp.quit(timeout=10)
    except Exception as e:
        raise EmailServiceException(f"SMTP Verification failed: {str(e)}") from e
    finally:
        with suppress(Exception):
            smtp.close()

    return True


async def _fetch_headers(imap: aioimaplib.IMAP4, seq: str) -> bytes:
    """Fetch message headers with a QQ-friendly fallback."""
    resp = await imap.fetch(seq, "(BODY.PEEK[HEADER.FIELDS (SUBJECT FROM DATE)])")
    if resp.result != "OK":
        resp = await imap.fetch(seq, "(RFC822.HEADER)")
    if resp.result != "OK":
        return b""
    return _extract_rfc822_bytes(resp.lines)

async def _fetch_recent_for_account(creds: dict, limit: int) -> list:
    address = creds["email_address"]
    try:
        async with _imap_session(
            creds["imap_server"],
            creds["imap_port"],
            address,
            creds["password"],
        ) as (imap, exists):
            if not exists:
                logger.info("IMAP inbox empty for %s", address)
                return []

            start = max(1, exists - limit + 1)
            results = []
            for num in range(exists, start - 1, -1):
                raw_email = await _fetch_headers(imap, str(num))
                if not raw_email:
                    logger.warning("IMAP FETCH returned no headers for %s seq=%s", address, num)
                    continue

                msg = email.message_from_bytes(raw_email)
                results.append({
                    "id": str(num),
                    "account_id": creds["account_id"],
                    "account_email": address,
                    "subject": _decode_header_str(msg.get("Subject", "")),
                    "from": _decode_header_str(msg.get("From", "")),
                    "date": _parse_email_date(msg.get("Date", "")),
                })
            logger.info("Fetched %d/%d recent emails from %s", len(results), exists, address)
            return results
    except Exception as e:
        logger.error("Error fetching from %s: %s", address, e, exc_info=True)
        raise EmailServiceException("邮箱读取失败，请检查连接和授权") from e


class EmailBatch(list):
    """Keep list callers compatible while exposing verified per-account coverage."""

    def __init__(self, items: list, coverage: list[dict]):
        super().__init__(items)
        self.coverage = coverage


async def get_recent_emails(user_id: str, limit: int = 10, account_ids: list[str] = None) -> list:
    """Fetches recent emails. If account_ids is None, fetches from all configured accounts."""
    creds_list = await get_email_credentials(user_id)
    if not creds_list:
        raise EmailServiceException("No email accounts configured for this user.")
        
    if account_ids:
        creds_list = [c for c in creds_list if c["account_id"] in account_ids]
        if not creds_list:
            raise EmailServiceException(f"None of the specified accounts were found.")

    for c in creds_list:
        c["password"] = decrypt_password(c["encrypted_password"])

    tasks = [_fetch_recent_for_account(c, limit) for c in creds_list]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    coverage = []
    all_emails = []
    for creds, result in zip(creds_list, results):
        success = not isinstance(result, BaseException)
        coverage.append(
            {
                "email": creds["email_address"],
                "succeeded": success,
                "count": len(result) if success else 0,
            }
        )
        if success:
            all_emails.extend(result)
    if not any(item["succeeded"] for item in coverage):
        raise EmailServiceException("所有邮箱均读取失败，请检查连接和授权")
    all_emails.sort(key=_date_sort_key, reverse=True)
    return EmailBatch(all_emails[:limit], coverage)


async def read_email(user_id: str, email_id: str, account_id: str = None) -> dict:
    """Reads the full content of a specific email."""
    creds = await _get_credentials(user_id, account_id)
    address = creds["email_address"]

    try:
        async with _imap_session(
            creds["imap_server"],
            creds["imap_port"],
            address,
            creds["password"],
        ) as (imap, _exists):
            res = await imap.fetch(str(email_id), "(RFC822)")
            if res.result != "OK":
                raise EmailServiceException(f"Failed to fetch email {email_id}.")

            raw_email = _extract_rfc822_bytes(res.lines)
            if not raw_email:
                raise EmailServiceException(f"Empty FETCH payload for email {email_id}.")

            msg = email.message_from_bytes(raw_email)

            body_text = ""
            html_body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    if content_type == "text/plain" and not body_text:
                        try:
                            body_text = part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", errors="replace")
                        except Exception:
                            pass
                    elif content_type == "text/html" and not html_body:
                        try:
                            html_body = part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", errors="replace")
                        except Exception:
                            pass
            else:
                content_type = msg.get_content_type()
                try:
                    payload = msg.get_payload(decode=True)
                    decoded = payload.decode(msg.get_content_charset() or "utf-8", errors="replace")
                    if content_type == "text/html":
                        html_body = decoded
                    else:
                        body_text = decoded
                except Exception:
                    body_text = str(msg.get_payload())

            if not html_body and body_text:
                html_body = f"<pre style='font-family: inherit; white-space: pre-wrap;'>{body_text}</pre>"

            if not body_text and html_body:
                import re
                import html as html_lib
                text = re.sub(r'<style.*?>.*?</style>', ' ', html_body, flags=re.IGNORECASE | re.DOTALL)
                text = re.sub(r'<script.*?>.*?</script>', ' ', text, flags=re.IGNORECASE | re.DOTALL)
                text = re.sub(r'<[^>]+>', ' ', text)
                text = html_lib.unescape(text)
                body_text = ' '.join(text.split())

            return {
                "id": email_id,
                "account_id": creds["account_id"],
                "account_email": address,
                "subject": _decode_header_str(msg.get("Subject", "")),
                "from": _decode_header_str(msg.get("From", "")),
                "date": _parse_email_date(msg.get("Date", "")),
                "body": body_text,
                "html_body": html_body,
            }
    except EmailServiceException:
        raise
    except Exception as e:
        raise EmailServiceException(f"Error reading email: {str(e)}") from e
