import email
import asyncio
from email.message import EmailMessage
from email.header import decode_header
from email.utils import parsedate_to_datetime
import aioimaplib
import aiosmtplib
from database import get_email_credentials
from utils.crypto import decrypt_password

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

async def verify_account(imap_server, imap_port, smtp_server, smtp_port, email_address, password):
    """Verifies that the provided IMAP and SMTP settings work."""
    # Test IMAP
    try:
        imap = aioimaplib.IMAP4_SSL(host=imap_server, port=imap_port)
        await imap.wait_hello_from_server()
        await imap.login(email_address, password)
        await imap.logout()
    except Exception as e:
        raise EmailServiceException(f"IMAP Verification failed: {str(e)}")

    # Test SMTP
    try:
        smtp = aiosmtplib.SMTP(hostname=smtp_server, port=smtp_port, use_tls=True)
        await smtp.connect()
        await smtp.login(email_address, password)
        await smtp.quit()
    except Exception as e:
        # Retry with STARTTLS if implicit SSL fails (common for 587 port)
        try:
            smtp = aiosmtplib.SMTP(hostname=smtp_server, port=smtp_port, use_tls=False)
            await smtp.connect()
            await smtp.starttls()
            await smtp.login(email_address, password)
            await smtp.quit()
        except Exception as e2:
            raise EmailServiceException(f"SMTP Verification failed: {str(e2)}")

    return True

async def _fetch_recent_for_account(creds: dict, limit: int) -> list:
    try:
        imap = aioimaplib.IMAP4_SSL(host=creds["imap_server"], port=creds["imap_port"])
        await imap.wait_hello_from_server()
        await imap.login(creds["email_address"], creds["password"])
        await imap.select("INBOX")
        
        status, messages = await imap.search("ALL")
        if status != "OK":
            raise EmailServiceException(f"Failed to search inbox for {creds['email_address']}.")
            
        msg_nums = messages[0].split()
        if not msg_nums:
            await imap.logout()
            return []
            
        recent_nums = msg_nums[-limit:]
        
        results = []
        for num in reversed(recent_nums):
            res_status, msg_data = await imap.fetch(num.decode(), "(BODY.PEEK[HEADER.FIELDS (SUBJECT FROM DATE)])")
            if res_status == "OK":
                raw_email = b""
                for response_part in msg_data:
                    if isinstance(response_part, bytearray):
                        raw_email += response_part
                    elif isinstance(response_part, tuple):
                        raw_email += response_part[1]
                
                msg = email.message_from_bytes(raw_email)
                
                date_str = msg.get("Date", "")
                try:
                    dt = parsedate_to_datetime(date_str)
                    iso_date = dt.isoformat()
                except Exception:
                    iso_date = date_str

                results.append({
                    "id": num.decode(),
                    "account_id": creds["account_id"],
                    "account_email": creds["email_address"],
                    "subject": _decode_header_str(msg.get("Subject", "")),
                    "from": _decode_header_str(msg.get("From", "")),
                    "date": iso_date
                })
        
        await imap.logout()
        return results
    except Exception as e:
        print(f"Error fetching from {creds['email_address']}: {e}")
        return []

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
    results = await asyncio.gather(*tasks)
    
    all_emails = []
    for res in results:
        all_emails.extend(res)
        
    # Sort all emails by date descending
    all_emails.sort(key=lambda x: x["date"], reverse=True)
    return all_emails[:limit]

async def read_email(user_id: str, email_id: str, account_id: str = None) -> dict:
    """Reads the full content of a specific email."""
    creds = await _get_credentials(user_id, account_id)
    
    try:
        imap = aioimaplib.IMAP4_SSL(host=creds["imap_server"], port=creds["imap_port"])
        await imap.wait_hello_from_server()
        await imap.login(creds["email_address"], creds["password"])
        await imap.select("INBOX")
        
        res_status, msg_data = await imap.fetch(str(email_id), "(RFC822)")
        if res_status != "OK":
            raise EmailServiceException(f"Failed to fetch email {email_id}.")
            
        raw_email = b""
        for response_part in msg_data:
            if isinstance(response_part, bytearray):
                raw_email += response_part
            elif isinstance(response_part, tuple):
                raw_email += response_part[1]
                
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
                content = msg.get_payload(decode=True).decode(msg.get_content_charset() or "utf-8", errors="replace")
                if content_type == "text/html":
                    html_body = content
                else:
                    body_text = content
            except Exception:
                body_text = str(msg.get_payload())

        if not html_body and body_text:
            html_body = f"<pre style='font-family: inherit; white-space: pre-wrap;'>{body_text}</pre>"
            
        if not body_text and html_body:
            import re
            import html as html_lib
            # Basic fallback to extract text from HTML for LLM context and UI preview
            text = re.sub(r'<style.*?>.*?</style>', ' ', html_body, flags=re.IGNORECASE | re.DOTALL)
            text = re.sub(r'<script.*?>.*?</script>', ' ', text, flags=re.IGNORECASE | re.DOTALL)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = html_lib.unescape(text)
            body_text = ' '.join(text.split())
        date_str = msg.get("Date", "")
        try:
            dt = parsedate_to_datetime(date_str)
            iso_date = dt.isoformat()
        except Exception:
            iso_date = date_str

        result = {
            "id": email_id,
            "account_id": creds["account_id"],
            "account_email": creds["email_address"],
            "subject": _decode_header_str(msg.get("Subject", "")),
            "from": _decode_header_str(msg.get("From", "")),
            "date": iso_date,
            "body": body_text,
            "html_body": html_body
        }
        
        await imap.logout()
        return result
    except Exception as e:
        raise EmailServiceException(f"Error reading email: {str(e)}")

async def draft_and_send_email(user_id: str, to: str, subject: str, body: str, account_id: str = None):
    """Sends an email using the user's SMTP credentials."""
    creds = await _get_credentials(user_id, account_id)
    
    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = subject
    msg['From'] = creds["email_address"]
    msg['To'] = to
    
    try:
        if creds["smtp_port"] == 465:
            # implicit TLS
            smtp = aiosmtplib.SMTP(hostname=creds["smtp_server"], port=creds["smtp_port"], use_tls=True)
            await smtp.connect()
            await smtp.login(creds["email_address"], creds["password"])
            await smtp.send_message(msg)
            await smtp.quit()
        else:
            # explicit TLS
            smtp = aiosmtplib.SMTP(hostname=creds["smtp_server"], port=creds["smtp_port"], use_tls=False)
            await smtp.connect()
            await smtp.starttls()
            await smtp.login(creds["email_address"], creds["password"])
            await smtp.send_message(msg)
            await smtp.quit()
    except Exception as e:
        raise EmailServiceException(f"Failed to send email: {str(e)}")
