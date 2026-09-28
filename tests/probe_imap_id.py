"""Live IMAP ID probe against stored accounts. Never prints passwords.

Run: uv run python tests/probe_imap_id.py
"""

from __future__ import annotations

import asyncio
import imaplib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import aioimaplib  # noqa: E402
from aioimaplib.aioimaplib import Command  # noqa: E402

from core.database import get_email_credentials  # noqa: E402
from db.user import get_active_user  # noqa: E402
from services.email import (  # noqa: E402
    IMAP_CLIENT_ID,
    IMAP_TIMEOUT,
    build_imap_id_arg,
    _extract_rfc822_bytes,
)
from utils.crypto import decrypt_password  # noqa: E402


def _mask(address: str) -> str:
    local, _, domain = address.partition("@")
    if len(local) <= 2:
        return f"*@{domain}"
    return f"{local[:2]}***@{domain}"


def _text(resp) -> str:
    lines = getattr(resp, "lines", None)
    if lines is None:
        return str(resp)
    parts = []
    for line in lines:
        if isinstance(line, (bytes, bytearray)):
            parts.append(bytes(line).decode("utf-8", errors="replace"))
        else:
            parts.append(str(line))
    return " ".join(parts)[:300]


async def _try_aioimaplib(host, port, user, password, id_mode: str) -> str:
    imap = aioimaplib.IMAP4_SSL(host=host, port=port, timeout=IMAP_TIMEOUT)
    try:
        await imap.wait_hello_from_server()
        login = await imap.login(user, password)
        if login.result != "OK":
            return f"LOGIN FAIL: {_text(login)}"

        if id_mode == "library":
            resp = await imap.id(**IMAP_CLIENT_ID)
            id_note = f"ID(library)={resp.result} {_text(resp)}"
        elif id_mode == "netease":
            payload = build_imap_id_arg(IMAP_CLIENT_ID)
            cmd = Command("ID", imap.protocol.new_tag(), payload, loop=imap.protocol.loop)
            resp = await imap.protocol.execute(cmd)
            id_note = f"ID(netease)={resp.result} {_text(resp)}"
        else:
            id_note = "ID skipped"

        select = await imap.select("INBOX")
        return f"{id_note} | SELECT={select.result} {_text(select)}"
    except Exception as e:
        return f"EXC {type(e).__name__}: {e}"
    finally:
        try:
            await imap.logout()
        except Exception:
            pass


def _try_stdlib(host, port, user, password) -> str:
    try:
        conn = imaplib.IMAP4_SSL(host, port)
        typ, data = conn.login(user, password)
        if typ != "OK":
            return f"LOGIN FAIL: {typ} {data}"
        imaplib.Commands["ID"] = ("AUTH",)
        payload = build_imap_id_arg(IMAP_CLIENT_ID)
        typ, data = conn._simple_command("ID", payload)
        id_note = f"ID(stdlib)={typ} {data}"
        typ, data = conn.select("INBOX")
        conn.logout()
        return f"{id_note} | SELECT={typ} exists={data}"
    except Exception as e:
        return f"EXC {type(e).__name__}: {e}"


async def _fetch_one(host, port, user, password) -> str:
    imap = aioimaplib.IMAP4_SSL(host=host, port=port, timeout=IMAP_TIMEOUT)
    try:
        await imap.wait_hello_from_server()
        await imap.login(user, password)
        payload = build_imap_id_arg(IMAP_CLIENT_ID)
        cmd = Command("ID", imap.protocol.new_tag(), payload, loop=imap.protocol.loop)
        await imap.protocol.execute(cmd)
        select = await imap.select("INBOX")
        if select.result != "OK":
            return f"SELECT failed: {_text(select)}"
        from aioimaplib.aioimaplib import extract_exists

        exists = extract_exists(select) or 0
        if exists < 1:
            return "inbox empty (SELECT OK)"
        fetch = await imap.fetch(str(exists), "(BODY.PEEK[HEADER.FIELDS (SUBJECT FROM DATE)])")
        raw = _extract_rfc822_bytes(fetch.lines)
        subject = ""
        for line in raw.split(b"\n"):
            if line.lower().startswith(b"subject:"):
                subject = line.decode("utf-8", errors="replace").strip()
                break
        return f"SELECT OK exists={exists} fetch={fetch.result} {subject[:80]}"
    except Exception as e:
        return f"EXC {type(e).__name__}: {e}"
    finally:
        try:
            await imap.logout()
        except Exception:
            pass


async def main() -> int:
    active = await get_active_user()
    if not active:
        print("No active user in DB.")
        return 1
    creds = await get_email_credentials(active["id"])
    if not creds:
        print("No email accounts stored.")
        return 1

    print(f"Probing {len(creds)} account(s) for user {active.get('username')}")
    rc = 0
    for cred in creds:
        address = cred["email_address"]
        host = cred["imap_server"]
        port = int(cred["imap_port"] or 993)
        password = decrypt_password(cred["encrypted_password"])
        print(f"\n=== {_mask(address)} {host}:{port} ===")
        print("  aioimaplib.id() :", await _try_aioimaplib(host, port, address, password, "library"))
        print("  netease atom    :", await _try_aioimaplib(host, port, address, password, "netease"))
        print("  stdlib imaplib  :", await asyncio.to_thread(_try_stdlib, host, port, address, password))
        fetch = await _fetch_one(host, port, address, password)
        print("  fetch latest    :", fetch)
        if "Unsafe Login" in fetch or fetch.startswith("EXC") or "SELECT failed" in fetch:
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
