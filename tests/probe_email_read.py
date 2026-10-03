"""Read-only inbox probe for the active user; prints no credentials or mail content.

Run: uv run python tests/probe_email_read.py
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from db.user import get_active_user
from services.email import get_recent_emails


async def main() -> int:
    logging.disable(logging.CRITICAL)
    user = await get_active_user()
    if user is None:
        print("No active user")
        return 1
    batch = await get_recent_emails(
        user["id"], limit=1, per_account=True, allow_all_failed=True
    )
    for account in batch.coverage:
        print(json.dumps({
            "provider_domain": account["email"].partition("@")[2],
            "succeeded": account["succeeded"],
            "count": account["count"],
            "error_code": account["error_code"],
            "error": account["error"],
        }, ensure_ascii=False))
    return int(any(not account["succeeded"] for account in batch.coverage))


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
