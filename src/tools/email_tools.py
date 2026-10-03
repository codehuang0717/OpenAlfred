from typing import Optional, Literal
from langchain.tools import ToolRuntime, tool
from langchain.messages import ToolMessage
from langgraph.types import Command
import json

# Local imports
from services.email import get_recent_emails as _get_recent_emails
from services.email import read_email as _read_email
from services.email import EmailServiceException
from tools.todos import _get_user_id
from services.tool_observations import (
    observed,
    failed,
    fields as display_fields,
    action,
    changes,
)


@tool
async def get_recent_emails(
    runtime: ToolRuntime, 
    limit: int = 10, 
    account_filter: Optional[str] = None
) -> Command:
    """Get recent emails from user's inbox. 
    If the user specifies an email provider or address (e.g., 'qq', 'gmail'), pass it as account_filter to strictly fetch from that account.
    Returns subject, sender, date, email ID, and available account info."""
    user_id = await _get_user_id(runtime)
    try:
        from core.database import get_email_credentials
        creds = await get_email_credentials(user_id)
        accounts = [{"account_id": c["account_id"], "email": c["email_address"], "provider": c["provider"]} for c in creds]
        
        target_account_ids = None
        if account_filter:
            filter_lower = account_filter.lower()
            target_account_ids = [
                c["account_id"] for c in accounts 
                if filter_lower in c["email"].lower() or filter_lower in c["provider"].lower()
            ]
            if not target_account_ids:
                observed(
                    None,
                    "没有匹配的邮箱，请检查筛选或邮箱设置",
                    outcome="blocked",
                    actions=[action("settings", "邮箱设置", "email")],
                )
                return Command(
                    update={
                        "messages": [
                            ToolMessage(
                                content=f"No email accounts found matching '{account_filter}'. Available accounts: {json.dumps(accounts)}",
                                tool_call_id=runtime.tool_call_id,
                            )
                        ]
                    }
                )
                
        emails = await _get_recent_emails(user_id=user_id, limit=limit, account_ids=target_account_ids)
        coverage = getattr(emails, "coverage", [])
        partial = any(not item["succeeded"] for item in coverage)
        observed(
            None,
            f"读取到 {len(emails)} 封最近邮件"
            + ("，部分邮箱读取失败" if partial else ""),
            outcome="partial" if partial else "completed" if emails else "empty",
            details=display_fields(
                邮箱范围="\n".join(
                    f"{item['email']} · {'成功' if item['succeeded'] else '读取失败'} · {item['count']} 封"
                    + (f" · {item['error']}" if item.get("error") else "")
                    for item in coverage
                )
                or "服务未提供逐邮箱统计",
                邮件="\n".join(
                    f"{item.get('subject') or '无主题'} · {item.get('from', '')} · {item.get('date', '')}"
                    for item in emails
                ),
            ),
            actions=[action("panel", "查看邮件", "emails")],
        )
        # Format for rich rendering in the frontend
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=json.dumps({
                            "type": "email_list",
                            "emails": emails,
                            "accounts": accounts,
                            "coverage": coverage,
                        }),
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )
    except EmailServiceException as e:
        failed(None, "邮件读取失败，请检查邮箱设置")
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Failed to fetch emails: {str(e)}. Please check your email configuration in settings.",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )
    except Exception as e:
        failed(None, "邮件读取失败，未取得可信结果")
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"An unexpected error occurred while fetching emails: {str(e)}.",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )


@tool
async def read_email(
    runtime: ToolRuntime, 
    email_id: str, 
    account_id: str
) -> Command:
    """Read the full content of a specific email. Requires both email_id and account_id from get_recent_emails."""
    user_id = await _get_user_id(runtime)
    try:
        email_data = await _read_email(user_id=user_id, email_id=email_id, account_id=account_id)

        observed(
            None,
            f"已读取：{email_data.get('subject') or '无主题'}",
            outcome="truncated"
            if len(email_data.get("body", "")) > 4000
            else "completed",
            details=display_fields(
                发件人=email_data.get("from"),
                收件人=email_data.get("to"),
                日期=email_data.get("date"),
                正文=email_data.get("body", "")[:4000],
            ),
            actions=[action("email", "打开完整邮件", email_id, account_id)],
        )
        # Remove html_body so we don't blow up the LLM token limit
        if "html_body" in email_data:
            del email_data["html_body"]
            
        # Truncate body if too long
        if "body" in email_data and len(email_data["body"]) > 4000:
            email_data["body"] = email_data["body"][:4000] + "\n...[Content truncated]..."
            
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=json.dumps({
                            "type": "email_content",
                            "email": email_data
                        }),
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )
    except EmailServiceException as e:
        failed(None, "邮件读取失败，请检查邮箱设置")
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Failed to read email {email_id}: {str(e)}",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )
    except Exception as e:
        failed(None, "邮件读取失败，未取得可信结果")
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"An unexpected error occurred while reading email: {str(e)}.",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )


@tool
async def get_email_accounts(runtime: ToolRuntime) -> Command:
    """Get a list of all configured email accounts and their account_ids."""
    user_id = await _get_user_id(runtime)
    try:
        from core.database import get_email_credentials
        creds = await get_email_credentials(user_id)
        accounts = [{"account_id": c["account_id"], "email": c["email_address"], "provider": c["provider"]} for c in creds]
        observed(
            None,
            f"已获取 {len(accounts)} 个可用邮箱" if accounts else "尚未配置邮箱",
            outcome="completed" if accounts else "empty",
            details=display_fields(
                邮箱="\n".join(f"{a['email']} · {a['provider']}" for a in accounts)
            ),
            actions=[action("settings", "邮箱设置", "email")],
        )
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=json.dumps(accounts),
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )
    except Exception as e:
        failed(None, "读取邮箱账户失败")
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Error fetching accounts: {str(e)}",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )


@tool
async def create_email_draft(runtime: ToolRuntime, account_id: str, to_address: str, subject: str, body: str) -> str:
    """Save an editable email draft, never send. Use a real account and recipient supplied by the user."""
    from db.email_drafts import create_draft
    from schemas.email_workflow import DraftFields
    from services.email_worker import notify
    from utils.auth_utils import require_runtime_user_id
    owner = require_runtime_user_id(runtime)
    fields = DraftFields(account_id=account_id, to_address=to_address, subject=subject, body=body)
    draft = await create_draft(owner, fields.model_dump(), f"tool:{runtime.tool_call_id}")
    observed(
        None,
        f"已保存草稿：{draft['subject'] or '无主题'}，尚未发送",
        details=display_fields(
            收件人=draft["to_address"],
            主题=draft["subject"],
            正文=draft["body"],
            版本=draft["revision"],
            状态={
                "draft": "草稿",
                "proposal": "修改建议",
                "adopted": "建议已采用",
                "queued": "等待发送",
                "sending": "正在提交邮件",
                "accepted": "已提交邮箱服务器",
                "partial": "部分收件人被拒绝",
                "failed": "发送失败",
                "unknown": "结果待确认",
                "cancelled": "已取消",
            }.get(draft["status"], "状态待核实"),
        ),
        actions=[action("email_draft", "打开草稿", draft["id"])],
    )
    await notify(owner, draft["id"])
    return json.dumps({"type": "email_draft", "draft_id": draft["id"], "subject": draft["subject"]}, ensure_ascii=False)


@tool
async def get_email_draft(runtime: ToolRuntime, draft_id: str) -> str:
    """Read the latest owned email draft and revision before proposing an edit."""
    from db.email_drafts import get_draft
    from utils.auth_utils import require_runtime_user_id
    draft = await get_draft(require_runtime_user_id(runtime), draft_id)
    observed(
        None,
        f"已读取草稿：{draft['subject'] or '无主题'}",
        details=display_fields(
            收件人=draft["to_address"],
            主题=draft["subject"],
            正文=draft["body"],
            版本=draft["revision"],
            状态={
                "draft": "草稿",
                "proposal": "修改建议",
                "adopted": "建议已采用",
                "queued": "等待发送",
                "sending": "正在提交邮件",
                "accepted": "已提交邮箱服务器",
                "partial": "部分收件人被拒绝",
                "failed": "发送失败",
                "unknown": "结果待确认",
                "cancelled": "已取消",
            }.get(draft["status"], "状态待核实"),
        ),
        actions=[action("email_draft", "打开草稿", draft["id"])],
    )
    return json.dumps({key: draft[key] for key in ("id", "account_id", "to_address", "subject", "body", "revision", "status")}, ensure_ascii=False)


@tool
async def update_email_draft(runtime: ToolRuntime, draft_id: str, revision: int, account_id: str,
                             to_address: str, subject: str, body: str) -> str:
    """Propose changes to the exact draft revision. Preserve the original until the user adopts the proposal."""
    from db.email_drafts import create_draft, get_draft
    from schemas.email_workflow import DraftFields
    from services.email_worker import notify
    from utils.auth_utils import require_runtime_user_id
    owner = require_runtime_user_id(runtime)
    original = await get_draft(owner, draft_id)
    fields = DraftFields(account_id=account_id, to_address=to_address, subject=subject, body=body)
    draft = await create_draft(owner, fields.model_dump(), f"tool:{runtime.tool_call_id}", proposal_for=draft_id, base_revision=revision)
    observed(
        None,
        f"修改建议已保存：{draft['subject'] or '无主题'}，等待采纳",
        outcome="proposal",
        details=changes(
            original,
            draft,
            {"to_address": "收件人变更", "subject": "主题变更", "body": "正文变更"},
        )
        + display_fields(
            发件邮箱="选择了不同邮箱"
            if original["account_id"] != draft["account_id"]
            else "沿用原草稿邮箱",
            基于版本=revision,
            说明="原草稿未覆盖，邮件未发送",
        ),
        actions=[action("email_draft", "查看修改建议", draft["id"])],
    )
    await notify(owner, draft["id"])
    return json.dumps({"type": "email_draft", "draft_id": draft["id"], "subject": draft["subject"]}, ensure_ascii=False)


email_tools = [
    get_recent_emails,
    read_email,
    get_email_accounts,
    create_email_draft,
    get_email_draft,
    update_email_draft,
]
